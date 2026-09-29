"""Guidance schedule and guided routing policy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .bayesian_congestion_game import LocalSourceSignal, RoutingActionFeatures


# Guidance schedule
@dataclass(frozen=True)
class GuidanceLambdaSchedule:
    """按现有三阶段 Episode 边界计算 [0, 1] 内的 λ。"""

    stage2_start: float = 0.0
    stage2_peak: float = 1.0
    stage3_end: float = 0.3

    def __post_init__(self) -> None:
        values = (
            self.stage2_start,
            self.stage2_peak,
            self.stage3_end,
        )
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("Guidance λ 参数必须位于 [0, 1]。")
        if self.stage2_start > self.stage2_peak:
            raise ValueError("stage2_start 不能大于 stage2_peak。")
        if self.stage3_end > self.stage2_peak:
            raise ValueError("stage3_end 不能大于 stage2_peak。")

    @staticmethod
    def _interpolate(
        start: float,
        end: float,
        progress: float,
    ) -> float:
        progress = max(0.0, min(1.0, float(progress)))
        if progress == 0.0:
            return float(start)
        if progress == 1.0:
            return float(end)
        return float(start + (end - start) * progress)

    def lambda_for_episode(
        self,
        *,
        stage: str,
        episode: int,
        stage_start_episode: int,
        stage_end_episode: int,
    ) -> float:
        """返回指定 Episode 的 Guidance λ。"""

        stage = str(stage)
        episode = int(episode)
        stage_start_episode = int(stage_start_episode)
        stage_end_episode = int(stage_end_episode)
        if stage_end_episode < stage_start_episode:
            raise ValueError("训练阶段结束 Episode 不能早于开始 Episode。")

        if stage == "host_pretrain":
            return 0.0
        if stage not in {"routing_train", "joint_finetune"}:
            raise ValueError(f"未知训练阶段：{stage}")

        if stage_start_episode == stage_end_episode:
            return self.stage2_peak if stage == "routing_train" else self.stage3_end

        denominator = max(
            1,
            stage_end_episode - stage_start_episode,
        )
        progress = (episode - stage_start_episode) / denominator

        if stage == "routing_train":
            return self._interpolate(
                self.stage2_start,
                self.stage2_peak,
                progress,
            )

        if stage == "joint_finetune":
            return self._interpolate(
                self.stage2_peak,
                self.stage3_end,
                progress,
            )

        raise ValueError(f"未知训练阶段：{stage}")


# Guided policy
class GuidedRoutingPolicy:
    """将基础 Routing MASAC 与可选启发式 Bias 统一为一个动作接口。"""

    def __init__(
        self,
        base_policy: Any,
        action_target_dc_ids: Sequence[str],
        *,
        enabled: bool = False,
        bayesian_game: Optional[Any] = None,
    ) -> None:
        self.base_policy = base_policy
        self.action_target_dc_ids = tuple(
            str(target_dc_id) for target_dc_id in action_target_dc_ids
        )
        if len(self.action_target_dc_ids) != int(base_policy.action_dim):
            raise ValueError("action_target_dc_ids 数量必须等于 Routing Actor action_dim。")
        if len(set(self.action_target_dc_ids)) != len(self.action_target_dc_ids):
            raise ValueError("action_target_dc_ids 不能包含重复动作目标。")

        self.enabled = bool(enabled)
        self.bayesian_game = bayesian_game
        self.last_breakdowns = ()
        self._feasible_mask = None

    def build_action_context(
        self,
        source_dc_id: str,
        historical_feedback_provider: Optional[Any] = None,
        *,
        cpu_request: float = 0.0,
        gpu_request: float = 0.0,
        source_signals: Optional[Mapping[str, LocalSourceSignal]] = None,
    ) -> Mapping[str, RoutingActionFeatures]:
        """从历史反馈构造动作特征，不读取远端实时状态。

        缺少某个 Pair 历史样本时使用 0.5 中性先验；这不会伪造成功记录，
        只表示当前 Evidence 不足，真正的远端风险仍由 Bayesian confidence 抑制。
        """

        source_dc_id = str(source_dc_id)
        action_context: Dict[str, RoutingActionFeatures] = {}
        for target_dc_id in self.action_target_dc_ids:
            success_score = 0.5
            sla_score = 0.5
            delay_score = 0.5

            # 只有 target 是其他 Edge DC 时才查询历史 Neighbor Pair。
            if target_dc_id != source_dc_id and historical_feedback_provider is not None:
                feedback = historical_feedback_provider.get_feedback(
                    source_dc_id,
                    target_dc_id,
                )
                if feedback is not None:
                    success_score = float(feedback.get("success_ewma", 0.5))
                    sla_score = float(feedback.get("sla_success_ewma", 0.5))
                    # Provider 中 completion_time_ewma 越大表示耗时越长，
                    # 而 Guided Policy 的 delay_score 定义为越大越好。
                    delay_score = 1.0 - float(feedback.get("completion_time_ewma", 0.5))

            action_context[target_dc_id] = RoutingActionFeatures(
                target_dc_id=target_dc_id,
                success_score=max(0.0, min(1.0, success_score)),
                sla_score=max(0.0, min(1.0, sla_score)),
                delay_score=max(0.0, min(1.0, delay_score)),
                cpu_request=float(cpu_request),
                gpu_request=float(gpu_request),
                source_signal=(source_signals or {}).get(target_dc_id, LocalSourceSignal()),
            )

        return action_context

    def _bias_vector(
        self,
        action_bias: Mapping[Any, float],
    ) -> torch.Tensor:
        """将 action index 或 target DC key 的 Bias 转成 Actor 向量。"""

        if not isinstance(action_bias, Mapping):
            raise TypeError("action_bias 必须是动作索引/目标到 Bias 的映射。")

        bias_values = np.zeros(
            int(self.base_policy.action_dim),
            dtype=np.float32,
        )
        target_to_index = {
            target_dc_id: index for index, target_dc_id in enumerate(self.action_target_dc_ids)
        }

        for raw_key, raw_value in action_bias.items():
            if isinstance(raw_key, (int, np.integer)):
                action_index = int(raw_key)
            else:
                target_dc_id = str(raw_key)
                if target_dc_id not in target_to_index:
                    raise KeyError(f"Bias 包含未知 Routing action target：{target_dc_id}")
                action_index = target_to_index[target_dc_id]

            if not 0 <= action_index < len(bias_values):
                raise IndexError(f"Bias action index 越界：{action_index}")
            bias_value = float(raw_value)
            if not math.isfinite(bias_value):
                raise ValueError("action_bias 必须全部是有限数。")
            bias_values[action_index] = bias_value

        return torch.as_tensor(
            bias_values,
            dtype=torch.float32,
            device=self.base_policy.device,
        )

    def _select_with_bias(
        self,
        local_obs: np.ndarray,
        agent_index: int,
        action_bias: Mapping[Any, float],
        guidance_lambda: float,
        deterministic: bool,
    ) -> int:
        """仅在需要指导时执行 ``logits + λ × bias`` 的动作采样。

        ``guidance_lambda`` 是调度器给出的归一化强度；具体 Bias 尺度仍由
        Bayesian Game 的 ``BCGH_GUIDANCE_SCALE`` 控制，避免把训练阶段调度
        与效用量纲耦合在一起。
        """

        local_obs_array = np.asarray(
            local_obs,
            dtype=np.float32,
        ).copy()
        local_obs_tensor = torch.as_tensor(
            local_obs_array,
            dtype=torch.float32,
            device=self.base_policy.device,
        ).unsqueeze(0)
        agent_index_tensor = torch.tensor(
            [int(agent_index)],
            dtype=torch.long,
            device=self.base_policy.device,
        )

        guidance_lambda = float(guidance_lambda)
        if not math.isfinite(guidance_lambda) or not 0.0 <= guidance_lambda <= 1.0:
            raise ValueError("guidance_lambda 必须位于 [0, 1]。")

        with torch.no_grad():
            # 只在本适配层加 Bias；基础 Actor 参数和输入保持不变。
            logits = self.base_policy.actor.forward(
                local_obs=local_obs_tensor,
                agent_indices=agent_index_tensor,
            )
            guided_logits = logits + guidance_lambda * self._bias_vector(action_bias).unsqueeze(0)
            if self._feasible_mask is not None:
                mask = torch.as_tensor(
                    self._feasible_mask, dtype=torch.bool, device=self.base_policy.device
                ).unsqueeze(0)
                if not bool(mask.any()):
                    raise RuntimeError("No statically feasible routing action")
                guided_logits = guided_logits.masked_fill(~mask, float("-inf"))
            action_probs = F.softmax(
                guided_logits,
                dim=-1,
            )

            if deterministic:
                action_tensor = torch.argmax(
                    action_probs,
                    dim=-1,
                )
            else:
                action_tensor = torch.distributions.Categorical(probs=action_probs).sample()

        return int(action_tensor.item())

    def select_action(
        self,
        local_obs: np.ndarray,
        agent_index: int,
        source_dc_id: str,
        *,
        action_context: Optional[Mapping[str, Any]] = None,
        action_bias: Optional[Mapping[Any, float]] = None,
        guidance_lambda: float = 0.3,
        deterministic: bool = False,
    ) -> int:
        """统一选择 Routing Action。

        ``action_bias`` 优先级高于 ``bayesian_game`` 自动计算结果，便于
        测试和未来接入其他启发式提供者。默认关闭时直接调用基础策略，
        不改变原有采样实现。``guidance_lambda`` 只影响启用指导时的
        action-level Bias，不进入 Observation、Replay 或 Actor 参数更新。
        """

        self.last_breakdowns = ()
        self._feasible_mask = None
        if not self.enabled and action_bias is None:
            return self.base_policy.select_action(
                local_obs=local_obs,
                agent_index=agent_index,
                deterministic=deterministic,
            )

        if action_bias is None:
            if self.bayesian_game is None:
                raise RuntimeError("Guided Policy 已启用，但未提供 Bayesian Game 或 action_bias。")
            if action_context is None:
                raise RuntimeError("Guided Policy 已启用，但缺少候选动作 action_context。")
            self.last_breakdowns = self.bayesian_game.evaluate_actions(
                str(source_dc_id),
                action_context,
            )
            by_target = {item.target_dc_id: item for item in self.last_breakdowns}
            action_bias = {target: item.bias for target, item in by_target.items()}
            self._feasible_mask = [
                by_target[target].feasible for target in self.action_target_dc_ids
            ]

        return self._select_with_bias(
            local_obs=local_obs,
            agent_index=agent_index,
            action_bias=action_bias,
            guidance_lambda=guidance_lambda,
            deterministic=deterministic,
        )
