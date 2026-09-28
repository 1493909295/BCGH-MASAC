"""BGH-MASAC 三阶段 Guidance λ 调度。

第 8 步把引导强度从动作策略中抽离为纯 Episode-level schedule：

    Stage 1 Host Pretrain : λ = 0
    Stage 2 Routing Train : λ 从 start 线性升到 peak
    Stage 3 Joint Finetune: λ 从 peak 线性降到 end

λ 只缩放 action-level Bias，不改变 Actor 参数、Observation 或 Replay。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = (
    "GuidanceLambdaSchedule",
)


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
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in values
        ):
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
        progress = (
            episode - stage_start_episode
        ) / denominator

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
