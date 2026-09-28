from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import os
from pathlib import Path
import time
from typing import  Dict, Optional, Union
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from h_sac_model import (RoutingDiscreteActor, RoutingTwinDiscreteCritic,hard_update,soft_update,  LocalHostDiscreteActor,LocalHostTwinCritic,)
from routing_replay_buffer import (RoutingReplayBatch,RoutingReplayBuffer,)
from host_replay_buffer import (HostReplayBatch, HostReplayBuffer,)

__all__ = (
    # Routing MASAC
    "RoutingMASACConfig",
    "RoutingMASAC",

    # Independent Local Host SAC
    "HostSACConfig",
    "LocalHostSAC",
    "atomic_checkpoint_text_write",
)


_CHECKPOINT_RETRYABLE_WINDOWS_ERROR_CODES = {
    5,     # 目标文件被映射/扫描时，Windows 也可能返回“拒绝访问”
    32,    # 文件正被其他进程使用
    33,    # 文件区域被锁定
    87,    # Windows ERROR_INVALID_PARAMETER
    1224,  # 文件存在打开的用户映射区段
}

_CHECKPOINT_RETRYABLE_ERRNOS = {
    13,  # EACCES
    22,  # EINVAL；部分 Windows 文件占用会被 Python 映射到此错误
}


def _is_retryable_checkpoint_error(
        error: BaseException,
) -> bool:
    """判断 checkpoint 写入失败是否属于 Windows 瞬时文件占用。"""

    winerror = getattr(error, "winerror", None)
    if winerror in _CHECKPOINT_RETRYABLE_WINDOWS_ERROR_CODES:
        return True

    errno_value = getattr(error, "errno", None)
    if errno_value in _CHECKPOINT_RETRYABLE_ERRNOS:
        return True

    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "error code: 5",
            "error code: 32",
            "error code: 33",
            "error code: 87",
            "error code: 1224",
            "errno 13",
            "errno 22",
            "errno 5",
            "errno 32",
            "errno 33",
            "errno 1224",
            "access is denied",
            "access denied",
            "invalid argument",
            "being used by another process",
            "user-mapped section open",
        )
    )


def _atomic_torch_save(
        checkpoint: Dict,
        file_path: Union[str, Path],
        max_attempts: int = 8,
        retry_delay_seconds: float = 0.25,
) -> None:
    """
    将 PyTorch checkpoint 原子写入目标路径。

    先写同目录中的唯一临时文件，再使用 ``os.replace`` 替换目标文件。
    Windows Defender、索引器或模型预览器短暂映射旧 checkpoint 时，
    对错误码 5、32、33、1224 做有限重试，避免训练因瞬时文件锁中断。
    """

    target_path = Path(file_path)
    target_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    max_attempts = max(int(max_attempts), 1)
    retry_delay_seconds = max(float(retry_delay_seconds), 0.0)
    last_error: Optional[BaseException] = None

    def build_temporary_path(attempt_index: int) -> Path:
        return target_path.with_name(
            f".{target_path.name}."
            f"{os.getpid()}."
            f"{time.time_ns()}."
            f"{attempt_index}.tmp"
        )

    temporary_path = build_temporary_path(0)

    try:
        # 唯一临时文件本身也可能被安全软件抢占；写入失败时换一个名字重试。
        for attempt_index in range(max_attempts):
            temporary_path = build_temporary_path(
                attempt_index
            )

            try:
                torch.save(
                    checkpoint,
                    temporary_path,
                )
                break

            except (OSError, RuntimeError) as error:
                last_error = error
                try:
                    temporary_path.unlink(
                        missing_ok=True
                    )
                except OSError:
                    pass

                if (
                        not _is_retryable_checkpoint_error(error)
                        or attempt_index + 1 >= max_attempts
                ):
                    raise RuntimeError(
                        "Checkpoint 临时文件写入失败："
                        f"{target_path}"
                    ) from error

                time.sleep(
                    retry_delay_seconds
                    * float(attempt_index + 1)
                )

        # 临时文件已经完整落盘。目标被映射占用时只重试原子替换，
        # 不重复序列化大型模型。
        for attempt_index in range(max_attempts):
            try:
                os.replace(
                    temporary_path,
                    target_path,
                )

                if (
                        not target_path.is_file()
                        or target_path.stat().st_size <= 0
                ):
                    raise RuntimeError(
                        "Checkpoint 原子写入后文件不存在或为空："
                        f"{target_path}"
                    )

                return

            except (OSError, RuntimeError) as error:
                last_error = error

                if (
                        not _is_retryable_checkpoint_error(error)
                        or attempt_index + 1 >= max_attempts
                ):
                    raise RuntimeError(
                        "Checkpoint 原子替换失败："
                        f"{target_path}"
                    ) from error

                time.sleep(
                    retry_delay_seconds
                    * float(attempt_index + 1)
                )

    finally:
        try:
            temporary_path.unlink(
                missing_ok=True
            )
        except OSError:
            # 临时文件也可能被安全软件短暂占用；它不会被当作正式
            # checkpoint，后续启动不读取该文件。
            pass

    raise RuntimeError(
        "Checkpoint 保存失败："
        f"{target_path}"
    ) from last_error


def atomic_checkpoint_text_write(
        text: str,
        file_path: Union[str, Path],
        *,
        encoding: str = "utf-8",
        max_attempts: int = 8,
        retry_delay_seconds: float = 0.25,
) -> None:
    """原子写入 checkpoint 配套文本，并重试 Windows 瞬时占用。"""

    if not isinstance(text, str):
        raise TypeError("Checkpoint 文本内容必须是 str。")
    if not text:
        raise ValueError("Checkpoint 文本内容不能为空。")

    target_path = Path(file_path).resolve()
    target_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    max_attempts = max(int(max_attempts), 1)
    retry_delay_seconds = max(float(retry_delay_seconds), 0.0)
    temporary_paths = []
    temporary_path: Optional[Path] = None
    last_error: Optional[BaseException] = None

    def build_temporary_path(attempt_index: int) -> Path:
        return target_path.with_name(
            f".{target_path.name}."
            f"{os.getpid()}."
            f"{time.time_ns()}."
            f"{attempt_index}.tmp"
        )

    try:
        # 先完整写入同目录唯一文件，避免直接截断仍被映射的正式 JSON。
        for attempt_index in range(max_attempts):
            temporary_path = build_temporary_path(
                attempt_index
            )
            temporary_paths.append(temporary_path)

            try:
                with temporary_path.open(
                        mode="x",
                        encoding=encoding,
                        newline="\n",
                ) as file_handle:
                    written_character_count = (
                        file_handle.write(text)
                    )
                    file_handle.flush()
                    os.fsync(file_handle.fileno())

                if written_character_count != len(text):
                    raise OSError(
                        "Checkpoint 文本未完整写入临时文件。"
                    )
                break

            except OSError as error:
                last_error = error
                try:
                    temporary_path.unlink(
                        missing_ok=True
                    )
                except OSError:
                    pass

                if (
                        not _is_retryable_checkpoint_error(error)
                        or attempt_index + 1 >= max_attempts
                ):
                    raise RuntimeError(
                        "Checkpoint 文本临时文件写入失败："
                        f"{target_path}"
                    ) from error

                time.sleep(
                    retry_delay_seconds
                    * float(attempt_index + 1)
                )

        if temporary_path is None:
            raise RuntimeError(
                "Checkpoint 文本临时文件未创建："
                f"{target_path}"
            )

        # 目标 JSON 被 PyCharm、索引器或安全软件占用时，只重试替换。
        for attempt_index in range(max_attempts):
            try:
                os.replace(
                    temporary_path,
                    target_path,
                )

                if (
                        not target_path.is_file()
                        or target_path.stat().st_size <= 0
                ):
                    raise RuntimeError(
                        "Checkpoint 文本原子写入后文件不存在或为空："
                        f"{target_path}"
                    )
                return

            except (OSError, RuntimeError) as error:
                last_error = error

                if (
                        not _is_retryable_checkpoint_error(error)
                        or attempt_index + 1 >= max_attempts
                ):
                    raise RuntimeError(
                        "Checkpoint 文本原子替换失败："
                        f"{target_path}"
                    ) from error

                time.sleep(
                    retry_delay_seconds
                    * float(attempt_index + 1)
                )

    finally:
        for candidate_path in temporary_paths:
            try:
                candidate_path.unlink(
                    missing_ok=True
                )
            except OSError:
                pass

    raise RuntimeError(
        "Checkpoint 文本保存失败："
        f"{target_path}"
    ) from last_error


# 超参数配置

@dataclass(frozen=True)
class RoutingMASACConfig:
    gamma: float = 0.99
    tau: float = 0.005

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4

    actor_hidden_dim: int = 256
    critic_hidden_dim: int = 256

    initial_alpha: float = 0.1
    target_entropy_ratio: float = 0.2

    max_grad_norm: Optional[float] = 10.0

    policy_update_interval: int = 1
    target_update_interval: int = 1

    device: Optional[str] = None
    seed: int = 42

@dataclass(frozen=True)
class HostSACConfig:
    """
    单个 DC 的 Independent Local Host SAC 配置。
    """

    gamma: float = 0.99
    tau: float = 0.005

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4

    actor_hidden_dim: int = 256
    critic_hidden_dim: int = 256

    initial_alpha: float = 0.1
    target_entropy_ratio: float = 0.2

    max_grad_norm: Optional[float] = 10.0

    policy_update_interval: int = 1
    target_update_interval: int = 1

    device: Optional[str] = None
    seed: int = 42

# 把 ReplayBatch 中 NumPy 数组转换后的张量
@dataclass(frozen=True)
class RoutingTensorBatch:
    agent_indices: torch.Tensor

    # Routing Actor input
    local_obs: torch.Tensor

    # CTDE Routing Critic input
    global_states: torch.Tensor

    actions: torch.Tensor
    rewards: torch.Tensor

    next_agent_indices: torch.Tensor
    next_local_obs: torch.Tensor
    next_global_states: torch.Tensor

    terminated: torch.Tensor
    truncated: torch.Tensor
    done: torch.Tensor

    # 当前 RoutingReplayBuffer 已不保存 forced transition，
    # 暂时保留该字段以兼容现有训练检查接口。
    is_forced_action: torch.Tensor

@dataclass(frozen=True)
class HostTensorBatch:
    """
    HostReplayBatch 转换为 GPU Tensor 后的内部结构。

    Host 层采用：
        One Job
        -> One Host Decision
        -> One terminal transition
    """

    host_obs: torch.Tensor

    actions: torch.Tensor
    rewards: torch.Tensor

    next_host_obs: torch.Tensor

    terminated: torch.Tensor
    truncated: torch.Tensor
    done: torch.Tensor

class RoutingMASAC:
    def __init__(
            self,
            local_obs_dim: int,
            global_state_dim: int,
            action_dim: int,
            num_agents: int,
            config: Optional[
                RoutingMASACConfig
            ] = None,
    ) -> None:

        local_obs_dim = int(local_obs_dim)
        global_state_dim = int(global_state_dim)
        action_dim = int(action_dim)
        num_agents = int(num_agents)

        # 未传入配置时创建默认配置
        if config is None:
            config = RoutingMASACConfig()
        self.config = config

        self.local_obs_dim = local_obs_dim
        self.global_state_dim = global_state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents

        self.device = resolve_training_device(
            config.device
        )

        # 若用户提供随机种子，则设置 PyTorch 随机种子
        if config.seed is not None:
            torch.manual_seed(
                int(config.seed)
            )
            # CUDA 存在时同时设置全部 CUDA 设备随机种子
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(
                    int(config.seed)
                )

        # 创建参数共享 Actor
        self.actor = RoutingDiscreteActor(
            local_obs_dim=local_obs_dim,
            action_dim=action_dim,
            num_agents=num_agents,
            hidden_dim=config.actor_hidden_dim,
        ).to(self.device)

        # 创建在线双 Critic
        self.critic = RoutingTwinDiscreteCritic(
            global_state_dim=(
                global_state_dim
            ),

            action_dim=(
                action_dim
            ),

            num_agents=(
                num_agents
            ),

            hidden_dim=(
                config.critic_hidden_dim
            ),
        ).to(
            self.device
        )

        # 创建目标双 Critic
        self.target_critic = RoutingTwinDiscreteCritic(
            global_state_dim=(
                global_state_dim
            ),

            action_dim=(
                action_dim
            ),

            num_agents=(
                num_agents
            ),

            hidden_dim=(
                config.critic_hidden_dim
            ),
        ).to(
            self.device
        )

        # 初始化时让目标 Critic 参数完全等于在线 Critic
        hard_update(
            target_network=self.target_critic,
            source_network=self.critic,
        )

        # 目标 Critic 不参与梯度更新
        for parameter in self.target_critic.parameters():
            parameter.requires_grad_(False)

        # 创建 Actor 优化器
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=float(config.actor_lr),
        )

        # 创建 Critic 优化器
        # TwinDiscreteCritic.parameters() 会包含 Q1 和 Q2 的参数
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=float(config.critic_lr),
        )

        # SAC 中 alpha 必须保持正数
        # 因此不直接优化 alpha，而是优化 log_alpha
        self.log_alpha = torch.tensor(
            math.log(float(config.initial_alpha)),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )

        # alpha 优化器只更新 log_alpha 这一个张量
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha],
            lr=float(config.alpha_lr),
        )

        # 记录已经执行了多少次网络更新
        self.update_step = 0

    # 返回当前温度系数 alpha
    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    # 根据单条环境观测选择一个普通动作
    def select_action(
            self,
            local_obs: np.ndarray,
            agent_index: int,
            deterministic: bool = False,
    ) -> int:
        """
        根据当前 Routing Observation 选择动作。

        所有 Routing action 均属于正常策略动作，
        不接受 action_mask。
        """

        local_obs_array = np.asarray(
            local_obs,
            dtype=np.float32,
        ).copy()

        agent_index = int(
            agent_index
        )

        local_obs_tensor = torch.as_tensor(
            local_obs_array,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        agent_index_tensor = torch.tensor(
            [agent_index],
            dtype=torch.long,
            device=self.device,
        )

        with torch.no_grad():

            action_probs, _, _ = (
                self.actor.get_policy(
                    local_obs=local_obs_tensor,
                    agent_indices=(
                        agent_index_tensor
                    ),
                )
            )

            if deterministic:

                action_tensor = torch.argmax(
                    action_probs,
                    dim=-1,
                )

            else:

                distribution = (
                    torch.distributions
                        .Categorical(
                        probs=action_probs
                    )
                )

                action_tensor = (
                    distribution.sample()
                )

        return int(
            action_tensor.item()
        )

    # 从 ReplayBuffer 采样并执行一次完整 SAC 更新
    def update(
            self,
            replay_buffer: RoutingReplayBuffer,
            batch_size: int,
    ) -> Dict[str, float]:
        batch_size = int(batch_size)

        # 默认排除环境强制动作经验，并且不放回采样
        replay_batch = replay_buffer.sample(
            batch_size=batch_size,
            include_forced_actions=False,
            replace=False,
        )

        # 把 NumPy batch 转换成设备上的 Tensor batch
        batch = self._batch_to_tensors(replay_batch)

        # 更新在线双 Critic
        critic_info = self._update_critic(batch)

        # Critic 更新完成后，更新步数加 1
        self.update_step += 1

        nan_value = torch.full(
            (),
            float("nan"),
            dtype=torch.float32,
            device=self.device,
        )

        # 先给 Actor 和 alpha 统计值设置默认值
        # 当本次未到更新间隔时，这些值会保持 NaN
        actor_loss_value = nan_value
        alpha_loss_value = nan_value
        entropy_value = nan_value
        target_entropy_value = nan_value

        # 到达策略更新间隔时，更新 Actor 和 alpha
        if (self.update_step % int(self.config.policy_update_interval) == 0):
            # 更新 Actor
            actor_info = self._update_actor(batch)

            # 更新温度参数 alpha
            alpha_info = self._update_alpha(batch)

            # 读取统计值
            actor_loss_value = actor_info["actor_loss"]
            entropy_value = actor_info["policy_entropy"]
            alpha_loss_value = alpha_info["alpha_loss"]
            target_entropy_value = alpha_info["target_entropy"]

        # 到达目标网络更新间隔时，执行软更新
        if (self.update_step % int(self.config.target_update_interval) == 0):
            soft_update(
                target_network=self.target_critic,
                source_network=self.critic,
                tau=float(self.config.tau),
            )

        update_info = {
            "update_step": float(self.update_step),
            "critic_loss": critic_info["critic_loss"],
            "q1_loss": critic_info["q1_loss"],
            "q2_loss": critic_info["q2_loss"],
            "mean_q1": critic_info["mean_q1"],
            "mean_q2": critic_info["mean_q2"],
            "mean_target_q": critic_info["mean_target_q"],
            "actor_loss": actor_loss_value,
            "alpha_loss": alpha_loss_value,
            "alpha": self.alpha.detach(),
            "policy_entropy": entropy_value,
            "target_entropy": target_entropy_value,
        }

        return update_info

    # 保存模型参数、优化器状态和更新步数
    def save(self,file_path: Union[str, Path]) -> None:

        file_path = Path(file_path)

        # 自动创建父目录。
        file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint = {
            # ======================================================
            # Algorithm identity
            #
            # 防止以后把 Host SAC checkpoint
            # 错误加载到 Routing MASAC。
            # ======================================================
            "algorithm_role": "routing_masac_ctde",

            "local_obs_dim": self.local_obs_dim,
            "global_state_dim": self.global_state_dim,
            "action_dim": self.action_dim,
            "num_agents": self.num_agents,

            "config": asdict(
                self.config
            ),

            "actor_state_dict":
                self.actor.state_dict(),

            "critic_state_dict":
                self.critic.state_dict(),

            "target_critic_state_dict":
                self.target_critic.state_dict(),

            "actor_optimizer_state_dict":
                self.actor_optimizer.state_dict(),

            "critic_optimizer_state_dict":
                self.critic_optimizer.state_dict(),

            "alpha_optimizer_state_dict":
                self.alpha_optimizer.state_dict(),

            "log_alpha":
                self.log_alpha.detach().cpu(),

            "update_step":
                self.update_step,
        }

        _atomic_torch_save(
            checkpoint,
            file_path,
        )

    # 从 checkpoint 恢复模型
    def load(self, file_path: Union[str, Path], load_optimizers: bool = True) -> None:

        # 读取 checkpoint 并映射到当前设备。
        checkpoint = torch.load(
            Path(file_path),
            map_location=self.device,
            weights_only=False,
        )

        # ==============================================================
        # Routing checkpoint identity validation
        #
        # 禁止：
        #   - Host checkpoint 错载进 Routing；
        #   - 不同 Observation 结构；
        #   - 不同 Routing action space；
        #   - 不同 Agent 数量
        #     的 checkpoint 被静默恢复。
        # ==============================================================

        role = checkpoint.get(
            "algorithm_role"
        )

        if (
                role
                != "routing_masac_ctde"
        ):
            raise RuntimeError(
                "当前 checkpoint 不是 "
                "Routing MASAC + CTDE："
                f"{role!r}"
            )

        dimension_checks = {
            "local_obs_dim":
                self.local_obs_dim,

            "global_state_dim":
                self.global_state_dim,

            "action_dim":
                self.action_dim,

            "num_agents":
                self.num_agents,
        }

        for field_name, current_value in (
                dimension_checks.items()
        ):
            saved_value = checkpoint.get(
                field_name
            )

            if (
                    saved_value is None
                    or int(saved_value)
                    != int(current_value)
            ):
                raise RuntimeError(
                    "Routing checkpoint "
                    "结构不兼容："
                    f"field={field_name}, "
                    f"saved={saved_value}, "
                    f"current={current_value}"
                )

        # 加载 Actor 参数。
        self.actor.load_state_dict(
            checkpoint["actor_state_dict"]
        )

        # 加载在线 Critic 参数。
        self.critic.load_state_dict(
            checkpoint["critic_state_dict"]
        )

        # 加载目标 Critic 参数。
        self.target_critic.load_state_dict(
            checkpoint["target_critic_state_dict"]
        )

        # 恢复 log_alpha 数值。
        with torch.no_grad():
            self.log_alpha.copy_(
                checkpoint["log_alpha"].to(
                    device=self.device,
                    dtype=torch.float32,
                )
            )

        # 根据参数决定是否恢复优化器。
        if load_optimizers:
            self.actor_optimizer.load_state_dict(
                checkpoint["actor_optimizer_state_dict"]
            )
            self.critic_optimizer.load_state_dict(
                checkpoint["critic_optimizer_state_dict"]
            )
            self.alpha_optimizer.load_state_dict(
                checkpoint["alpha_optimizer_state_dict"]
            )

        # 恢复更新步数。
        self.update_step = int(
            checkpoint.get("update_step", 0)
        )

    def train_mode(self) -> None:
        self.actor.train()
        self.critic.train()
        self.target_critic.eval()

    def eval_mode(self) -> None:
        self.actor.eval()
        self.critic.eval()
        self.target_critic.eval()

    # ==============================================================
    # Common Device Resolver
    #
    # RoutingMASAC 和 LocalHostSAC 都需要设备选择，
    # 但 Host SAC 不应该依赖 RoutingMASAC 类。
    #
    # 因此设备解析作为 module-level utility。
    # ==============================================================


    ##################### 辅助函数 ######################


    # 把 ReplayBatch 中的 NumPy 数组转换成张量
    def _batch_to_tensors(
            self,
            batch: RoutingReplayBatch,
    ) -> RoutingTensorBatch:

        agent_indices = torch.as_tensor(
            batch.agent_indices,
            dtype=torch.long,
            device=self.device,
        )
        local_obs = torch.as_tensor(
            batch.local_obs,
            dtype=torch.float32,
            device=self.device,
        )
        global_states = torch.as_tensor(
            batch.global_states,
            dtype=torch.float32,
            device=self.device,
        )
        # action_masks = torch.as_tensor(
        #     batch.action_masks,
        #     dtype=torch.bool,
        #     device=self.device,
        # )
        actions = torch.as_tensor(
            batch.actions,
            dtype=torch.long,
            device=self.device,
        )
        rewards = torch.as_tensor(
            batch.rewards,
            dtype=torch.float32,
            device=self.device,
        )
        next_agent_indices = torch.as_tensor(
            batch.next_agent_indices,
            dtype=torch.long,
            device=self.device,
        )
        next_local_obs = torch.as_tensor(
            batch.next_local_obs,
            dtype=torch.float32,
            device=self.device,
        )
        next_global_states = torch.as_tensor(
            batch.next_global_states,
            dtype=torch.float32,
            device=self.device,
        )
        # next_action_masks = torch.as_tensor(
        #     batch.next_action_masks,
        #     dtype=torch.bool,
        #     device=self.device,
        # )
        terminated = torch.as_tensor(
            batch.terminated,
            dtype=torch.float32,
            device=self.device,
        )
        truncated = torch.as_tensor(
            batch.truncated,
            dtype=torch.float32,
            device=self.device,
        )
        done = torch.as_tensor(
            batch.done,
            dtype=torch.float32,
            device=self.device,
        )
        is_forced_action = torch.as_tensor(
            batch.is_forced_action,
            dtype=torch.bool,
            device=self.device,
        )

        return RoutingTensorBatch(
            agent_indices=agent_indices,
            local_obs=local_obs,
            global_states=global_states,
            # action_masks=action_masks,
            actions=actions,
            rewards=rewards,
            next_agent_indices=next_agent_indices,
            next_local_obs=next_local_obs,
            next_global_states=next_global_states,
            # next_action_masks=next_action_masks,
            terminated=terminated,
            truncated=truncated,
            done=done,
            is_forced_action=is_forced_action,
        )

    # 更新在线双Q net
    def _update_critic(self, batch: RoutingTensorBatch) -> Dict[str, float]:

        # 使用目标网络计算TD目标作为监督标签使用
        with torch.no_grad():

            # 安全处理next_agent_index
            # 因为在终止经验中没有下一智能体会把next_agent_index记为 -1，但是F.one_hot不能接收-1，所以暂时换成0
            safe_next_agent_indices = torch.where(
                batch.done.to(dtype=torch.bool),
                torch.zeros_like(batch.next_agent_indices),
                batch.next_agent_indices,
            )

            # 终止经验的 next_action_mask 通常全 0，为了让 Actor.get_policy() 能正常运行，临时替换成全 1
            # safe_next_action_masks = torch.where(
            #     batch.done.to(dtype=torch.bool).unsqueeze(1),
            #     torch.ones_like(batch.next_action_masks),
            #     batch.next_action_masks,
            # )

            # 使用当前 Actor 计算下一决策状态的动作概率。
            next_action_probs, (
                next_action_log_probs
            ), _ = self.actor.get_policy(
                local_obs=batch.next_local_obs,
                agent_indices=(
                    safe_next_agent_indices
                ),
            )

            # 使用目标双 Critic 计算下一全局状态的 Q1 和 Q2。
            target_q1_all, target_q2_all = self.target_critic(
                global_state=batch.next_global_states,
                agent_indices=safe_next_agent_indices,
            )

            # 对每个动作取较小的目标 Q 值。
            target_min_q_all = torch.minimum(target_q1_all, target_q2_all,)

            # alpha 在 Critic 更新中只作为常数使用。
            alpha = self.alpha.detach()

            # 计算离散 SAC 的下一状态 soft value：
            # sum_a pi(a|o') * [min(Q1,Q2) - alpha*log pi(a|o')]
            next_soft_value = (
                    next_action_probs
                    * (
                            target_min_q_all
                            - alpha * next_action_log_probs
                    )
            ).sum(dim=-1)

            # done=True 时不再 bootstrap。
            not_done = 1.0 - batch.done

            # 计算 TD 目标。
            target_q = (
                    batch.rewards
                    + float(self.config.gamma)
                    * not_done
                    * next_soft_value
            )

        # 使用在线双 Critic 计算当前全局状态的全部动作 Q 值。
        current_q1_all, current_q2_all = self.critic(
            global_state=batch.global_states,
            agent_indices=batch.agent_indices,
        )

        # 从 Q1 的全部动作 Q 值中取出实际执行动作对应的值。
        current_q1 = current_q1_all.gather(dim=1, index=batch.actions.unsqueeze(1),).squeeze(1)

        # 从 Q2 中取出实际动作对应的值。
        current_q2 = current_q2_all.gather(dim=1, index=batch.actions.unsqueeze(1),).squeeze(1)

        # Q1 使用均方误差拟合 TD 目标。
        q1_loss = F.mse_loss(current_q1, target_q,)

        # Q2 同样使用均方误差。
        q2_loss = F.mse_loss(current_q2, target_q,)

        # 双 Critic 总损失是两个损失之和。
        critic_loss = q1_loss + q2_loss

        # 清空上一次 Critic 反向传播留下的梯度。
        self.critic_optimizer.zero_grad(set_to_none=True)

        # 反向传播计算 Critic 参数梯度。
        critic_loss.backward()

        # 如果设置了梯度裁剪上限，则裁剪 Critic 梯度范数。
        if self.config.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(
                self.critic.parameters(),
                max_norm=float(self.config.max_grad_norm),
            )

        # 根据梯度更新在线双 Critic 参数。
        self.critic_optimizer.step()

        # return {
        #     "critic_loss": float(critic_loss.detach().item()),
        #     "q1_loss": float(q1_loss.detach().item()),
        #     "q2_loss": float(q2_loss.detach().item()),
        #     "mean_q1": float(current_q1.detach().mean().item()),
        #     "mean_q2": float(current_q2.detach().mean().item()),
        #     "mean_target_q": float(target_q.detach().mean().item()),
        # }
        return {
            "critic_loss": critic_loss.detach(),
            "q1_loss": q1_loss.detach(),
            "q2_loss": q2_loss.detach(),
            "mean_q1": (current_q1.detach().mean()),
            "mean_q2": (current_q2.detach().mean()),
            "mean_target_q": (target_q.detach().mean()),
        }

    # 更新actor
    def _update_actor(self, batch: RoutingTensorBatch,) -> Dict[str, float]:

        # 计算当前局部观测下所有普通动作的概率和 log 概率。
        action_probs, action_log_probs, _ = (
            self.actor.get_policy(
                local_obs=batch.local_obs,
                agent_indices=batch.agent_indices,
            )
        )

        # Actor 更新不需要修改 Critic 参数
        # Q 值只作为评价当前策略动作好坏的固定信号
        with torch.no_grad():
            q1_all, q2_all = self.critic(
                global_state=batch.global_states,
                agent_indices=batch.agent_indices,
            )

            # 对每个动作使用较小的 Q 值
            min_q_all = torch.minimum(q1_all, q2_all,)

        # Actor 更新中 alpha 也作为常数使用
        alpha = self.alpha.detach()

        # 离散 SAC Actor 损失：
        # sum_a pi(a|o) * [alpha*log pi(a|o) - min(Q1,Q2)]
        actor_loss_per_sample = (
                action_probs
                * (
                        alpha * action_log_probs
                        - min_q_all
                )
        ).sum(dim=-1)

        # 对 batch 求平均。
        actor_loss = actor_loss_per_sample.mean()

        # 清空 Actor 旧梯度。
        self.actor_optimizer.zero_grad(set_to_none=True)

        # 反向传播计算 Actor 梯度。
        actor_loss.backward()

        # 如果启用梯度裁剪，则裁剪 Actor 梯度。
        if self.config.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                max_norm=float(self.config.max_grad_norm),
            )

        # 更新 Actor 参数。
        self.actor_optimizer.step()

        # 计算当前策略熵：-sum_a pi log pi。
        policy_entropy = -(action_probs * action_log_probs).sum(dim=-1)

        # return {
        #     "actor_loss": float(actor_loss.detach().item()),
        #     "policy_entropy": float(
        #         policy_entropy.detach().mean().item()
        #     ),
        # }
        return {
            "actor_loss": (actor_loss.detach()),
            "policy_entropy": (policy_entropy.detach().mean()),
        }

    # 更新温度系数alpha
    def _update_alpha(self, batch: RoutingTensorBatch,) -> Dict[str, float]:

        # alpha 更新时不需要更新 Actor，因此不记录 Actor 计算图。
        with torch.no_grad():
            action_probs, (
                action_log_probs
            ), _ = self.actor.get_policy(
                local_obs=batch.local_obs,
                agent_indices=(
                    batch.agent_indices
                ),
            )

            # 计算每条样本当前策略熵。
            policy_entropy = -(
                    action_probs
                    * action_log_probs
            ).sum(dim=-1)

            target_entropy_value = (
                    float(
                        self.config
                            .target_entropy_ratio
                    )
                    * math.log(
                float(
                    self.action_dim
                )
            )
            )

            target_entropy = torch.full_like(
                policy_entropy,
                fill_value=(
                    target_entropy_value
                ),
            )

            # 统计每条样本的合法动作数量。
            # valid_action_count = batch.action_masks.sum(
            #     dim=-1
            # ).to(dtype=torch.float32)

            # 合法动作数量至少为 1。
            # valid_action_count = valid_action_count.clamp_min(
            #     1.0
            # )

            # 目标熵随合法动作数量变化。
            # target_entropy = (
            #         float(self.config.target_entropy_ratio)
            #         * torch.log(valid_action_count)
            # )

        # alpha 损失：
        # 当实际熵低于目标熵时，梯度下降会增大 log_alpha；
        # 当实际熵高于目标熵时，梯度下降会减小 log_alpha。
        alpha_loss = (self.log_alpha * (policy_entropy.detach() - target_entropy.detach())).mean()

        # 清空 alpha 优化器旧梯度。
        self.alpha_optimizer.zero_grad(set_to_none=True)

        # 对 log_alpha 反向传播。
        alpha_loss.backward()

        # 更新 log_alpha。
        self.alpha_optimizer.step()

        # 防止极端情况下 alpha 数值爆炸或趋近完全为 0。
        # 这里限制的是 log_alpha，因此 alpha 大约位于 [exp(-20), exp(5)]。
        with torch.no_grad():
            self.log_alpha.clamp_(
                min=-20.0,
                max=5.0,
            )

        # 返回日志值。
        # return {
        #     "alpha_loss": float(alpha_loss.detach().item()),
        #     "target_entropy": float(
        #         target_entropy.detach().mean().item()
        #     ),
        # }
        return {
            "alpha_loss": (alpha_loss.detach()),
            "target_entropy": (target_entropy.detach().mean()),
        }

class LocalHostSAC:
    """
    单个 Edge DC 的独立 Local Host SAC。

    本类不依赖：
        PettingZoo
        agent_index
        centralized global state
        action_mask
    """

    def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            config: Optional[HostSACConfig] = None,
    ) -> None:

        if config is None:
            config = HostSACConfig()

        self.config = config

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)

        self.device = resolve_training_device(
            config.device
        )

        self.actor = LocalHostDiscreteActor(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=config.actor_hidden_dim,
        ).to(self.device)

        self.critic = LocalHostTwinCritic(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=config.critic_hidden_dim,
        ).to(self.device)

        self.target_critic = LocalHostTwinCritic(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=config.critic_hidden_dim,
        ).to(self.device)

        hard_update(
            target_network=self.target_critic,
            source_network=self.critic,
        )

        for parameter in self.target_critic.parameters():
            parameter.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=float(config.actor_lr),
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=float(config.critic_lr),
        )

        self.log_alpha = torch.tensor(
            math.log(float(config.initial_alpha)),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )

        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha],
            lr=float(config.alpha_lr),
        )

        self.update_step = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(
            self,
            host_obs: np.ndarray,
            deterministic: bool = False,
    ) -> int:

        obs = np.asarray(
            host_obs,
            dtype=np.float32,
        )

        if obs.shape != (self.obs_dim,):
            raise ValueError(
                "Host Observation shape 错误："
                f"expected={(self.obs_dim,)}, "
                f"actual={obs.shape}"
            )

        obs_tensor = torch.as_tensor(
            obs,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():

            probabilities, _, _ = (
                self.actor.get_policy(
                    obs_tensor
                )
            )

            if deterministic:
                action_tensor = torch.argmax(
                    probabilities,
                    dim=-1,
                )
            else:
                distribution = (
                    torch.distributions.Categorical(
                        probs=probabilities
                    )
                )

                action_tensor = distribution.sample()

        return int(
            action_tensor.item()
        )

    def update(
            self,
            replay_buffer: HostReplayBuffer,
            batch_size: int,
    ) -> Dict[str, Union[float, torch.Tensor]]:
        """
        从当前 DC 自己的 HostReplayBuffer
        采样并执行一次 Local Host SAC 更新。

        Host Replay 中所有经验都是：

            One Job
            -> One Host Decision
            -> terminal

        因而不会使用其他 Job 的 Host Observation
        作为 bootstrap next state。
        """

        batch_size = int(
            batch_size
        )

        replay_batch = replay_buffer.sample(
            batch_size=batch_size,
            replace=False,
        )

        batch = self._batch_to_tensors(
            replay_batch
        )

        critic_info = self._update_critic(
            batch
        )

        self.update_step += 1

        nan_value = torch.full(
            (),
            float("nan"),
            dtype=torch.float32,
            device=self.device,
        )

        actor_loss_value = nan_value
        alpha_loss_value = nan_value
        entropy_value = nan_value
        target_entropy_value = nan_value

        if (
                self.update_step
                % int(
            self.config.policy_update_interval
        )
                == 0
        ):
            actor_info = self._update_actor(
                batch
            )

            alpha_info = self._update_alpha(
                batch
            )

            actor_loss_value = (
                actor_info["actor_loss"]
            )

            entropy_value = (
                actor_info["policy_entropy"]
            )

            alpha_loss_value = (
                alpha_info["alpha_loss"]
            )

            target_entropy_value = (
                alpha_info["target_entropy"]
            )

        if (
                self.update_step
                % int(
            self.config.target_update_interval
        )
                == 0
        ):
            soft_update(
                target_network=(
                    self.target_critic
                ),

                source_network=(
                    self.critic
                ),

                tau=float(
                    self.config.tau
                ),
            )

        return {
            "update_step":
                torch.tensor(
                    float(self.update_step),
                    dtype=torch.float32,
                    device=self.device,
                ),

            "critic_loss":
                critic_info["critic_loss"],

            "q1_loss":
                critic_info["q1_loss"],

            "q2_loss":
                critic_info["q2_loss"],

            "mean_q1":
                critic_info["mean_q1"],

            "mean_q2":
                critic_info["mean_q2"],

            "mean_target_q":
                critic_info[
                    "mean_target_q"
                ],

            "actor_loss":
                actor_loss_value,

            "alpha_loss":
                alpha_loss_value,

            "alpha":
                self.alpha.detach(),

            "policy_entropy":
                entropy_value,

            "target_entropy":
                target_entropy_value,
        }

    def _batch_to_tensors(
            self,
            batch: HostReplayBatch,
    ) -> HostTensorBatch:

        return HostTensorBatch(
            host_obs=torch.as_tensor(
                batch.host_obs,
                dtype=torch.float32,
                device=self.device,
            ),

            actions=torch.as_tensor(
                batch.actions,
                dtype=torch.long,
                device=self.device,
            ),

            rewards=torch.as_tensor(
                batch.rewards,
                dtype=torch.float32,
                device=self.device,
            ),

            next_host_obs=torch.as_tensor(
                batch.next_host_obs,
                dtype=torch.float32,
                device=self.device,
            ),

            terminated=torch.as_tensor(
                batch.terminated,
                dtype=torch.float32,
                device=self.device,
            ),

            truncated=torch.as_tensor(
                batch.truncated,
                dtype=torch.float32,
                device=self.device,
            ),

            done=torch.as_tensor(
                batch.done,
                dtype=torch.float32,
                device=self.device,
            ),
        )

    def _update_critic(
            self,
            batch: HostTensorBatch,
    ) -> Dict[str, torch.Tensor]:
        """
        Local Host Critic update。

        当前 Host Transition 永远 terminal：

            done = 1

        所以 TD target：

            y = r

        不使用其他 Job 的 Host Observation bootstrap。
        """

        if not bool(
                torch.all(
                    batch.done
                    > 0.5
                ).item()
        ):
            raise RuntimeError(
                "HostReplayBuffer 中出现 "
                "non-terminal Host transition。"
            )

        target_q = (
            batch.rewards.detach()
        )

        q1_all, q2_all = (
            self.critic(
                batch.host_obs
            )
        )

        action_index = (
            batch.actions.unsqueeze(1)
        )

        q1 = q1_all.gather(
            dim=1,
            index=action_index,
        ).squeeze(1)

        q2 = q2_all.gather(
            dim=1,
            index=action_index,
        ).squeeze(1)

        q1_loss = F.mse_loss(
            q1,
            target_q,
        )

        q2_loss = F.mse_loss(
            q2,
            target_q,
        )

        critic_loss = (
                q1_loss
                + q2_loss
        )

        self.critic_optimizer.zero_grad(
            set_to_none=True
        )

        critic_loss.backward()

        if (
                self.config.max_grad_norm
                is not None
        ):
            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(),
                float(
                    self.config.max_grad_norm
                ),
            )

        self.critic_optimizer.step()

        return {
            "critic_loss":
                critic_loss.detach(),

            "q1_loss":
                q1_loss.detach(),

            "q2_loss":
                q2_loss.detach(),

            "mean_q1":
                q1.detach().mean(),

            "mean_q2":
                q2.detach().mean(),

            "mean_target_q":
                target_q.detach().mean(),
        }

    def _update_actor(
            self,
            batch: HostTensorBatch,
    ) -> Dict[str, torch.Tensor]:

        action_probs, action_log_probs, _ = (
            self.actor.get_policy(
                batch.host_obs
            )
        )

        with torch.no_grad():
            q1_values, q2_values = (
                self.critic(
                    batch.host_obs
                )
            )

            min_q_values = torch.minimum(
                q1_values,
                q2_values,
            )

        actor_loss = (
                action_probs
                * (
                        self.alpha.detach()
                        * action_log_probs
                        - min_q_values
                )
        ).sum(
            dim=-1
        ).mean()

        self.actor_optimizer.zero_grad(
            set_to_none=True
        )

        actor_loss.backward()

        if (
                self.config.max_grad_norm
                is not None
        ):
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                float(
                    self.config.max_grad_norm
                ),
            )

        self.actor_optimizer.step()

        policy_entropy = -(
                action_probs
                * action_log_probs
        ).sum(
            dim=-1
        ).mean()

        return {
            "actor_loss":
                actor_loss.detach(),

            "policy_entropy":
                policy_entropy.detach(),
        }

    def _update_alpha(
            self,
            batch: HostTensorBatch,
    ) -> Dict[str, torch.Tensor]:

        with torch.no_grad():
            action_probs, action_log_probs, _ = (
                self.actor.get_policy(
                    batch.host_obs
                )
            )

            policy_entropy = -(
                    action_probs
                    * action_log_probs
            ).sum(
                dim=-1
            )

            target_entropy = torch.full_like(
                policy_entropy,
                float(
                    self.config
                        .target_entropy_ratio
                )
                * math.log(
                    max(
                        self.action_dim,
                        1,
                    )
                ),
            )

        alpha_loss = (
                self.log_alpha
                * (
                        policy_entropy.detach()
                        - target_entropy.detach()
                )
        ).mean()

        self.alpha_optimizer.zero_grad(
            set_to_none=True
        )

        alpha_loss.backward()

        self.alpha_optimizer.step()

        with torch.no_grad():
            self.log_alpha.clamp_(
                min=-20.0,
                max=5.0,
            )

        return {
            "alpha_loss":
                alpha_loss.detach(),

            "target_entropy":
                target_entropy
                    .detach()
                    .mean(),
        }

    def train_mode(
            self,
    ) -> None:

        self.actor.train()
        self.critic.train()
        self.target_critic.eval()

    def eval_mode(
            self,
    ) -> None:

        self.actor.eval()
        self.critic.eval()
        self.target_critic.eval()

    def save(
            self,
            file_path: Union[
                str,
                Path,
            ],
    ) -> None:

        file_path = Path(
            file_path
        )

        file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint = {
            "algorithm_role":
                "local_host_sac",

            "obs_dim":
                self.obs_dim,

            "action_dim":
                self.action_dim,

            "config":
                asdict(
                    self.config
                ),

            "actor_state_dict":
                self.actor.state_dict(),

            "critic_state_dict":
                self.critic.state_dict(),

            "target_critic_state_dict":
                self.target_critic
                    .state_dict(),

            "actor_optimizer_state_dict":
                self.actor_optimizer
                    .state_dict(),

            "critic_optimizer_state_dict":
                self.critic_optimizer
                    .state_dict(),

            "alpha_optimizer_state_dict":
                self.alpha_optimizer
                    .state_dict(),

            "log_alpha":
                self.log_alpha
                    .detach()
                    .cpu(),

            "update_step":
                int(
                    self.update_step
                ),
        }

        _atomic_torch_save(
            checkpoint,
            file_path,
        )

    def load(
            self,
            file_path: Union[
                str,
                Path,
            ],
            load_optimizers: bool = True,
    ) -> None:

        checkpoint = torch.load(
            Path(file_path),
            map_location=self.device,
            weights_only=False,
        )

        role = checkpoint.get(
            "algorithm_role"
        )

        if (
                role
                != "local_host_sac"
        ):
            raise RuntimeError(
                "当前 checkpoint "
                "不是 Local Host SAC："
                f"{role!r}"
            )

        # ==============================================================
        # Host checkpoint dimension validation
        #
        # 每个 DC 的 Host 数可能不同，
        # 因此 action_dim 必须逐 DC 严格一致。
        # ==============================================================

        dimension_checks = {
            "obs_dim":
                self.obs_dim,

            "action_dim":
                self.action_dim,
        }

        for field_name, current_value in (
                dimension_checks.items()
        ):
            saved_value = checkpoint.get(
                field_name
            )

            if (
                    saved_value is None
                    or int(saved_value)
                    != int(current_value)
            ):
                raise RuntimeError(
                    "Local Host SAC checkpoint "
                    "结构不兼容："
                    f"field={field_name}, "
                    f"saved={saved_value}, "
                    f"current={current_value}"
                )

        self.actor.load_state_dict(
            checkpoint[
                "actor_state_dict"
            ]
        )

        self.critic.load_state_dict(
            checkpoint[
                "critic_state_dict"
            ]
        )

        self.target_critic.load_state_dict(
            checkpoint[
                "target_critic_state_dict"
            ]
        )

        with torch.no_grad():
            self.log_alpha.copy_(
                checkpoint[
                    "log_alpha"
                ].to(
                    device=self.device,
                    dtype=torch.float32,
                )
            )

        if load_optimizers:
            self.actor_optimizer.load_state_dict(
                checkpoint[
                    "actor_optimizer_state_dict"
                ]
            )

            self.critic_optimizer.load_state_dict(
                checkpoint[
                    "critic_optimizer_state_dict"
                ]
            )

            self.alpha_optimizer.load_state_dict(
                checkpoint[
                    "alpha_optimizer_state_dict"
                ]
            )

        self.update_step = int(
            checkpoint.get(
                "update_step",
                0,
            )
        )





# ==============================================================
# Common Device Resolver
#
# RoutingMASAC 和 LocalHostSAC 都需要设备选择，
# 但 Host SAC 不应该依赖 RoutingMASAC 类。
#
# 因此设备解析作为 module-level utility。
# ==============================================================

def resolve_training_device(
        configured_device: Optional[str],
) -> torch.device:

    device_name = (
        "cuda:0"
        if configured_device is None
        else str(
            configured_device
        )
    )

    # 当前项目要求 GPU-only training。
    if not device_name.startswith(
        "cuda"
    ):
        raise ValueError(
            "当前项目要求只使用 GPU 训练，"
            f"但配置的设备是：{device_name}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA 不可用，程序拒绝退回 CPU 训练。\n"
            "请检查：\n"
            "1. NVIDIA 驱动；\n"
            "2. CUDA 版 PyTorch；\n"
            "3. 当前 Conda / Python 环境。"
        )

    device = torch.device(
        device_name
    )

    gpu_index = (
        0
        if device.index is None
        else int(
            device.index
        )
    )

    gpu_count = (
        torch.cuda.device_count()
    )

    if gpu_index >= gpu_count:
        raise RuntimeError(
            f"指定了 GPU 编号 {gpu_index}，"
            f"但 PyTorch 只检测到 {gpu_count} 张 GPU。"
        )

    return device
