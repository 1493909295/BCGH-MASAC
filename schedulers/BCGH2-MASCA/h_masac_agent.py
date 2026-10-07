"""BCGH2-MASCA networks and routing/Host SAC agents."""

from __future__ import annotations

import math
import sys
from functools import lru_cache
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import config as conf

from .experience import (
    HostReplayBatch,
    HostReplayBuffer,
    RoutingReplayBatch,
    RoutingReplayBuffer,
)

DEFAULT_EPS = 1e-8


def build_agent_one_hot(agent_indices: torch.Tensor, num_agents: int) -> torch.Tensor:

    num_agents = int(num_agents)
    agent_indices = agent_indices.to(dtype=torch.long)

    one_hot = F.one_hot(
        agent_indices,
        num_classes=num_agents,
    )
    one_hot = one_hot.to(dtype=torch.float32)
    return one_hot


def initialize_linear_layer(layer: nn.Linear, gain: float = 1.0) -> None:

    nn.init.orthogonal_(layer.weight, gain=float(gain))

    if layer.bias is not None:
        nn.init.constant_(
            layer.bias,
            0.0,
        )


class RoutingDiscreteActor(nn.Module):
    """
    所有边缘智能体共用同一个 Actor。
    为了让共享 Actor 知道当前是哪一个智能体在决策，
    网络输入中除了 local_obs，还会拼接 agent one-hot。

    网络输入 shape：
        local_obs      -> (batch_size, local_obs_dim)
        agent_indices  -> (batch_size,)

    网络输出 shape：
        logits         -> (batch_size, action_dim)       网络原始偏好分数
        probabilities  -> (batch_size, action_dim)       动作概率
    """

    def __init__(
        self,
        local_obs_dim: int,
        action_dim: int,
        num_agents: int,
        hidden_dim: int = conf.ROUTING_ACTOR_HIDDEN_DIM,
    ) -> None:
        super().__init__()

        local_obs_dim = int(local_obs_dim)
        action_dim = int(action_dim)
        num_agents = int(num_agents)
        hidden_dim = int(hidden_dim)

        self.local_obs_dim = local_obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim

        actor_input_dim = local_obs_dim + num_agents

        self.fc1 = nn.Linear(actor_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, action_dim)

        initialize_linear_layer(self.fc1, gain=nn.init.calculate_gain("relu"))
        initialize_linear_layer(self.fc2, gain=nn.init.calculate_gain("relu"))
        initialize_linear_layer(
            self.output_layer,
            gain=conf.ROUTING_ACTOR_GAIN,
        )

    def forward(
        self, local_obs: torch.Tensor, agent_indices: torch.Tensor
    ) -> torch.Tensor:

        agent_one_hot = build_agent_one_hot(
            agent_indices=agent_indices, num_agents=self.num_agents
        )

        agent_one_hot = agent_one_hot.to(device=local_obs.device)

        agent_one_hot = agent_one_hot.to(dtype=local_obs.dtype)

        actor_input = torch.cat([local_obs, agent_one_hot], dim=-1)

        hidden = F.relu(self.fc1(actor_input))
        hidden = F.relu(self.fc2(hidden))

        logits = self.output_layer(hidden)

        return logits

    def get_policy(
        self,
        local_obs: torch.Tensor,
        agent_indices: torch.Tensor,
        eps: float = DEFAULT_EPS,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        logits = self.forward(
            local_obs=local_obs,
            agent_indices=agent_indices,
        )

        action_probs = F.softmax(
            logits,
            dim=-1,
        )

        action_log_probs = torch.log(action_probs.clamp_min(float(eps)))

        return (
            action_probs,
            action_log_probs,
            logits,
        )

    def sample_action(
        self,
        local_obs: torch.Tensor,
        agent_indices: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:

        action_probs, action_log_probs, _ = self.get_policy(
            local_obs=local_obs,
            agent_indices=agent_indices,
        )

        distribution = torch.distributions.Categorical(probs=action_probs)

        sampled_actions = distribution.sample()

        selected_log_probs = action_log_probs.gather(
            dim=1,
            index=sampled_actions.unsqueeze(1),
        ).squeeze(1)

        return (
            sampled_actions,
            selected_log_probs,
            action_probs,
            action_log_probs,
        )


class RoutingDiscreteQNetwork(nn.Module):
    """
    Routing MASAC 的 centralized discrete Q network。

    输入：
        global_state
            CTDE 训练阶段的 Routing centralized state

        agent_indices
            当前正在做 Routing decision 的 Edge DC identity

    网络内部：
        global_state
            +
        agent one-hot

    输出：
        当前 Routing Agent 对全部 Routing actions 的 Q values。

    注意：
        1. 这是 Routing 层 Critic；
        2. 不属于 Local Host SAC；
        3. 不读取 Host Observation；
        4. 不与任何 LocalHostQNetwork 共享参数。
    """

    def __init__(
        self,
        global_state_dim: int,
        action_dim: int,
        num_agents: int,
        hidden_dim: int = conf.ROUTING_CRITIC_HIDDEN_DIM,
    ) -> None:
        super().__init__()

        self.global_state_dim = int(global_state_dim)

        self.action_dim = int(action_dim)

        self.num_agents = int(num_agents)

        self.hidden_dim = int(hidden_dim)

        if self.global_state_dim <= 0:
            raise ValueError("Routing global_state_dim 必须 > 0")

        if self.action_dim <= 0:
            raise ValueError("Routing action_dim 必须 > 0")

        if self.num_agents <= 0:
            raise ValueError("Routing num_agents 必须 > 0")

        critic_input_dim = self.global_state_dim + self.num_agents

        self.fc1 = nn.Linear(
            critic_input_dim,
            self.hidden_dim,
        )

        self.fc2 = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
        )

        self.output_layer = nn.Linear(
            self.hidden_dim,
            self.action_dim,
        )

        initialize_linear_layer(
            self.fc1,
            gain=nn.init.calculate_gain("relu"),
        )

        initialize_linear_layer(
            self.fc2,
            gain=nn.init.calculate_gain("relu"),
        )

        initialize_linear_layer(
            self.output_layer,
            gain=conf.ROUTING_CRITIC_GAIN,
        )

    def forward(
        self,
        global_state: torch.Tensor,
        agent_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        返回当前 Routing Agent 对全部 Routing actions
        的 centralized Q values。
        """

        agent_one_hot = build_agent_one_hot(
            agent_indices=(agent_indices),
            num_agents=(self.num_agents),
        )

        agent_one_hot = agent_one_hot.to(
            device=global_state.device,
            dtype=global_state.dtype,
        )

        critic_input = torch.cat(
            [
                global_state,
                agent_one_hot,
            ],
            dim=-1,
        )

        hidden = F.relu(self.fc1(critic_input))

        hidden = F.relu(self.fc2(hidden))

        return self.output_layer(hidden)


class RoutingTwinDiscreteCritic(nn.Module):
    """
    Routing MASAC 的 centralized Twin-Q Critic。

    两个 Q 网络：

        Q1(global_state, agent_id)
        Q2(global_state, agent_id)

    完全服务于 Routing MASAC + CTDE。

    不与任何 Local Host SAC Critic 共享：
        - 参数
        - Optimizer
        - Target Critic
        - ReplayBuffer
    """

    def __init__(
        self,
        global_state_dim: int,
        action_dim: int,
        num_agents: int,
        hidden_dim: int = conf.ROUTING_CRITIC_HIDDEN_DIM,
    ) -> None:
        super().__init__()

        self.q1 = RoutingDiscreteQNetwork(
            global_state_dim=(global_state_dim),
            action_dim=(action_dim),
            num_agents=(num_agents),
            hidden_dim=(hidden_dim),
        )

        self.q2 = RoutingDiscreteQNetwork(
            global_state_dim=(global_state_dim),
            action_dim=(action_dim),
            num_agents=(num_agents),
            hidden_dim=(hidden_dim),
        )

    def forward(
        self,
        global_state: torch.Tensor,
        agent_indices: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        同时返回 Q1 / Q2 对全部 Routing actions 的估计。
        """

        q1_values = self.q1(
            global_state=(global_state),
            agent_indices=(agent_indices),
        )

        q2_values = self.q2(
            global_state=(global_state),
            agent_indices=(agent_indices),
        )

        return (
            q1_values,
            q2_values,
        )


def hard_update(target_network: nn.Module, source_network: nn.Module) -> None:
    target_network.load_state_dict(source_network.state_dict())


def soft_update(
    target_network: nn.Module,
    source_network: nn.Module,
    tau: float,
) -> None:

    tau = float(tau)
    with torch.no_grad():
        for target_parameter, source_parameter in zip(
            target_network.parameters(),
            source_network.parameters(),
        ):
            target_parameter.mul_(1.0 - tau)
            target_parameter.add_(
                source_parameter,
                alpha=tau,
            )


class LocalHostDiscreteActor(nn.Module):
    """
    单个 Edge DC 的 Local Host SAC Actor。

    输入：
        当前 Job + 当前 Local DC + 当前 DC 全部真实 Host

    输出：
        当前 DC 内各 Host 的离散动作概率。

    注意：
        1. 不属于 PettingZoo；
        2. 不使用 agent one-hot；
        3. 不使用 global state；
        4. 不使用 action mask；
        5. 不使用 Host padding。
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = conf.HOST_ACTOR_HIDDEN_DIM,
    ) -> None:
        super().__init__()

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)

        self.fc1 = nn.Linear(
            self.obs_dim,
            self.hidden_dim,
        )

        self.fc2 = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
        )

        self.output_layer = nn.Linear(
            self.hidden_dim,
            self.action_dim,
        )

        initialize_linear_layer(
            self.fc1,
            gain=nn.init.calculate_gain("relu"),
        )

        initialize_linear_layer(
            self.fc2,
            gain=nn.init.calculate_gain("relu"),
        )

        initialize_linear_layer(
            self.output_layer,
            gain=conf.HOST_ACTOR_GAIN,
        )

    def forward(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:

        hidden = F.relu(self.fc1(obs))

        hidden = F.relu(self.fc2(hidden))

        return self.output_layer(hidden)

    def get_policy(
        self,
        obs: torch.Tensor,
        eps: float = DEFAULT_EPS,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:

        logits = self.forward(obs)

        probabilities = F.softmax(
            logits,
            dim=-1,
        )

        log_probabilities = torch.log(probabilities.clamp_min(float(eps)))

        return (
            probabilities,
            log_probabilities,
            logits,
        )


class LocalHostQNetwork(nn.Module):
    """
    Local Host SAC Critic。

    输入仅为当前 DC 的 Host Observation；
    输出当前 DC 所有 Host action 的 Q value。
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = conf.HOST_CRITIC_HIDDEN_DIM,
    ) -> None:
        super().__init__()

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)

        self.hidden_dim = int(hidden_dim)
        self.fc1 = nn.Linear(
            self.obs_dim,
            hidden_dim,
        )

        self.fc2 = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        self.output_layer = nn.Linear(
            hidden_dim,
            self.action_dim,
        )

        initialize_linear_layer(
            self.fc1,
            gain=nn.init.calculate_gain("relu"),
        )

        initialize_linear_layer(
            self.fc2,
            gain=nn.init.calculate_gain("relu"),
        )

        initialize_linear_layer(
            self.output_layer,
            gain=conf.Q_NET_GAIN,
        )

    def forward(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:

        hidden = F.relu(self.fc1(obs))

        hidden = F.relu(self.fc2(hidden))

        return self.output_layer(hidden)


class LocalHostTwinCritic(nn.Module):
    """
    Local Host SAC 使用独立 Twin Q。

    不与 Routing MASAC Critic 共享任何参数。
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = conf.HOST_CRITIC_HIDDEN_DIM,
    ) -> None:
        super().__init__()

        self.q1 = LocalHostQNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        )

        self.q2 = LocalHostQNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        )

    def forward(
        self,
        obs: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        return (
            self.q1(obs),
            self.q2(obs),
        )


_CHECKPOINT_RETRYABLE_WINDOWS_ERROR_CODES = {
    5,  # 目标文件被映射/扫描时，Windows 也可能返回“拒绝访问”
    32,  # 文件正被其他进程使用
    33,  # 文件区域被锁定
    87,  # Windows ERROR_INVALID_PARAMETER
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
        for attempt_index in range(max_attempts):
            temporary_path = build_temporary_path(attempt_index)

            try:
                torch.save(
                    checkpoint,
                    temporary_path,
                )
                break

            except (OSError, RuntimeError) as error:
                last_error = error
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

                if (
                    not _is_retryable_checkpoint_error(error)
                    or attempt_index + 1 >= max_attempts
                ):
                    raise RuntimeError(
                        "Checkpoint 临时文件写入失败：" f"{target_path}"
                    ) from error

                time.sleep(retry_delay_seconds * float(attempt_index + 1))

        for attempt_index in range(max_attempts):
            try:
                os.replace(
                    temporary_path,
                    target_path,
                )

                if not target_path.is_file() or target_path.stat().st_size <= 0:
                    raise RuntimeError(
                        "Checkpoint 原子写入后文件不存在或为空：" f"{target_path}"
                    )

                return

            except (OSError, RuntimeError) as error:
                last_error = error

                if (
                    not _is_retryable_checkpoint_error(error)
                    or attempt_index + 1 >= max_attempts
                ):
                    raise RuntimeError(
                        "Checkpoint 原子替换失败：" f"{target_path}"
                    ) from error

                time.sleep(retry_delay_seconds * float(attempt_index + 1))

    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass

    raise RuntimeError("Checkpoint 保存失败：" f"{target_path}") from last_error


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
        for attempt_index in range(max_attempts):
            temporary_path = build_temporary_path(attempt_index)
            temporary_paths.append(temporary_path)

            try:
                with temporary_path.open(
                    mode="x",
                    encoding=encoding,
                    newline="\n",
                ) as file_handle:
                    written_character_count = file_handle.write(text)
                    file_handle.flush()
                    os.fsync(file_handle.fileno())

                if written_character_count != len(text):
                    raise OSError("Checkpoint 文本未完整写入临时文件。")
                break

            except OSError as error:
                last_error = error
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

                if (
                    not _is_retryable_checkpoint_error(error)
                    or attempt_index + 1 >= max_attempts
                ):
                    raise RuntimeError(
                        "Checkpoint 文本临时文件写入失败：" f"{target_path}"
                    ) from error

                time.sleep(retry_delay_seconds * float(attempt_index + 1))

        if temporary_path is None:
            raise RuntimeError("Checkpoint 文本临时文件未创建：" f"{target_path}")

        for attempt_index in range(max_attempts):
            try:
                os.replace(
                    temporary_path,
                    target_path,
                )

                if not target_path.is_file() or target_path.stat().st_size <= 0:
                    raise RuntimeError(
                        "Checkpoint 文本原子写入后文件不存在或为空：" f"{target_path}"
                    )
                return

            except (OSError, RuntimeError) as error:
                last_error = error

                if (
                    not _is_retryable_checkpoint_error(error)
                    or attempt_index + 1 >= max_attempts
                ):
                    raise RuntimeError(
                        "Checkpoint 文本原子替换失败：" f"{target_path}"
                    ) from error

                time.sleep(retry_delay_seconds * float(attempt_index + 1))

    finally:
        for candidate_path in temporary_paths:
            try:
                candidate_path.unlink(missing_ok=True)
            except OSError:
                pass

    raise RuntimeError("Checkpoint 文本保存失败：" f"{target_path}") from last_error


@dataclass(frozen=True)
class RoutingMASACConfig:
    enable_heuristic_guidance: bool = True
    enable_reward_shaping: bool = True
    enable_discount_reduction: bool = True
    heuristic_beta: float = 0.98
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


@dataclass(frozen=True)
class RoutingTensorBatch:
    next_heuristic_scores: torch.Tensor
    heuristic_valid: torch.Tensor
    agent_indices: torch.Tensor

    local_obs: torch.Tensor

    global_states: torch.Tensor

    actions: torch.Tensor
    rewards: torch.Tensor

    next_agent_indices: torch.Tensor
    next_local_obs: torch.Tensor
    next_global_states: torch.Tensor

    terminated: torch.Tensor
    truncated: torch.Tensor
    done: torch.Tensor

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


def validate_routing_guidance(config):
    if not math.isfinite(config.heuristic_beta) or not 0 <= config.heuristic_beta <= 1:
        raise ValueError("heuristic_beta must be in [0, 1]")
    if not math.isfinite(config.gamma) or not 0 <= config.gamma < 1:
        raise ValueError("Routing gamma must be in [0, 1)")


def routing_td_target(
    rewards, scores, done, next_soft_value, config, valid=None, *, validate=True
):
    """H-MAR-style target on immutable same-job successor scores."""
    validate_routing_guidance(config)
    if validate:
        if valid is not None and not bool(valid.all()):
            raise ValueError("Missing captured routing heuristic score")
        if not bool(torch.isfinite(scores).all()):
            raise ValueError("Non-finite routing heuristic score")
        if bool((scores[done.to(dtype=torch.bool)] != 0).any()):
            raise ValueError("Terminal routing heuristic must be zero")
    enabled = config.enable_heuristic_guidance
    gamma = float(config.gamma)
    beta = float(config.heuristic_beta)
    bonus = (
        gamma * (1 - beta) * scores
        if enabled and config.enable_reward_shaping
        else torch.zeros_like(rewards)
    )
    effective_gamma = (
        gamma * beta if enabled and config.enable_discount_reduction else gamma
    )
    modified = rewards + bonus
    target = modified + effective_gamma * (1 - done) * next_soft_value
    return target, bonus, modified, effective_gamma


class RoutingMASAC:
    def __init__(
        self,
        local_obs_dim: int,
        global_state_dim: int,
        action_dim: int,
        num_agents: int,
        config: Optional[RoutingMASACConfig] = None,
    ) -> None:

        local_obs_dim = int(local_obs_dim)
        global_state_dim = int(global_state_dim)
        action_dim = int(action_dim)
        num_agents = int(num_agents)

        if config is None:
            config = RoutingMASACConfig()
        self.config = config

        self.local_obs_dim = local_obs_dim
        self.global_state_dim = global_state_dim
        validate_routing_guidance(config)
        self.action_dim = action_dim
        self.num_agents = num_agents

        self.device = resolve_training_device(config.device)

        if config.seed is not None:
            torch.manual_seed(int(config.seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(config.seed))

        self.actor = RoutingDiscreteActor(
            local_obs_dim=local_obs_dim,
            action_dim=action_dim,
            num_agents=num_agents,
            hidden_dim=config.actor_hidden_dim,
        ).to(self.device)

        self.critic = RoutingTwinDiscreteCritic(
            global_state_dim=(global_state_dim),
            action_dim=(action_dim),
            num_agents=(num_agents),
            hidden_dim=(config.critic_hidden_dim),
        ).to(self.device)

        self.target_critic = RoutingTwinDiscreteCritic(
            global_state_dim=(global_state_dim),
            action_dim=(action_dim),
            num_agents=(num_agents),
            hidden_dim=(config.critic_hidden_dim),
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

        agent_index = int(agent_index)

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

            action_probs, _, _ = self.actor.get_policy(
                local_obs=local_obs_tensor,
                agent_indices=(agent_index_tensor),
            )

            if deterministic:

                action_tensor = torch.argmax(
                    action_probs,
                    dim=-1,
                )

            else:

                distribution = torch.distributions.Categorical(probs=action_probs)

                action_tensor = distribution.sample()

        return int(action_tensor.item())

    def update(
        self,
        replay_buffer: RoutingReplayBuffer,
        batch_size: int,
    ) -> Dict[str, float]:
        batch_size = int(batch_size)

        replay_batch = replay_buffer.sample(
            batch_size=batch_size,
            include_forced_actions=False,
            replace=False,
        )

        batch = self._batch_to_tensors(replay_batch)

        critic_info = self._update_critic(batch)

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

        if self.update_step % int(self.config.policy_update_interval) == 0:
            actor_info = self._update_actor(batch)

            alpha_info = self._update_alpha(batch)

            actor_loss_value = actor_info["actor_loss"]
            entropy_value = actor_info["policy_entropy"]
            alpha_loss_value = alpha_info["alpha_loss"]
            target_entropy_value = alpha_info["target_entropy"]

        if self.update_step % int(self.config.target_update_interval) == 0:
            soft_update(
                target_network=self.target_critic,
                source_network=self.critic,
                tau=float(self.config.tau),
            )

        update_info = {
            "heuristic_reward_mean": critic_info["heuristic_reward_mean"],
            "modified_reward_mean": critic_info["modified_reward_mean"],
            "raw_reward_mean": critic_info["raw_reward_mean"],
            "effective_gamma": critic_info["effective_gamma"],
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

    def save(self, file_path: Union[str, Path]) -> None:

        file_path = Path(file_path)

        file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint = {
            "algorithm_role": "bcgh2_masca_routing_ctde_v1",
            "environment_structure": getattr(self, "environment_structure", None),
            "local_obs_dim": self.local_obs_dim,
            "global_state_dim": self.global_state_dim,
            "action_dim": self.action_dim,
            "num_agents": self.num_agents,
            "config": asdict(self.config),
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "target_critic_state_dict": self.target_critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "alpha_optimizer_state_dict": self.alpha_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "update_step": self.update_step,
        }

        _atomic_torch_save(
            checkpoint,
            file_path,
        )

    def load(self, file_path: Union[str, Path], load_optimizers: bool = True) -> None:

        checkpoint = torch.load(
            Path(file_path),
            map_location=self.device,
            weights_only=False,
        )

        role = checkpoint.get("algorithm_role")

        if role != "bcgh2_masca_routing_ctde_v1":
            raise RuntimeError(
                "当前 checkpoint 不是 " "Routing MASAC + CTDE：" f"{role!r}"
            )

        dimension_checks = {
            "local_obs_dim": self.local_obs_dim,
            "global_state_dim": self.global_state_dim,
            "action_dim": self.action_dim,
            "num_agents": self.num_agents,
        }

        for field_name, current_value in dimension_checks.items():
            saved_value = checkpoint.get(field_name)

            if saved_value is None or int(saved_value) != int(current_value):
                raise RuntimeError(
                    "Routing checkpoint "
                    "结构不兼容："
                    f"field={field_name}, "
                    f"saved={saved_value}, "
                    f"current={current_value}"
                )

        saved_config = {
            k: v for k, v in checkpoint.get("config", {}).items() if k != "device"
        }
        current_config = {k: v for k, v in asdict(self.config).items() if k != "device"}
        if saved_config != current_config:
            raise RuntimeError(
                "Routing checkpoint training configuration differs; use explicit weight initialization"
            )

        if checkpoint.get("environment_structure") is not None:
            self.environment_structure = checkpoint["environment_structure"]
        self.actor.load_state_dict(checkpoint["actor_state_dict"])

        self.critic.load_state_dict(checkpoint["critic_state_dict"])

        self.target_critic.load_state_dict(checkpoint["target_critic_state_dict"])

        with torch.no_grad():
            self.log_alpha.copy_(
                checkpoint["log_alpha"].to(
                    device=self.device,
                    dtype=torch.float32,
                )
            )

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

        self.update_step = int(checkpoint.get("update_step", 0))

    def initialize_weights(self, file_path: Union[str, Path]) -> None:
        """Explicit warm start: weights only; no old optimizers, alpha or replay."""
        checkpoint = torch.load(
            Path(file_path), map_location=self.device, weights_only=False
        )
        if checkpoint.get("algorithm_role") not in {
            "routing_masac_ctde",
            "bcgh2_masca_routing_ctde_v1",
        }:
            raise ValueError("Expected compatible Routing weights")
        for name in ("local_obs_dim", "global_state_dim", "action_dim", "num_agents"):
            if checkpoint.get(name) != getattr(self, name):
                raise ValueError(f"Routing weight shape mismatch: {name}")
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.target_critic.load_state_dict(checkpoint["target_critic_state_dict"])

    def train_mode(self) -> None:
        self.actor.train()
        self.critic.train()
        self.target_critic.eval()

    def eval_mode(self) -> None:
        self.actor.eval()
        self.critic.eval()
        self.target_critic.eval()

    def _batch_to_tensors(
        self,
        batch: RoutingReplayBatch,
    ) -> RoutingTensorBatch:
        # Validate on CPU before transfer to avoid synchronizing CUDA per update.
        if not np.all(batch.heuristic_valid) or not np.all(
            np.isfinite(batch.next_heuristic_scores)
        ):
            raise ValueError("Replay batch has missing or non-finite heuristic scores")
        if np.any(batch.next_heuristic_scores[batch.done.astype(bool)] != 0):
            raise ValueError("Terminal routing heuristic must be zero")

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
            next_heuristic_scores=torch.as_tensor(
                batch.next_heuristic_scores, dtype=torch.float32, device=self.device
            ),
            heuristic_valid=torch.as_tensor(
                batch.heuristic_valid, dtype=torch.bool, device=self.device
            ),
            agent_indices=agent_indices,
            local_obs=local_obs,
            global_states=global_states,
            actions=actions,
            rewards=rewards,
            next_agent_indices=next_agent_indices,
            next_local_obs=next_local_obs,
            next_global_states=next_global_states,
            terminated=terminated,
            truncated=truncated,
            done=done,
            is_forced_action=is_forced_action,
        )

    def _update_critic(self, batch: RoutingTensorBatch) -> Dict[str, float]:

        with torch.no_grad():

            safe_next_agent_indices = torch.where(
                batch.done.to(dtype=torch.bool),
                torch.zeros_like(batch.next_agent_indices),
                batch.next_agent_indices,
            )

            next_action_probs, (next_action_log_probs), _ = self.actor.get_policy(
                local_obs=batch.next_local_obs,
                agent_indices=(safe_next_agent_indices),
            )

            target_q1_all, target_q2_all = self.target_critic(
                global_state=batch.next_global_states,
                agent_indices=safe_next_agent_indices,
            )

            target_min_q_all = torch.minimum(
                target_q1_all,
                target_q2_all,
            )

            alpha = self.alpha.detach()

            next_soft_value = (
                next_action_probs * (target_min_q_all - alpha * next_action_log_probs)
            ).sum(dim=-1)

            not_done = 1.0 - batch.done

            target_q, heuristic_reward, modified_reward, effective_gamma = (
                routing_td_target(
                    batch.rewards,
                    batch.next_heuristic_scores,
                    batch.done,
                    next_soft_value,
                    self.config,
                    batch.heuristic_valid,
                    validate=False,
                )
            )

        current_q1_all, current_q2_all = self.critic(
            global_state=batch.global_states,
            agent_indices=batch.agent_indices,
        )

        current_q1 = current_q1_all.gather(
            dim=1,
            index=batch.actions.unsqueeze(1),
        ).squeeze(1)

        current_q2 = current_q2_all.gather(
            dim=1,
            index=batch.actions.unsqueeze(1),
        ).squeeze(1)

        q1_loss = F.mse_loss(
            current_q1,
            target_q,
        )

        q2_loss = F.mse_loss(
            current_q2,
            target_q,
        )

        critic_loss = q1_loss + q2_loss

        self.critic_optimizer.zero_grad(set_to_none=True)

        critic_loss.backward()

        if self.config.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(
                self.critic.parameters(),
                max_norm=float(self.config.max_grad_norm),
            )

        self.critic_optimizer.step()

        return {
            "critic_loss": critic_loss.detach(),
            "q1_loss": q1_loss.detach(),
            "q2_loss": q2_loss.detach(),
            "mean_q1": (current_q1.detach().mean()),
            "mean_q2": (current_q2.detach().mean()),
            "mean_target_q": (target_q.detach().mean()),
            "heuristic_reward_mean": heuristic_reward.detach().mean(),
            "modified_reward_mean": modified_reward.detach().mean(),
            "raw_reward_mean": batch.rewards.detach().mean(),
            "effective_gamma": target_q.new_tensor(effective_gamma),
        }

    def _update_actor(
        self,
        batch: RoutingTensorBatch,
    ) -> Dict[str, float]:

        action_probs, action_log_probs, _ = self.actor.get_policy(
            local_obs=batch.local_obs,
            agent_indices=batch.agent_indices,
        )

        with torch.no_grad():
            q1_all, q2_all = self.critic(
                global_state=batch.global_states,
                agent_indices=batch.agent_indices,
            )

            min_q_all = torch.minimum(
                q1_all,
                q2_all,
            )

        alpha = self.alpha.detach()

        actor_loss_per_sample = (
            action_probs * (alpha * action_log_probs - min_q_all)
        ).sum(dim=-1)

        actor_loss = actor_loss_per_sample.mean()

        self.actor_optimizer.zero_grad(set_to_none=True)

        actor_loss.backward()

        if self.config.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                max_norm=float(self.config.max_grad_norm),
            )

        self.actor_optimizer.step()

        policy_entropy = -(action_probs * action_log_probs).sum(dim=-1)

        return {
            "actor_loss": (actor_loss.detach()),
            "policy_entropy": (policy_entropy.detach().mean()),
        }

    def _update_alpha(
        self,
        batch: RoutingTensorBatch,
    ) -> Dict[str, float]:

        with torch.no_grad():
            action_probs, (action_log_probs), _ = self.actor.get_policy(
                local_obs=batch.local_obs,
                agent_indices=(batch.agent_indices),
            )

            policy_entropy = -(action_probs * action_log_probs).sum(dim=-1)

            target_entropy_value = float(self.config.target_entropy_ratio) * math.log(
                float(self.action_dim)
            )

            target_entropy = torch.full_like(
                policy_entropy,
                fill_value=(target_entropy_value),
            )

        alpha_loss = (
            self.log_alpha * (policy_entropy.detach() - target_entropy.detach())
        ).mean()

        self.alpha_optimizer.zero_grad(set_to_none=True)

        alpha_loss.backward()

        self.alpha_optimizer.step()

        with torch.no_grad():
            self.log_alpha.clamp_(
                min=-20.0,
                max=5.0,
            )

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

        self.device = resolve_training_device(config.device)

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

            probabilities, _, _ = self.actor.get_policy(obs_tensor)

            if deterministic:
                action_tensor = torch.argmax(
                    probabilities,
                    dim=-1,
                )
            else:
                distribution = torch.distributions.Categorical(probs=probabilities)

                action_tensor = distribution.sample()

        return int(action_tensor.item())

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

        batch_size = int(batch_size)

        replay_batch = replay_buffer.sample(
            batch_size=batch_size,
            replace=False,
        )
        if not np.all(replay_batch.done > 0.5):
            raise RuntimeError("HostReplayBuffer 中出现 non-terminal Host transition。")

        batch = self._batch_to_tensors(replay_batch)

        critic_info = self._update_critic(batch)

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

        if self.update_step % int(self.config.policy_update_interval) == 0:
            actor_info = self._update_actor(batch)

            alpha_info = self._update_alpha(batch)

            actor_loss_value = actor_info["actor_loss"]

            entropy_value = actor_info["policy_entropy"]

            alpha_loss_value = alpha_info["alpha_loss"]

            target_entropy_value = alpha_info["target_entropy"]

        if self.update_step % int(self.config.target_update_interval) == 0:
            soft_update(
                target_network=(self.target_critic),
                source_network=(self.critic),
                tau=float(self.config.tau),
            )

        return {
            "update_step": torch.tensor(
                float(self.update_step),
                dtype=torch.float32,
                device=self.device,
            ),
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

        target_q = batch.rewards.detach()

        q1_all, q2_all = self.critic(batch.host_obs)

        action_index = batch.actions.unsqueeze(1)

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

        critic_loss = q1_loss + q2_loss

        self.critic_optimizer.zero_grad(set_to_none=True)

        critic_loss.backward()

        if self.config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(),
                float(self.config.max_grad_norm),
            )

        self.critic_optimizer.step()

        return {
            "critic_loss": critic_loss.detach(),
            "q1_loss": q1_loss.detach(),
            "q2_loss": q2_loss.detach(),
            "mean_q1": q1.detach().mean(),
            "mean_q2": q2.detach().mean(),
            "mean_target_q": target_q.detach().mean(),
        }

    def _update_actor(
        self,
        batch: HostTensorBatch,
    ) -> Dict[str, torch.Tensor]:

        action_probs, action_log_probs, _ = self.actor.get_policy(batch.host_obs)

        with torch.no_grad():
            q1_values, q2_values = self.critic(batch.host_obs)

            min_q_values = torch.minimum(
                q1_values,
                q2_values,
            )

        actor_loss = (
            (action_probs * (self.alpha.detach() * action_log_probs - min_q_values))
            .sum(dim=-1)
            .mean()
        )

        self.actor_optimizer.zero_grad(set_to_none=True)

        actor_loss.backward()

        if self.config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                float(self.config.max_grad_norm),
            )

        self.actor_optimizer.step()

        policy_entropy = -(action_probs * action_log_probs).sum(dim=-1).mean()

        return {
            "actor_loss": actor_loss.detach(),
            "policy_entropy": policy_entropy.detach(),
        }

    def _update_alpha(
        self,
        batch: HostTensorBatch,
    ) -> Dict[str, torch.Tensor]:

        with torch.no_grad():
            action_probs, action_log_probs, _ = self.actor.get_policy(batch.host_obs)

            policy_entropy = -(action_probs * action_log_probs).sum(dim=-1)

            target_entropy = torch.full_like(
                policy_entropy,
                float(self.config.target_entropy_ratio)
                * math.log(
                    max(
                        self.action_dim,
                        1,
                    )
                ),
            )

        alpha_loss = (
            self.log_alpha * (policy_entropy.detach() - target_entropy.detach())
        ).mean()

        self.alpha_optimizer.zero_grad(set_to_none=True)

        alpha_loss.backward()

        self.alpha_optimizer.step()

        with torch.no_grad():
            self.log_alpha.clamp_(
                min=-20.0,
                max=5.0,
            )

        return {
            "alpha_loss": alpha_loss.detach(),
            "target_entropy": target_entropy.detach().mean(),
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

        file_path = Path(file_path)

        file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint = {
            "algorithm_role": "local_host_sac",
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "config": asdict(self.config),
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "target_critic_state_dict": self.target_critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "alpha_optimizer_state_dict": self.alpha_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "update_step": int(self.update_step),
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

        role = checkpoint.get("algorithm_role")

        if role != "local_host_sac":
            raise RuntimeError("当前 checkpoint " "不是 Local Host SAC：" f"{role!r}")

        dimension_checks = {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
        }

        for field_name, current_value in dimension_checks.items():
            saved_value = checkpoint.get(field_name)

            if saved_value is None or int(saved_value) != int(current_value):
                raise RuntimeError(
                    "Local Host SAC checkpoint "
                    "结构不兼容："
                    f"field={field_name}, "
                    f"saved={saved_value}, "
                    f"current={current_value}"
                )

        self.actor.load_state_dict(checkpoint["actor_state_dict"])

        self.critic.load_state_dict(checkpoint["critic_state_dict"])

        self.target_critic.load_state_dict(checkpoint["target_critic_state_dict"])

        with torch.no_grad():
            self.log_alpha.copy_(
                checkpoint["log_alpha"].to(
                    device=self.device,
                    dtype=torch.float32,
                )
            )

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

        self.update_step = int(
            checkpoint.get(
                "update_step",
                0,
            )
        )


@lru_cache(maxsize=None)
def resolve_training_device(
    configured_device: Optional[str],
) -> torch.device:
    device = torch.device(
        "cuda:0" if configured_device is None else str(configured_device)
    )
    if device.type != "cuda":
        raise ValueError(f"当前项目要求只使用 GPU 训练，但配置的设备是：{device}")
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA 不可用：Python={sys.executable}, PyTorch={torch.__version__}, "
            f"CUDA={torch.version.cuda}。请检查 NVIDIA 驱动和 CUDA 版 PyTorch。"
        )
    gpu_index = torch.cuda.current_device() if device.index is None else device.index
    gpu_count = torch.cuda.device_count()
    if gpu_index >= gpu_count:
        raise RuntimeError(
            f"指定了 GPU 编号 {gpu_index}，但 PyTorch 只检测到 {gpu_count} 张 GPU。"
        )
    device = torch.device("cuda", gpu_index)
    try:
        with torch.inference_mode(False), torch.enable_grad():
            x = torch.ones((8, 8), device=device, requires_grad=True)
            weights = torch.full((8, 8), 0.125).to(device)
            loss = (x @ weights).exp().log_softmax(dim=-1).square().mean()
            loss.backward()
            loss.detach().cpu()
            torch.cuda.synchronize(device)
    except (RuntimeError, OSError) as exc:
        try:
            gpu_name = torch.cuda.get_device_name(device)
            capability = torch.cuda.get_device_capability(device)
            architectures = torch.cuda.get_arch_list()
        except RuntimeError:
            gpu_name, capability, architectures = "unknown", None, []
        message = (
            f"CUDA 运算预检失败：device={device}, GPU={gpu_name}, CC={capability}, "
            f"PyTorch={torch.__version__}, CUDA={torch.version.cuda}, 编译架构={architectures}。\n"
            f"原因：{exc}\n请按《服务器端V100环境安装命令.txt》安装支持该 GPU 的 PyTorch 构建。"
            f"\n当前 Python：{sys.executable}"
        )
        if capability == (7, 0):
            message += (
                "\nV100/V100S 使用 CUDA 12.6 构建；在没有运行训练的环境中执行：\n"
                "python -m pip install --upgrade torch==2.10.0+cu126 "
                "--index-url https://download.pytorch.org/whl/cu126"
            )
        raise RuntimeError(message) from exc
    return device
