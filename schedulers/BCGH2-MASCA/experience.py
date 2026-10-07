"""Transitions, replay storage, and pending job traces."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class HostTransition:
    """
    Local Host SAC 的一条正式 terminal transition。

    Host 层采用：

        One Job
            ↓
        One Host Decision
            ↓
        One Terminal Transition

    因此同一个 Job 不存在 Host-level successor decision。

    数学形式：

        (
            host_obs,
            host_action,
            host_reward,
            next_host_obs=0,
            done=True
        )

    注意：
        1. 本对象不是 ReplayBuffer；
        2. 本对象不负责 sample；
        3. 本对象只描述一条已经 Finalize 的 Host experience；
        4. 后续 HostReplayBuffer 只负责保存这种对象。
    """

    job_id: str
    dc_id: str

    decision_time: float

    host_obs: FloatArray

    action: int
    host_id: str
    action_source: str

    reward: float

    next_host_obs: FloatArray

    terminated: bool
    truncated: bool
    done: bool

    execution_result: str

    terminal_reason: str
    terminal_time: float


FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class RoutingTransition:
    """
    Job 已经完成 Finalize 后，
    正式写入 RoutingReplayBuffer 的训练经验。

    与 PendingRoutingStep 的区别：

        PendingRoutingStep
            -> Job 尚未完成时保存因果事实

        RoutingTransition
            -> Job terminal 后由完整因果链生成
            -> reward / next_state / done 均已最终确定

    本对象一旦生成，不再允许 delayed reward correction。
    """

    job_id: str

    agent_id: str
    agent_index: int

    env_time: float

    local_obs: FloatArray
    global_state: FloatArray

    action: int
    action_type: str
    action_source: str

    reward: float

    next_agent_id: Optional[str]
    next_agent_index: int

    next_env_time: float

    next_local_obs: FloatArray
    next_global_state: FloatArray

    terminated: bool
    truncated: bool
    done: bool

    terminal_reason: Optional[str]

    next_heuristic_score: float = 0.0
    heuristic_valid: bool = False
    heuristic_timestamp: float = 0.0
    heuristic_version: int = 1


@dataclass(frozen=True)
class HostReplayBatch:

    host_obs: np.ndarray

    actions: np.ndarray
    rewards: np.ndarray

    next_host_obs: np.ndarray

    terminated: np.ndarray
    truncated: np.ndarray
    done: np.ndarray


class HostReplayBuffer:
    """
    单个 Edge DC 的 Local Host SAC 专用经验池。

    一套 LocalHostSAC：
        对应一个 HostReplayBuffer。

    不允许不同 DC 共用 Host ReplayBuffer，
    因为：
        Host Observation 维度可能不同；
        Host action_dim 可能不同；
        Actor/Critic 也完全独立。
    """

    def __init__(
        self,
        *,
        dc_id: str,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        seed: Optional[int] = None,
    ) -> None:

        self.dc_id = str(dc_id)

        self.capacity = int(capacity)

        self.obs_dim = int(obs_dim)

        self.action_dim = int(action_dim)

        self.rng = np.random.default_rng(seed)

        self._position = 0
        self._size = 0
        self.total_added = 0

        self.host_obs = np.zeros(
            (
                self.capacity,
                self.obs_dim,
            ),
            dtype=np.float32,
        )

        self.actions = np.zeros(
            (self.capacity,),
            dtype=np.int64,
        )

        self.rewards = np.zeros(
            (self.capacity,),
            dtype=np.float32,
        )

        self.next_host_obs = np.zeros(
            (
                self.capacity,
                self.obs_dim,
            ),
            dtype=np.float32,
        )

        self.terminated = np.zeros(
            (self.capacity,),
            dtype=np.bool_,
        )

        self.truncated = np.zeros(
            (self.capacity,),
            dtype=np.bool_,
        )

        self.done = np.zeros(
            (self.capacity,),
            dtype=np.bool_,
        )

    def __len__(
        self,
    ) -> int:

        return int(self._size)

    def add(
        self,
        transition: HostTransition,
    ) -> None:

        if str(transition.dc_id) != self.dc_id:
            raise RuntimeError(
                "HostTransition 写入了错误的 DC Replay："
                f"transition_dc={transition.dc_id}, "
                f"buffer_dc={self.dc_id}"
            )

        if not bool(transition.done):
            raise RuntimeError(
                "HostReplayBuffer 只允许 "
                "One-Job terminal transition："
                f"job={transition.job_id}"
            )

        if not (0 <= int(transition.action) < self.action_dim):
            raise ValueError(
                "Host action 越界："
                f"dc={self.dc_id}, "
                f"action={transition.action}, "
                f"action_dim={self.action_dim}"
            )

        obs = np.asarray(
            transition.host_obs,
            dtype=np.float32,
        )

        next_obs = np.asarray(
            transition.next_host_obs,
            dtype=np.float32,
        )

        if obs.shape != (self.obs_dim,):
            raise ValueError(
                "Host Observation shape 错误："
                f"dc={self.dc_id}, "
                f"expected={(self.obs_dim,)}, "
                f"actual={obs.shape}"
            )

        index = int(self._position)

        self.host_obs[index] = obs

        self.actions[index] = int(transition.action)

        self.rewards[index] = float(transition.reward)

        self.next_host_obs[index] = next_obs

        self.terminated[index] = bool(transition.terminated)

        self.truncated[index] = bool(transition.truncated)

        self.done[index] = bool(transition.done)

        self._position = (self._position + 1) % self.capacity

        self._size = min(
            self._size + 1,
            self.capacity,
        )

        self.total_added += 1

    def can_sample(
        self,
        batch_size: int,
    ) -> bool:

        return self._size >= int(batch_size)

    def sample(
        self,
        batch_size: int,
        replace: bool = False,
    ) -> HostReplayBatch:

        batch_size = int(batch_size)

        if not self.can_sample(batch_size):
            raise RuntimeError(
                "HostReplayBuffer 经验不足："
                f"dc={self.dc_id}, "
                f"size={self._size}, "
                f"batch={batch_size}"
            )

        indices = self.rng.choice(
            self._size,
            size=batch_size,
            replace=bool(replace),
        )

        return HostReplayBatch(
            host_obs=(self.host_obs[indices].copy()),
            actions=(self.actions[indices].copy()),
            rewards=(self.rewards[indices].copy()),
            next_host_obs=(self.next_host_obs[indices].copy()),
            terminated=(self.terminated[indices].copy()),
            truncated=(self.truncated[indices].copy()),
            done=(self.done[indices].copy()),
        )

    def clear(
        self,
    ) -> None:

        self._position = 0
        self._size = 0
        self.total_added = 0


@dataclass(frozen=True)
class RoutingReplayBatch:

    agent_indices: np.ndarray

    local_obs: np.ndarray
    global_states: np.ndarray

    actions: np.ndarray
    rewards: np.ndarray

    next_agent_indices: np.ndarray

    next_local_obs: np.ndarray
    next_global_states: np.ndarray

    terminated: np.ndarray
    truncated: np.ndarray
    done: np.ndarray

    is_forced_action: np.ndarray
    next_heuristic_scores: np.ndarray
    heuristic_valid: np.ndarray
    heuristic_timestamps: np.ndarray
    heuristic_versions: np.ndarray


class RoutingReplayBuffer:
    """
    Routing MASAC 专用 ReplayBuffer。

    第十九步以后，本经验池只接受已经 Finalize 的
    RoutingTransition。

    不再负责：
        Pending Edge successor
        delayed reward correction
        terminal reward correction
        Job causal chain
    """

    def __init__(
        self,
        capacity: int,
        local_obs_dim: int,
        global_state_dim: int,
        seed: Optional[int] = None,
    ) -> None:

        self.capacity = int(capacity)

        self.local_obs_dim = int(local_obs_dim)

        self.global_state_dim = int(global_state_dim)

        if self.capacity <= 0:
            raise ValueError("RoutingReplayBuffer capacity 必须 > 0")

        self.rng = np.random.default_rng(seed)

        self._position = 0
        self._size = 0

        self.total_added = 0
        self.next_heuristic_scores = np.zeros(self.capacity, dtype=np.float32)
        self.heuristic_valid = np.zeros(self.capacity, dtype=np.bool_)
        self.heuristic_timestamps = np.zeros(self.capacity, dtype=np.float64)
        self.heuristic_versions = np.ones(self.capacity, dtype=np.int32)

        self.agent_indices = np.full(
            (self.capacity,),
            -1,
            dtype=np.int64,
        )

        self.local_obs = np.zeros(
            (
                self.capacity,
                self.local_obs_dim,
            ),
            dtype=np.float32,
        )

        self.global_states = np.zeros(
            (
                self.capacity,
                self.global_state_dim,
            ),
            dtype=np.float32,
        )

        self.actions = np.zeros(
            (self.capacity,),
            dtype=np.int64,
        )

        self.rewards = np.zeros(
            (self.capacity,),
            dtype=np.float32,
        )

        self.next_agent_indices = np.full(
            (self.capacity,),
            -1,
            dtype=np.int64,
        )

        self.next_local_obs = np.zeros(
            (
                self.capacity,
                self.local_obs_dim,
            ),
            dtype=np.float32,
        )

        self.next_global_states = np.zeros(
            (
                self.capacity,
                self.global_state_dim,
            ),
            dtype=np.float32,
        )

        self.terminated = np.zeros(
            (self.capacity,),
            dtype=np.bool_,
        )

        self.truncated = np.zeros(
            (self.capacity,),
            dtype=np.bool_,
        )

        self.done = np.zeros(
            (self.capacity,),
            dtype=np.bool_,
        )

    def __len__(
        self,
    ) -> int:

        return int(self._size)

    @property
    def num_trainable_actions(
        self,
    ) -> int:

        return int(self._size)

    @property
    def num_forced_actions(
        self,
    ) -> int:

        return 0

    def add(
        self,
        transition: RoutingTransition,
    ) -> None:

        if transition.action_source == "forced":
            raise RuntimeError(
                "forced RoutingTransition "
                "禁止写入 RoutingReplayBuffer："
                f"job={transition.job_id}"
            )

        local_obs = np.asarray(
            transition.local_obs,
            dtype=np.float32,
        )

        global_state = np.asarray(
            transition.global_state,
            dtype=np.float32,
        )

        next_local_obs = np.asarray(
            transition.next_local_obs,
            dtype=np.float32,
        )

        next_global_state = np.asarray(
            transition.next_global_state,
            dtype=np.float32,
        )

        if local_obs.shape != (self.local_obs_dim,):
            raise ValueError("Routing local_obs shape 错误：" f"{local_obs.shape}")

        if global_state.shape != (self.global_state_dim,):
            raise ValueError(
                "Routing global_state shape 错误：" f"{global_state.shape}"
            )

        score = float(transition.next_heuristic_score)
        timestamp = float(transition.heuristic_timestamp)
        if (
            not transition.heuristic_valid
            or not np.isfinite(score)
            or not np.isfinite(timestamp)
        ):
            raise ValueError(
                "Routing replay requires a valid, finite captured heuristic score"
            )
        if int(transition.heuristic_version) != 1:
            raise ValueError("Unsupported heuristic score version")
        if transition.done and score != 0.0:
            raise ValueError("Terminal routing heuristic must be zero")
        if (
            timestamp < transition.env_time
            or timestamp > transition.next_env_time + 1e-9
        ):
            raise ValueError("Heuristic timestamp must belong to the routing successor")
        index = int(self._position)
        self.next_heuristic_scores[index] = score
        self.heuristic_valid[index] = True
        self.heuristic_timestamps[index] = timestamp
        self.heuristic_versions[index] = transition.heuristic_version

        self.agent_indices[index] = int(transition.agent_index)

        self.local_obs[index] = local_obs

        self.global_states[index] = global_state

        self.actions[index] = int(transition.action)

        self.rewards[index] = float(transition.reward)

        self.next_agent_indices[index] = int(transition.next_agent_index)

        self.next_local_obs[index] = next_local_obs

        self.next_global_states[index] = next_global_state

        self.terminated[index] = bool(transition.terminated)

        self.truncated[index] = bool(transition.truncated)

        self.done[index] = bool(transition.done)

        self._position = (self._position + 1) % self.capacity

        self._size = min(
            self._size + 1,
            self.capacity,
        )

        self.total_added += 1

    def can_sample(
        self,
        batch_size: int,
        include_forced_actions: bool = False,
    ) -> bool:

        del include_forced_actions

        return self._size >= int(batch_size)

    def sample(
        self,
        batch_size: int,
        include_forced_actions: bool = False,
        replace: bool = False,
    ) -> RoutingReplayBatch:

        del include_forced_actions

        batch_size = int(batch_size)

        if not self.can_sample(batch_size):
            raise RuntimeError(
                "RoutingReplayBuffer "
                "经验不足，无法采样："
                f"size={self._size}, "
                f"batch={batch_size}"
            )

        indices = self.rng.choice(
            self._size,
            size=batch_size,
            replace=bool(replace),
        )

        return RoutingReplayBatch(
            agent_indices=(self.agent_indices[indices].copy()),
            local_obs=(self.local_obs[indices].copy()),
            global_states=(self.global_states[indices].copy()),
            actions=(self.actions[indices].copy()),
            rewards=(self.rewards[indices].copy()),
            next_agent_indices=(self.next_agent_indices[indices].copy()),
            next_local_obs=(self.next_local_obs[indices].copy()),
            next_global_states=(self.next_global_states[indices].copy()),
            terminated=(self.terminated[indices].copy()),
            truncated=(self.truncated[indices].copy()),
            done=(self.done[indices].copy()),
            next_heuristic_scores=self.next_heuristic_scores[indices].copy(),
            heuristic_valid=self.heuristic_valid[indices].copy(),
            heuristic_timestamps=self.heuristic_timestamps[indices].copy(),
            heuristic_versions=self.heuristic_versions[indices].copy(),
            is_forced_action=np.zeros(
                (batch_size,),
                dtype=np.bool_,
            ),
        )

    def clear(
        self,
    ) -> None:

        self._position = 0
        self._size = 0
        self.total_added = 0
        self.next_heuristic_scores = np.zeros(self.capacity, dtype=np.float32)
        self.heuristic_valid = np.zeros(self.capacity, dtype=np.bool_)
        self.heuristic_timestamps = np.zeros(self.capacity, dtype=np.float64)
        self.heuristic_versions = np.ones(self.capacity, dtype=np.int32)


FloatArray = NDArray[np.float32]


@dataclass
class PendingRoutingStep:
    job_id: str
    sequence_index: int

    agent_id: str
    agent_index: int

    env_time: float

    local_obs: FloatArray
    global_state: FloatArray

    action: int
    action_type: str
    action_source: str

    immediate_reward: float

    source_dc_id: str
    target_dc_id: Optional[str]

    next_agent_id: Optional[str] = None
    next_agent_index: int = -1
    next_env_time: Optional[float] = None

    next_local_obs: Optional[FloatArray] = None
    next_global_state: Optional[FloatArray] = None

    successor_resolved: bool = False
    next_heuristic_score: float = 0.0
    heuristic_valid: bool = False
    heuristic_timestamp: float = 0.0
    heuristic_version: int = 1


@dataclass
class PendingHostStep:
    job_id: str
    dc_id: str

    env_time: float

    host_obs: FloatArray

    action: int
    host_id: str

    action_source: str

    execution_result: Optional[str] = None
    result_time: Optional[float] = None


@dataclass
class PendingRewardEvent:
    job_id: str

    env_time: float

    reward_delta: float
    reason: str

    terminal: bool


@dataclass
class PendingJobTrace:
    job_id: str

    routing_steps: list[PendingRoutingStep] = field(default_factory=list)

    host_step: Optional[PendingHostStep] = None

    reward_events: list[PendingRewardEvent] = field(default_factory=list)

    terminal: bool = False
    terminal_reason: Optional[str] = None
    terminal_time: Optional[float] = None


@dataclass(frozen=True)
class FinalizedJobTrace:
    """
    一个 Job 完整生命周期已经闭合后的不可继续修改因果链。

    FinalizedJobTrace 保存两种信息：

        1. 原始 Causal Facts
           - Routing Steps
           - Host Step
           - Reward Events
           - Terminal Outcome

        2. 从完整因果事实派生出的 Host terminal transition
           - 仅 Routing 最终选择 Self 的 Job 存在
           - Cloud / Drop Job 为 None

    注意：
        FinalizedJobTrace 本身仍然不是 ReplayBuffer。
    """

    job_id: str

    routing_steps: tuple[PendingRoutingStep, ...]

    routing_transitions: tuple[RoutingTransition, ...]

    host_step: Optional[PendingHostStep]

    reward_events: tuple[PendingRewardEvent, ...]

    host_transition: Optional[HostTransition]

    terminal_reason: str
    terminal_time: float


class PendingJobTraceStore:
    """
    保存当前 Episode 内尚未完成生命周期闭环的 Job 因果链。

    注意：
        1. 本类不是 ReplayBuffer；
        2. 不提供 sample()；
        3. 不直接参与 SAC update；
        4. Job terminal 前，所有 Routing / Host 决策事实只记录在这里；
        5. 后续步骤负责将完整 Trace 转换成正式 Replay Transition。
    """

    def __init__(self) -> None:

        self._traces: Dict[
            str,
            PendingJobTrace,
        ] = {}

        self._finalized_traces: Dict[
            str,
            FinalizedJobTrace,
        ] = {}

    def _get_or_create(
        self,
        job_id: str,
    ) -> PendingJobTrace:
        job_id = str(job_id)

        trace = self._traces.get(job_id)

        if trace is None:
            trace = PendingJobTrace(job_id=job_id)

            self._traces[job_id] = trace

        return trace

    def has_trace(
        self,
        job_id: str,
    ) -> bool:
        return str(job_id) in self._traces

    def get_trace(
        self,
        job_id: str,
    ) -> PendingJobTrace:

        job_id = str(job_id)

        trace = self._traces.get(job_id)

        if trace is None:
            raise KeyError("找不到 Pending Job Trace：" f"job={job_id}")

        return trace

    def record_routing_step(
        self,
        *,
        job_id: str,
        agent_id: str,
        agent_index: int,
        env_time: float,
        local_obs: FloatArray,
        global_state: FloatArray,
        action: int,
        action_type: str,
        action_source: str,
        immediate_reward: float,
        target_dc_id: Optional[str],
    ) -> None:
        """
        向指定 Job 的 Pending Causal Trace
        追加一个真实发生的 Routing Decision。

        严格因果约束：

            1. 已 terminal Job 不能继续 Routing；
            2. 上一个 Edge->Edge predecessor 必须先解析完成，
               才允许追加新的 Routing Step；
            3. action_type 与 target_dc_id 必须语义一致；
            4. 不做 visited-DC / cycle 限制。

        因此：
            DC1 -> DC3 -> DC1

        仍然完全允许。

        本函数检查的是“因果正确性”，
        不是限制 Routing 策略。
        """

        job_id = str(job_id)

        agent_id = str(agent_id)

        action_type = str(action_type)

        action_source = str(action_source)

        target_dc_id = None if target_dc_id is None else str(target_dc_id)

        trace = self._get_or_create(job_id)

        if trace.terminal:
            raise RuntimeError(
                "不能向已经 terminal 的 Job Trace "
                "继续添加 Routing Step："
                f"job={job_id}"
            )

        allowed_action_types = {
            "edge_dc",
            "self",
            "cloud",
            "drop",
        }

        if action_type not in allowed_action_types:
            raise ValueError(
                "PendingJobTrace 收到未知 Routing action_type："
                f"job={job_id}, "
                f"action_type={action_type}"
            )

        unresolved_edge_steps = [
            step
            for step in trace.routing_steps
            if (step.action_type == "edge_dc" and not step.successor_resolved)
        ]

        if unresolved_edge_steps:
            raise RuntimeError(
                "上一条 Edge Routing predecessor 尚未解析，"
                "不能继续追加新的 Routing Step："
                f"job={job_id}, "
                f"current_agent={agent_id}, "
                f"unresolved_count={len(unresolved_edge_steps)}"
            )

        if action_type == "edge_dc":

            if target_dc_id is None:
                raise RuntimeError(
                    "Edge Routing 缺少 target_dc_id："
                    f"job={job_id}, "
                    f"source={agent_id}"
                )

            if target_dc_id == agent_id:
                raise RuntimeError(
                    "edge_dc 的 target_dc_id "
                    "不能等于当前 source DC："
                    f"job={job_id}, "
                    f"dc={agent_id}"
                )

        elif action_type == "self":

            if target_dc_id != agent_id:
                raise RuntimeError(
                    "Self Routing 的 target_dc_id "
                    "必须等于当前 DC："
                    f"job={job_id}, "
                    f"source={agent_id}, "
                    f"target={target_dc_id}"
                )

        elif action_type == "cloud":

            if target_dc_id is None:
                raise RuntimeError("Cloud Routing 缺少 target_dc_id：" f"job={job_id}")

        elif action_type == "drop":

            if target_dc_id is not None:
                raise RuntimeError(
                    "Drop action 不应该存在 target_dc_id："
                    f"job={job_id}, "
                    f"target={target_dc_id}"
                )

        sequence_index = len(trace.routing_steps)

        trace.routing_steps.append(
            PendingRoutingStep(
                job_id=job_id,
                sequence_index=(sequence_index),
                agent_id=agent_id,
                agent_index=int(agent_index),
                env_time=float(env_time),
                local_obs=np.asarray(
                    local_obs,
                    dtype=np.float32,
                ).copy(),
                global_state=np.asarray(
                    global_state,
                    dtype=np.float32,
                ).copy(),
                action=int(action),
                action_type=(action_type),
                action_source=(action_source),
                immediate_reward=float(immediate_reward),
                source_dc_id=(agent_id),
                target_dc_id=(target_dc_id),
            )
        )

    def resolve_routing_successor(
        self,
        *,
        job_id: str,
        next_agent_id: str,
        next_agent_index: int,
        next_env_time: float,
        next_local_obs: FloatArray,
        next_global_state: FloatArray,
        next_heuristic_score: float,
        heuristic_valid: bool,
        heuristic_version: int = 1,
    ) -> bool:

        job_id = str(job_id)

        trace = self._traces.get(job_id)

        if trace is None:
            return False

        unresolved_steps = [
            step
            for step in trace.routing_steps
            if (step.action_type == "edge_dc" and not step.successor_resolved)
        ]

        if not unresolved_steps:
            return False

        if len(unresolved_steps) > 1:
            raise RuntimeError(
                "同一个 Job 同时出现多个未解析 "
                "Edge Routing predecessor："
                f"job={job_id}, "
                f"count={len(unresolved_steps)}"
            )

        previous_step = unresolved_steps[-1]

        next_agent_id = str(next_agent_id)

        next_env_time = float(next_env_time)

        expected_target_dc_id = previous_step.target_dc_id

        if expected_target_dc_id is None:
            raise RuntimeError(
                "未解析 Edge predecessor 缺少 target_dc_id："
                f"job={job_id}, "
                f"sequence={previous_step.sequence_index}"
            )

        if next_agent_id != str(expected_target_dc_id):
            raise RuntimeError(
                "Routing Causal Chain target 不一致："
                f"job={job_id}, "
                f"sequence={previous_step.sequence_index}, "
                f"source={previous_step.source_dc_id}, "
                f"expected_target={expected_target_dc_id}, "
                f"actual_next_agent={next_agent_id}"
            )

        if next_env_time < float(previous_step.env_time):
            raise RuntimeError(
                "Routing Causal Chain 时间倒退："
                f"job={job_id}, "
                f"previous_time={previous_step.env_time}, "
                f"next_time={next_env_time}"
            )

        score = float(next_heuristic_score)
        if not heuristic_valid or not np.isfinite(score) or heuristic_version != 1:
            raise ValueError(
                "Nonterminal routing successor requires a valid captured heuristic"
            )

        previous_step.next_agent_id = str(next_agent_id)

        previous_step.next_agent_index = int(next_agent_index)

        previous_step.next_env_time = float(next_env_time)

        previous_step.next_local_obs = np.asarray(
            next_local_obs,
            dtype=np.float32,
        ).copy()

        previous_step.next_global_state = np.asarray(
            next_global_state,
            dtype=np.float32,
        ).copy()

        previous_step.next_heuristic_score = score
        previous_step.heuristic_valid = True
        previous_step.heuristic_timestamp = next_env_time
        previous_step.heuristic_version = heuristic_version
        previous_step.successor_resolved = True

        return True

    def record_host_step(
        self,
        *,
        job_id: str,
        dc_id: str,
        env_time: float,
        host_obs: FloatArray,
        action: int,
        host_id: str,
        action_source: str,
    ) -> None:

        job_id = str(job_id)

        dc_id = str(dc_id)

        trace = self.get_trace(job_id)

        if not trace.routing_steps:
            raise RuntimeError(
                "Host Decision 出现时 Job 没有任何 Routing Step："
                f"job={job_id}, "
                f"dc={dc_id}"
            )

        last_routing_step = trace.routing_steps[-1]

        if last_routing_step.action_type != "self":
            raise RuntimeError(
                "Host Decision 的上一因果节点不是 Self Routing："
                f"job={job_id}, "
                f"last_routing_action="
                f"{last_routing_step.action_type}"
            )

        if last_routing_step.source_dc_id != dc_id:
            raise RuntimeError(
                "Host Decision DC 与 Self Routing DC 不一致："
                f"job={job_id}, "
                f"self_dc={last_routing_step.source_dc_id}, "
                f"host_dc={dc_id}"
            )

        if trace.terminal:
            raise RuntimeError("不能为 terminal Job 添加 Host Step：" f"job={job_id}")

        if trace.host_step is not None:
            raise RuntimeError("同一个 Job 出现了重复 Host Decision：" f"job={job_id}")

        trace.host_step = PendingHostStep(
            job_id=str(job_id),
            dc_id=str(dc_id),
            env_time=float(env_time),
            host_obs=np.asarray(
                host_obs,
                dtype=np.float32,
            ).copy(),
            action=int(action),
            host_id=str(host_id),
            action_source=str(action_source),
        )

    def record_host_result(
        self,
        *,
        job_id: str,
        result: str,
        env_time: float,
    ) -> None:

        trace = self._traces.get(str(job_id))

        if trace is None:
            raise RuntimeError("找不到 Host Result 对应的 " f"Job Trace：{job_id}")

        if trace.host_step is None:
            raise RuntimeError(
                "Job 尚无 Host Step，" "却收到 Host execution result：" f"job={job_id}"
            )

        trace.host_step.execution_result = str(result)

        trace.host_step.result_time = float(env_time)

    def record_reward_event(
        self,
        *,
        job_id: str,
        env_time: float,
        reward_delta: float,
        reason: str,
        terminal: bool,
    ) -> None:

        job_id = str(job_id)

        trace = self.get_trace(job_id)

        trace.reward_events.append(
            PendingRewardEvent(
                job_id=str(job_id),
                env_time=float(env_time),
                reward_delta=float(reward_delta),
                reason=str(reason),
                terminal=bool(terminal),
            )
        )

        if terminal:

            if trace.terminal:
                raise RuntimeError(
                    "同一个 Job 收到了重复 Terminal Event："
                    f"job={job_id}, "
                    f"old={trace.terminal_reason}, "
                    f"new={reason}"
                )

            trace.terminal = True
            trace.terminal_reason = str(reason)
            trace.terminal_time = float(env_time)

    def mark_terminal(
        self,
        *,
        job_id: str,
        env_time: float,
        reason: str,
    ) -> None:

        job_id = str(job_id)

        trace = self.get_trace(job_id)

        if trace.terminal:
            raise RuntimeError(
                "Job Trace 重复 terminal："
                f"job={job_id}, "
                f"old={trace.terminal_reason}, "
                f"new={reason}"
            )

        trace.terminal = True
        trace.terminal_reason = str(reason)
        trace.terminal_time = float(env_time)

    def _validate_trace_for_finalize(
        self,
        trace: PendingJobTrace,
    ) -> None:
        """
        Job Terminal 后，在真正 Finalize 之前，
        对整条 Job Causal Trace 做一次最终结构验证。

        本函数只验证“因果事实是否完整”，
        不进行 Reward Credit Assignment。
        """

        job_id = str(trace.job_id)

        if not trace.terminal:
            raise RuntimeError("不能 Finalize 尚未 terminal 的 Job：" f"job={job_id}")

        if trace.terminal_reason is None:
            raise RuntimeError("Terminal Job 缺少 terminal_reason：" f"job={job_id}")

        if trace.terminal_time is None:
            raise RuntimeError("Terminal Job 缺少 terminal_time：" f"job={job_id}")

        terminal_time = float(trace.terminal_time)

        if not trace.routing_steps:
            raise RuntimeError("Terminal Job 没有任何 Routing Step：" f"job={job_id}")

        routing_steps = trace.routing_steps

        for index, step in enumerate(routing_steps):
            if str(step.job_id) != job_id:
                raise RuntimeError(
                    "Routing Step job_id 与 Trace 不一致："
                    f"trace_job={job_id}, "
                    f"step_job={step.job_id}, "
                    f"sequence={index}"
                )

            if int(step.sequence_index) != index:
                raise RuntimeError(
                    "Routing sequence_index 不连续："
                    f"job={job_id}, "
                    f"expected={index}, "
                    f"actual={step.sequence_index}"
                )

            if step.action_type == "edge_dc":

                if not step.successor_resolved:
                    raise RuntimeError(
                        "Finalize 时仍存在未解析的 "
                        "Edge Routing predecessor："
                        f"job={job_id}, "
                        f"sequence={index}, "
                        f"source={step.source_dc_id}, "
                        f"target={step.target_dc_id}"
                    )

                if step.next_agent_id is None:
                    raise RuntimeError(
                        "Edge Routing 已标记 resolved，"
                        "但 next_agent_id 为 None："
                        f"job={job_id}, "
                        f"sequence={index}"
                    )

                if step.next_env_time is None:
                    raise RuntimeError(
                        "Edge Routing 缺少 next_env_time："
                        f"job={job_id}, "
                        f"sequence={index}"
                    )

                if step.next_local_obs is None:
                    raise RuntimeError(
                        "Edge Routing 缺少 next_local_obs："
                        f"job={job_id}, "
                        f"sequence={index}"
                    )

                if step.next_global_state is None:
                    raise RuntimeError(
                        "Edge Routing 缺少 next_global_state："
                        f"job={job_id}, "
                        f"sequence={index}"
                    )

                if str(step.next_agent_id) != str(step.target_dc_id):
                    raise RuntimeError(
                        "Edge Routing target 与 successor 不一致："
                        f"job={job_id}, "
                        f"sequence={index}, "
                        f"target={step.target_dc_id}, "
                        f"next_agent={step.next_agent_id}"
                    )

                if index + 1 >= len(routing_steps):
                    raise RuntimeError(
                        "Terminal Job 的 Routing Chain "
                        "不能以 edge_dc 结束："
                        f"job={job_id}, "
                        f"sequence={index}"
                    )

                next_step = routing_steps[index + 1]

                if str(next_step.agent_id) != str(step.next_agent_id):
                    raise RuntimeError(
                        "相邻 Routing Step 因果断裂："
                        f"job={job_id}, "
                        f"sequence={index}, "
                        f"resolved_next={step.next_agent_id}, "
                        f"next_step_agent={next_step.agent_id}"
                    )

            else:

                if index != len(routing_steps) - 1:
                    raise RuntimeError(
                        "Routing terminal action 后 "
                        "仍然存在新的 Routing Step："
                        f"job={job_id}, "
                        f"sequence={index}, "
                        f"action_type={step.action_type}"
                    )

        final_routing_step = routing_steps[-1]

        final_action_type = str(final_routing_step.action_type)

        if final_action_type not in {
            "self",
            "cloud",
            "drop",
        }:
            raise RuntimeError(
                "Terminal Job 的最后一条 Routing action "
                "不是 Routing terminal action："
                f"job={job_id}, "
                f"action_type={final_action_type}"
            )

        if final_action_type == "self":

            if trace.host_step is None:
                raise RuntimeError(
                    "Self Routing 后 Job 已 terminal，"
                    "但缺少 Host Step："
                    f"job={job_id}"
                )

            host_step = trace.host_step

            if str(host_step.job_id) != job_id:
                raise RuntimeError(
                    "Host Step job_id 不一致："
                    f"trace_job={job_id}, "
                    f"host_job={host_step.job_id}"
                )

            if str(host_step.dc_id) != str(final_routing_step.source_dc_id):
                raise RuntimeError(
                    "Host Step DC 与最终 Self DC 不一致："
                    f"job={job_id}, "
                    f"self_dc="
                    f"{final_routing_step.source_dc_id}, "
                    f"host_dc={host_step.dc_id}"
                )

            if host_step.execution_result is None:
                raise RuntimeError(
                    "Host Step 尚未记录 execution_result：" f"job={job_id}"
                )

            if host_step.result_time is None:
                raise RuntimeError("Host Step 尚未记录 result_time：" f"job={job_id}")

        elif final_action_type in {
            "cloud",
            "drop",
        }:

            if trace.host_step is not None:
                raise RuntimeError(
                    "Cloud/Drop Routing 后不应该出现 Host Step："
                    f"job={job_id}, "
                    f"action_type={final_action_type}"
                )

        terminal_events = [event for event in trace.reward_events if event.terminal]

        if final_action_type == "drop":

            if str(trace.terminal_reason) != "forced_drop":
                raise RuntimeError(
                    "Drop Routing 的 terminal_reason "
                    "不是 forced_drop："
                    f"job={job_id}, "
                    f"reason={trace.terminal_reason}"
                )

            if terminal_events:
                raise RuntimeError(
                    "Forced Drop 不应该重复存在 "
                    "terminal RewardEvent："
                    f"job={job_id}, "
                    f"count={len(terminal_events)}"
                )

        else:

            if len(terminal_events) != 1:
                raise RuntimeError(
                    "非 Forced-Drop Terminal Job "
                    "必须恰好有一个 terminal RewardEvent："
                    f"job={job_id}, "
                    f"count={len(terminal_events)}"
                )

            terminal_event = terminal_events[0]

            if str(terminal_event.reason) != str(trace.terminal_reason):
                raise RuntimeError(
                    "Terminal Event reason "
                    "与 Trace terminal_reason 不一致："
                    f"job={job_id}, "
                    f"event_reason={terminal_event.reason}, "
                    f"trace_reason={trace.terminal_reason}"
                )

        for event in trace.reward_events:

            if float(event.env_time) > terminal_time + 1e-9:
                raise RuntimeError(
                    "Reward Event 时间晚于 Job terminal："
                    f"job={job_id}, "
                    f"event_time={event.env_time}, "
                    f"terminal_time={terminal_time}"
                )

    def _build_terminal_host_transition(
        self,
        trace: PendingJobTrace,
    ) -> Optional[HostTransition]:
        """
        从已经完整 terminal 的 Job Causal Trace
        派生一条 Local Host SAC terminal transition。

        规则：

            final Routing != Self
                -> None

            final Routing == Self
                -> 必须恰好存在一个 Host Decision
                -> 构造恰好一条 HostTransition

        Host 层不存在 same-job successor Host Decision，因此：

            next_host_obs = zeros
            terminated = True
            truncated = False
            done = True

        本函数不会：
            - 创建 HostReplayBuffer；
            - 进行 SAC update；
            - 人工构造下一 Host Observation；
            - 把其他 Job 的 Host Observation 当作 next state。
        """

        job_id = str(trace.job_id)

        host_step = trace.host_step

        if host_step is None:
            return None

        if not trace.routing_steps:
            raise RuntimeError(
                "构造 HostTransition 时 " "Job 没有 Routing Step：" f"job={job_id}"
            )

        final_routing_step = trace.routing_steps[-1]

        if str(final_routing_step.action_type) != "self":
            raise RuntimeError(
                "存在 Host Step，"
                "但最终 Routing action 不是 Self："
                f"job={job_id}, "
                f"final_action="
                f"{final_routing_step.action_type}"
            )

        if host_step.execution_result is None:
            raise RuntimeError(
                "构造 HostTransition 时 "
                "Host execution_result 尚未记录："
                f"job={job_id}"
            )

        if not trace.terminal:
            raise RuntimeError(
                "不能从未 terminal Job " "构造 HostTransition：" f"job={job_id}"
            )

        if trace.terminal_reason is None:
            raise RuntimeError(
                "构造 HostTransition 时 " "缺少 terminal_reason：" f"job={job_id}"
            )

        if trace.terminal_time is None:
            raise RuntimeError(
                "构造 HostTransition 时 " "缺少 terminal_time：" f"job={job_id}"
            )

        host_decision_time = float(host_step.env_time)

        terminal_time = float(trace.terminal_time)

        if terminal_time < host_decision_time:
            raise RuntimeError(
                "Host terminal_time 早于 Host decision_time："
                f"job={job_id}, "
                f"decision_time={host_decision_time}, "
                f"terminal_time={terminal_time}"
            )

        host_reward_events = [
            event
            for event in trace.reward_events
            if (float(event.env_time) + 1e-9 >= host_decision_time)
        ]

        host_terminal_events = [
            event for event in host_reward_events if bool(event.terminal)
        ]

        if len(host_terminal_events) != 1:
            raise RuntimeError(
                "Host Job 必须恰好存在一个 "
                "terminal RewardEvent："
                f"job={job_id}, "
                f"terminal_event_count="
                f"{len(host_terminal_events)}"
            )

        terminal_event = host_terminal_events[0]

        if str(terminal_event.reason) != str(trace.terminal_reason):
            raise RuntimeError(
                "Host terminal RewardEvent "
                "与 Trace terminal_reason 不一致："
                f"job={job_id}, "
                f"event_reason={terminal_event.reason}, "
                f"trace_reason={trace.terminal_reason}"
            )

        host_reward = float(
            sum(float(event.reward_delta) for event in host_reward_events)
        )

        host_obs = np.asarray(
            host_step.host_obs,
            dtype=np.float32,
        ).copy()

        if host_obs.ndim != 1:
            raise RuntimeError(
                "HostTransition host_obs "
                "必须是一维向量："
                f"job={job_id}, "
                f"shape={host_obs.shape}"
            )

        next_host_obs = np.zeros_like(
            host_obs,
            dtype=np.float32,
        )

        return HostTransition(
            job_id=job_id,
            dc_id=str(host_step.dc_id),
            decision_time=(host_decision_time),
            host_obs=(host_obs),
            action=int(host_step.action),
            host_id=str(host_step.host_id),
            action_source=str(host_step.action_source),
            reward=(host_reward),
            next_host_obs=(next_host_obs),
            terminated=True,
            truncated=False,
            done=True,
            execution_result=str(host_step.execution_result),
            terminal_reason=str(trace.terminal_reason),
            terminal_time=(terminal_time),
        )

    def _build_finalized_routing_transitions(
        self,
        trace: PendingJobTrace,
    ) -> tuple[RoutingTransition, ...]:
        """
        从已经 terminal 的完整 Job Causal Trace
        一次性生成 Routing MASAC 正式训练经验。

        核心规则：

            1. forced Routing Step 不直接进入 Replay；
            2. 每个正常 Edge action 的 next_state
               必须来自 same-job successor；
            3. Self / Cloud / Actor Drop 为 Routing terminal；
            4. Terminal reward 只归属于最后一个
               Actor-controlled Routing Transition；
            5. Forced Drop penalty 归因到最近一次
               Actor-controlled Routing Transition；
            6. 不再修改已经进入 ReplayBuffer 的经验。
        """

        job_id = str(trace.job_id)

        routing_steps = list(trace.routing_steps)

        actor_step_indices = [
            index
            for index, step in enumerate(routing_steps)
            if step.action_source
            in {
                "random",
                "policy",
            }
        ]

        if not actor_step_indices:
            return tuple()

        final_rewards = {
            index: float(routing_steps[index].immediate_reward)
            for index in actor_step_indices
        }

        for event in trace.reward_events:

            event_time = float(event.env_time)

            candidate_indices = [
                index
                for index in actor_step_indices
                if (float(routing_steps[index].env_time) <= event_time + 1e-9)
            ]

            if not candidate_indices:
                continue

            target_index = candidate_indices[-1]

            final_rewards[target_index] += float(event.reward_delta)

        for forced_index, forced_step in enumerate(routing_steps):

            if forced_step.action_source != "forced":
                continue

            forced_reward = float(forced_step.immediate_reward)

            if abs(forced_reward) <= 1e-12:
                continue

            predecessor_indices = [
                index for index in actor_step_indices if index < forced_index
            ]

            if not predecessor_indices:
                continue

            target_index = predecessor_indices[-1]

            final_rewards[target_index] += forced_reward

        finalized_transitions = []

        for actor_position, step_index in enumerate(actor_step_indices):

            step = routing_steps[step_index]

            next_raw_step = (
                routing_steps[step_index + 1]
                if (step_index + 1 < len(routing_steps))
                else None
            )

            has_normal_routing_successor = bool(
                step.action_type == "edge_dc"
                and next_raw_step is not None
                and next_raw_step.action_source
                in {
                    "random",
                    "policy",
                }
            )

            if has_normal_routing_successor:

                if not step.successor_resolved:
                    raise RuntimeError(
                        "Finalized Edge Routing "
                        "缺少 same-job successor："
                        f"job={job_id}, "
                        f"sequence={step.sequence_index}"
                    )

                if step.next_local_obs is None:
                    raise RuntimeError(
                        "Finalized Edge Routing "
                        "缺少 next_local_obs："
                        f"job={job_id}"
                    )

                if step.next_global_state is None:
                    raise RuntimeError(
                        "Finalized Edge Routing "
                        "缺少 next_global_state："
                        f"job={job_id}"
                    )

                next_agent_id = str(step.next_agent_id)

                next_agent_index = int(step.next_agent_index)

                next_env_time = float(step.next_env_time)

                next_local_obs = np.asarray(
                    step.next_local_obs,
                    dtype=np.float32,
                ).copy()

                next_global_state = np.asarray(
                    step.next_global_state,
                    dtype=np.float32,
                ).copy()

                terminated = False
                truncated = False
                done = False

                terminal_reason = None
                if not step.heuristic_valid:
                    raise RuntimeError(
                        "Nonterminal routing successor is missing its heuristic score"
                    )
                heuristic_score = step.next_heuristic_score
                heuristic_timestamp = step.heuristic_timestamp

            else:

                next_agent_id = None
                next_agent_index = -1

                next_env_time = float(trace.terminal_time)

                next_local_obs = np.zeros_like(
                    np.asarray(
                        step.local_obs,
                        dtype=np.float32,
                    )
                )

                next_global_state = np.zeros_like(
                    np.asarray(
                        step.global_state,
                        dtype=np.float32,
                    )
                )

                terminated = True
                truncated = False
                done = True

                terminal_reason = str(trace.terminal_reason)
                heuristic_score = 0.0
                heuristic_timestamp = float(trace.terminal_time)

            finalized_transitions.append(
                RoutingTransition(
                    job_id=job_id,
                    agent_id=str(step.agent_id),
                    agent_index=int(step.agent_index),
                    env_time=float(step.env_time),
                    local_obs=np.asarray(
                        step.local_obs,
                        dtype=np.float32,
                    ).copy(),
                    global_state=np.asarray(
                        step.global_state,
                        dtype=np.float32,
                    ).copy(),
                    action=int(step.action),
                    action_type=str(step.action_type),
                    action_source=str(step.action_source),
                    reward=float(final_rewards[step_index]),
                    next_agent_id=(next_agent_id),
                    next_agent_index=(next_agent_index),
                    next_env_time=(next_env_time),
                    next_local_obs=(next_local_obs),
                    next_global_state=(next_global_state),
                    terminated=(terminated),
                    truncated=(truncated),
                    done=(done),
                    terminal_reason=(terminal_reason),
                    next_heuristic_score=heuristic_score,
                    heuristic_valid=True,
                    heuristic_timestamp=heuristic_timestamp,
                    heuristic_version=1,
                )
            )

        return tuple(finalized_transitions)

    def _build_finalized_trace(
        self,
        trace: PendingJobTrace,
    ) -> FinalizedJobTrace:
        """
        从已经通过结构验证的 PendingJobTrace
        生成完全独立的 FinalizedJobTrace 副本。

        使用 deepcopy 的原因：
            Finalized Trace 不应该继续引用 Pending 对象中的
            numpy array / Step 实例。
        """

        if trace.terminal_reason is None:
            raise RuntimeError(
                "Finalized Trace 缺少 terminal_reason：" f"job={trace.job_id}"
            )

        if trace.terminal_time is None:
            raise RuntimeError(
                "Finalized Trace 缺少 terminal_time：" f"job={trace.job_id}"
            )

        finalized_routing_steps = tuple(
            copy.deepcopy(step) for step in trace.routing_steps
        )

        finalized_host_step = (
            None if trace.host_step is None else copy.deepcopy(trace.host_step)
        )

        finalized_reward_events = tuple(
            copy.deepcopy(event) for event in trace.reward_events
        )

        finalized_routing_transitions = self._build_finalized_routing_transitions(trace)

        finalized_host_transition = self._build_terminal_host_transition(trace)

        return FinalizedJobTrace(
            job_id=str(trace.job_id),
            routing_steps=(finalized_routing_steps),
            routing_transitions=(finalized_routing_transitions),
            host_step=(finalized_host_step),
            reward_events=(finalized_reward_events),
            host_transition=(finalized_host_transition),
            terminal_reason=str(trace.terminal_reason),
            terminal_time=float(trace.terminal_time),
        )

    def finalize_terminal_trace(
        self,
        job_id: str,
    ) -> FinalizedJobTrace:
        """
        对一个已经 terminal 的 Job 一次性完成 Finalize。

        原子流程：

            Pending Trace
                ↓
            Structural Validation
                ↓
            Deep Copy
                ↓
            FinalizedJobTrace
                ↓
            从 Pending 区删除
                ↓
            写入 Finalized 区

        本函数不写 ReplayBuffer。
        """

        job_id = str(job_id)

        if job_id in self._finalized_traces:
            raise RuntimeError("同一个 Job 被重复 Finalize：" f"job={job_id}")

        trace = self.get_trace(job_id)

        self._validate_trace_for_finalize(trace)

        finalized_trace = self._build_finalized_trace(trace)

        self._finalized_traces[job_id] = finalized_trace

        del self._traces[job_id]

        return finalized_trace

    def has_finalized_trace(
        self,
        job_id: str,
    ) -> bool:

        return str(job_id) in self._finalized_traces

    def get_finalized_trace(
        self,
        job_id: str,
    ) -> FinalizedJobTrace:

        job_id = str(job_id)

        trace = self._finalized_traces.get(job_id)

        if trace is None:
            raise KeyError("找不到 Finalized Job Trace：" f"job={job_id}")

        return trace

    def pop_finalized_trace(
        self,
        job_id: str,
    ) -> FinalizedJobTrace:
        """
        FinalizedJobTrace 已经成功写入：

            RoutingReplayBuffer
            HostReplayBuffer

        后，才允许从 Trace Store 删除。

        注意：
            Finalize != Replay Flush

        必须先：
            finalize
                ↓
            replay add success
                ↓
            pop finalized trace
        """

        job_id = str(job_id)

        trace = self.get_finalized_trace(job_id)

        del self._finalized_traces[job_id]

        return trace

    def get_finalized_host_transition(
        self,
        job_id: str,
    ) -> Optional[HostTransition]:
        """
        返回指定已 Finalize Job 对应的 Local Host SAC
        terminal transition。

        Self Job:
            HostTransition

        Cloud / Drop Job:
            None
        """

        finalized_trace = self.get_finalized_trace(job_id)

        return finalized_trace.host_transition

    def is_terminal(
        self,
        job_id: str,
    ) -> bool:

        job_id = str(job_id)

        if job_id in self._finalized_traces:
            return True

        trace = self._traces.get(job_id)

        if trace is None:
            return False

        return bool(trace.terminal)

    def remove_trace(
        self,
        job_id: str,
    ) -> Optional[PendingJobTrace]:

        return self._traces.pop(
            str(job_id),
            None,
        )

    @property
    def open_trace_count(
        self,
    ) -> int:
        """
        尚未 terminal 的 Pending Trace 数。
        """

        return sum(1 for trace in self._traces.values() if not trace.terminal)

    @property
    def pending_trace_count(
        self,
    ) -> int:
        """
        尚未完成 Finalize 的 Trace 总数。

        包括：
            open
            terminal-but-not-finalized
        """

        return len(self._traces)

    @property
    def terminal_trace_count(
        self,
    ) -> int:
        """
        已经 terminal 的 Job 总数。

        正常情况下 terminal 后会立即 Finalize，
        因而主要来自 _finalized_traces。
        """

        pending_terminal_count = sum(
            1 for trace in self._traces.values() if trace.terminal
        )

        return pending_terminal_count + len(self._finalized_traces)

    @property
    def finalized_trace_count(
        self,
    ) -> int:

        return len(self._finalized_traces)

    @property
    def finalized_host_transition_count(
        self,
    ) -> int:
        """
        当前 Episode 已经形成的正式
        Local Host SAC terminal transition 数量。

        理论上应等于：
            最终 Routing action == Self
            且 Job 已 terminal
            的 Job 数。
        """

        return sum(
            1
            for trace in self._finalized_traces.values()
            if (trace.host_transition is not None)
        )

    @property
    def total_trace_count(
        self,
    ) -> int:

        return len(self._traces) + len(self._finalized_traces)

    def assert_no_open_trace(
        self,
    ) -> None:
        """
        Episode 结束以后，Pending 区必须完全为空。

        因为：

            open Job
                → 错误

            terminal 但未 Finalize
                → 同样错误
        """

        if not self._traces:
            return

        remaining_details = [
            {
                "job_id": job_id,
                "terminal": bool(trace.terminal),
                "terminal_reason": (trace.terminal_reason),
                "routing_steps": len(trace.routing_steps),
                "has_host_step": (trace.host_step is not None),
            }
            for job_id, trace in self._traces.items()
        ]

        raise RuntimeError(
            "Episode 已结束，但仍存在未 Finalize 的 "
            "Pending Job Causal Trace："
            f"{remaining_details[:20]}"
        )

    def reset_episode(
        self,
    ) -> None:
        """
        开始新 Episode 前清理上一 Episode Trace。

        当前第十六步 Finalized Trace 只保留到 Episode 边界。

        后续双 ReplayBuffer 建成以后，
        Finalized Trace 会在这里之前被 pop 并写入经验池。
        """

        if self._traces:
            self.assert_no_open_trace()

        self._traces.clear()

        self._finalized_traces.clear()

    def assert_no_unflushed_finalized_trace(
        self,
    ) -> None:

        if not self._finalized_traces:
            return

        raise RuntimeError(
            "Episode 已结束，但仍存在已经 Finalize "
            "却没有写入 ReplayBuffer 的 Job："
            f"{list(self._finalized_traces.keys())[:20]}"
        )


_LEGACY_REPLAY_NAMES = frozenset(
    {
        "ReplayBuffer",
        "ReplayBatch",
        "TransitionLike",
    }
)


def __getattr__(
    name: str,
):
    """
    对旧统一 Replay API 给出明确错误信息。

    这里故意不提供：

        ReplayBuffer = RoutingReplayBuffer

    这种兼容 alias。

    因为这种 alias 会让旧代码继续运行，
    但会模糊 Routing / Host 两个经验池的边界。
    """

    if name in _LEGACY_REPLAY_NAMES:
        raise AttributeError(
            "旧统一 Replay API 已经在第二十步移除："
            f"{name!r}。"
            "请根据调用层级显式使用 "
            "RoutingReplayBuffer / RoutingReplayBatch "
            "或者 "
            "HostReplayBuffer / HostReplayBatch。"
        )

    raise AttributeError(f"module 'replay_buffer' " f"has no attribute {name!r}")
