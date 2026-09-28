from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from routing_transition import (RoutingTransition,)


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

    # 为兼容当前 DiscreteMASAC TensorBatch。
    # 正式 RoutingReplayBuffer 不保存 forced transition，
    # 因此这里永远为 False。
    is_forced_action: np.ndarray


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

        self.capacity = int(
            capacity
        )

        self.local_obs_dim = int(
            local_obs_dim
        )

        self.global_state_dim = int(
            global_state_dim
        )

        if self.capacity <= 0:
            raise ValueError(
                "RoutingReplayBuffer capacity 必须 > 0"
            )

        self.rng = np.random.default_rng(
            seed
        )

        self._position = 0
        self._size = 0

        self.total_added = 0

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

        return int(
            self._size
        )

    @property
    def num_trainable_actions(
            self,
    ) -> int:

        # 本池中所有经验都已经完成 Finalize，
        # 因此全部可以训练。
        return int(
            self._size
        )

    @property
    def num_forced_actions(
            self,
    ) -> int:

        # forced transition 根本不会写入本池。
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

        if local_obs.shape != (
            self.local_obs_dim,
        ):
            raise ValueError(
                "Routing local_obs shape 错误："
                f"{local_obs.shape}"
            )

        if global_state.shape != (
            self.global_state_dim,
        ):
            raise ValueError(
                "Routing global_state shape 错误："
                f"{global_state.shape}"
            )

        index = int(
            self._position
        )

        self.agent_indices[
            index
        ] = int(
            transition.agent_index
        )

        self.local_obs[
            index
        ] = local_obs

        self.global_states[
            index
        ] = global_state

        self.actions[
            index
        ] = int(
            transition.action
        )

        self.rewards[
            index
        ] = float(
            transition.reward
        )

        self.next_agent_indices[
            index
        ] = int(
            transition.next_agent_index
        )

        self.next_local_obs[
            index
        ] = next_local_obs

        self.next_global_states[
            index
        ] = next_global_state

        self.terminated[
            index
        ] = bool(
            transition.terminated
        )

        self.truncated[
            index
        ] = bool(
            transition.truncated
        )

        self.done[
            index
        ] = bool(
            transition.done
        )

        self._position = (
            self._position + 1
        ) % self.capacity

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

        return (
            self._size
            >= int(batch_size)
        )

    def sample(
            self,
            batch_size: int,
            include_forced_actions: bool = False,
            replace: bool = False,
    ) -> RoutingReplayBatch:

        del include_forced_actions

        batch_size = int(
            batch_size
        )

        if not self.can_sample(
            batch_size
        ):
            raise RuntimeError(
                "RoutingReplayBuffer "
                "经验不足，无法采样："
                f"size={self._size}, "
                f"batch={batch_size}"
            )

        indices = self.rng.choice(
            self._size,
            size=batch_size,
            replace=bool(
                replace
            ),
        )

        return RoutingReplayBatch(
            agent_indices=(
                self.agent_indices[
                    indices
                ].copy()
            ),

            local_obs=(
                self.local_obs[
                    indices
                ].copy()
            ),

            global_states=(
                self.global_states[
                    indices
                ].copy()
            ),

            actions=(
                self.actions[
                    indices
                ].copy()
            ),

            rewards=(
                self.rewards[
                    indices
                ].copy()
            ),

            next_agent_indices=(
                self.next_agent_indices[
                    indices
                ].copy()
            ),

            next_local_obs=(
                self.next_local_obs[
                    indices
                ].copy()
            ),

            next_global_states=(
                self.next_global_states[
                    indices
                ].copy()
            ),

            terminated=(
                self.terminated[
                    indices
                ].copy()
            ),

            truncated=(
                self.truncated[
                    indices
                ].copy()
            ),

            done=(
                self.done[
                    indices
                ].copy()
            ),

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