from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from host_transition import HostTransition


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

        self.dc_id = str(
            dc_id
        )

        self.capacity = int(
            capacity
        )

        self.obs_dim = int(
            obs_dim
        )

        self.action_dim = int(
            action_dim
        )

        self.rng = np.random.default_rng(
            seed
        )

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

        return int(
            self._size
        )

    def add(
            self,
            transition: HostTransition,
    ) -> None:

        if str(
            transition.dc_id
        ) != self.dc_id:
            raise RuntimeError(
                "HostTransition 写入了错误的 DC Replay："
                f"transition_dc={transition.dc_id}, "
                f"buffer_dc={self.dc_id}"
            )

        if not bool(
            transition.done
        ):
            raise RuntimeError(
                "HostReplayBuffer 只允许 "
                "One-Job terminal transition："
                f"job={transition.job_id}"
            )

        if not (
            0
            <= int(transition.action)
            < self.action_dim
        ):
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

        if obs.shape != (
            self.obs_dim,
        ):
            raise ValueError(
                "Host Observation shape 错误："
                f"dc={self.dc_id}, "
                f"expected={(self.obs_dim,)}, "
                f"actual={obs.shape}"
            )

        index = int(
            self._position
        )

        self.host_obs[
            index
        ] = obs

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

        self.next_host_obs[
            index
        ] = next_obs

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
    ) -> bool:

        return (
            self._size
            >= int(batch_size)
        )

    def sample(
            self,
            batch_size: int,
            replace: bool = False,
    ) -> HostReplayBatch:

        batch_size = int(
            batch_size
        )

        if not self.can_sample(
            batch_size
        ):
            raise RuntimeError(
                "HostReplayBuffer 经验不足："
                f"dc={self.dc_id}, "
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

        return HostReplayBatch(
            host_obs=(
                self.host_obs[
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

            next_host_obs=(
                self.next_host_obs[
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
        )

    def clear(
            self,
    ) -> None:

        self._position = 0
        self._size = 0
        self.total_added = 0