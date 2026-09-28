from __future__ import annotations

from dataclasses import dataclass

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

    # ==========================================================
    # Job / DC identity
    # ==========================================================

    job_id: str
    dc_id: str

    # ==========================================================
    # Host decision state
    #
    # decision_time 是 Local Host SAC 真正执行 action 的时间，
    # 不是任务完成时间。
    # ==========================================================

    decision_time: float

    host_obs: FloatArray

    # ==========================================================
    # Host action
    # ==========================================================

    action: int
    host_id: str
    action_source: str

    # ==========================================================
    # One-Job terminal return
    #
    # reward 表示从 Host decision 以后直到该 Job terminal
    # 所产生的 Host-level reward。
    # ==========================================================

    reward: float

    # ==========================================================
    # Host 层不存在同 Job 下一次 Host Decision。
    #
    # 为了保持标准 SAC Transition 接口，
    # next_host_obs 使用与 host_obs 同 shape 的全 0 向量。
    # ==========================================================

    next_host_obs: FloatArray

    terminated: bool
    truncated: bool
    done: bool

    # ==========================================================
    # Debug / causal metadata
    # ==========================================================

    execution_result: str

    terminal_reason: str
    terminal_time: float