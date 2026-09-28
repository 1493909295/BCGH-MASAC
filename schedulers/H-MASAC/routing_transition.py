from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray


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

    # ==========================================================
    # Current Routing Decision
    # ==========================================================

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

    # ==========================================================
    # Same-job Routing Successor
    # ==========================================================

    next_agent_id: Optional[str]
    next_agent_index: int

    next_env_time: float

    next_local_obs: FloatArray
    next_global_state: FloatArray

    # ==========================================================
    # Terminal
    # ==========================================================

    terminated: bool
    truncated: bool
    done: bool

    terminal_reason: Optional[str]