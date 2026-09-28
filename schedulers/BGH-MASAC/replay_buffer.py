from __future__ import annotations

"""
H-MASAC Replay 子系统入口。

==============================================================
第二十步：修改 replay_buffer.py 的职责
==============================================================

旧架构中，本文件曾经实现一个统一 ReplayBuffer，并同时负责：

    1. Routing Transition 存储；
    2. forced action 过滤；
    3. Edge -> Edge same-job successor 回填；
    4. delayed reward correction；
    5. terminal reward correction；
    6. Job-level transition tracking；
    7. Replay sample。

这些职责在新的两层调度架构中已经被拆开。

当前正式架构：

    PendingJobTraceStore
        │
        │ Job 未 terminal
        ▼
    保存完整 Job Causal Trace
        │
        │ Job terminal
        ▼
    FinalizedJobTrace
        │
        ├───────────────┐
        ▼               ▼
    RoutingReplay   HostReplay
        │               │
        ▼               ▼
    Routing MASAC   Local Host SAC

因此本文件从第二十步开始：

    - 不再实现统一 ReplayBuffer；
    - 不再保存任何 Transition；
    - 不再修改已经写入经验池的 reward；
    - 不再维护 pending Edge successor；
    - 不再维护 Job-level mutable replay state；
    - 不再参与 Causal Trace；
    - 不再负责 SAC sample。

本文件仅作为 Replay 子系统的兼容入口，
暴露当前正式存在的两种 ReplayBuffer。

业务代码仍建议直接从各自模块导入：

    from routing_replay_buffer import RoutingReplayBuffer
    from host_replay_buffer import HostReplayBuffer

不要重新使用统一 ReplayBuffer。
"""

from host_replay_buffer import (
    HostReplayBatch,
    HostReplayBuffer,
)

from routing_replay_buffer import (
    RoutingReplayBatch,
    RoutingReplayBuffer,
)


# ==============================================================
# 当前 Replay 子系统公开接口
# ==============================================================

__all__ = (
    "RoutingReplayBatch",
    "RoutingReplayBuffer",
    "HostReplayBatch",
    "HostReplayBuffer",
)


# ==============================================================
# Legacy API Guard
#
# 如果旧代码仍试图访问：
#
#     ReplayBuffer
#     ReplayBatch
#     TransitionLike
#
# 不进行静默兼容。
#
# 原因：
# 静默把 ReplayBuffer 映射到 RoutingReplayBuffer 会再次掩盖
# “Routing Replay 与 Host Replay 已经彻底拆分”的架构事实。
# ==============================================================

_LEGACY_REPLAY_NAMES = frozenset({
    "ReplayBuffer",
    "ReplayBatch",
    "TransitionLike",
})


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

    raise AttributeError(
        f"module 'replay_buffer' "
        f"has no attribute {name!r}"
    )