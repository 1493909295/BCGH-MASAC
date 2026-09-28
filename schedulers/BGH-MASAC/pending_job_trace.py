from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray
from host_transition import HostTransition
from routing_transition import (RoutingTransition,)

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

    # Edge -> Edge 的真实 same-job successor
    # 在任务真正到达下一 DC 时再填写。
    next_agent_id: Optional[str] = None
    next_agent_index: int = -1
    next_env_time: Optional[float] = None

    next_local_obs: Optional[FloatArray] = None
    next_global_state: Optional[FloatArray] = None

    successor_resolved: bool = False


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

    routing_steps: list[
        PendingRoutingStep
    ] = field(default_factory=list)

    host_step: Optional[
        PendingHostStep
    ] = None

    reward_events: list[
        PendingRewardEvent
    ] = field(default_factory=list)

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

    routing_steps: tuple[
        PendingRoutingStep,
        ...
    ]

    routing_transitions: tuple[
        RoutingTransition,
        ...
    ]

    host_step: Optional[
        PendingHostStep
    ]

    reward_events: tuple[
        PendingRewardEvent,
        ...
    ]

    # ==========================================================
    # Local Host SAC 的正式单步 terminal experience。
    #
    # Self:
    #     HostTransition
    #
    # Cloud / Drop:
    #     None
    # ==========================================================

    host_transition: Optional[
        HostTransition
    ]

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
        # ==========================================================
        # 尚未 Finalize 的 Job。
        #
        # Job 从第一次 Routing Decision 开始进入这里。
        # ==========================================================

        self._traces: Dict[
            str,
            PendingJobTrace,
        ] = {}

        # ==========================================================
        # 已完成 terminal + structural validation 的 Job。
        #
        # 注意：
        # 这里仍然不是 ReplayBuffer。
        #
        # 后续步骤会把 FinalizedJobTrace 转换为：
        #
        #   Routing Replay Transition
        #   Host Replay Transition
        # ==========================================================

        self._finalized_traces: Dict[
            str,
            FinalizedJobTrace,
        ] = {}

    def _get_or_create(
            self,
            job_id: str,
    ) -> PendingJobTrace:
        job_id = str(job_id)

        trace = self._traces.get(
            job_id
        )

        if trace is None:
            trace = PendingJobTrace(
                job_id=job_id
            )

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

        trace = self._traces.get(
            job_id
        )

        if trace is None:
            raise KeyError(
                "找不到 Pending Job Trace："
                f"job={job_id}"
            )

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

        job_id = str(
            job_id
        )

        agent_id = str(
            agent_id
        )

        action_type = str(
            action_type
        )

        action_source = str(
            action_source
        )

        target_dc_id = (
            None
            if target_dc_id is None
            else str(target_dc_id)
        )

        trace = self._get_or_create(
            job_id
        )

        # ==========================================================
        # 已 terminal Job 不允许继续产生 Routing decision。
        # ==========================================================

        if trace.terminal:
            raise RuntimeError(
                "不能向已经 terminal 的 Job Trace "
                "继续添加 Routing Step："
                f"job={job_id}"
            )

        # ==========================================================
        # Routing Action Type 合法性检查。
        # ==========================================================

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

        # ==========================================================
        # 严格因果链：
        #
        # 在添加新的 Routing Step 之前，
        # 不允许该 Job 还有一个尚未解析的 Edge predecessor。
        #
        # 正常顺序必须是：
        #
        #   record R0: DC1 -> DC3
        #       ↓
        #   Job 到 DC3
        #       ↓
        #   resolve R0.next_state
        #       ↓
        #   才允许 record R1
        # ==========================================================

        unresolved_edge_steps = [
            step
            for step
            in trace.routing_steps
            if (
                    step.action_type == "edge_dc"
                    and not step.successor_resolved
            )
        ]

        if unresolved_edge_steps:
            raise RuntimeError(
                "上一条 Edge Routing predecessor 尚未解析，"
                "不能继续追加新的 Routing Step："
                f"job={job_id}, "
                f"current_agent={agent_id}, "
                f"unresolved_count={len(unresolved_edge_steps)}"
            )

        # ==========================================================
        # Action Semantic Validation
        # ==========================================================

        if action_type == "edge_dc":

            if target_dc_id is None:
                raise RuntimeError(
                    "Edge Routing 缺少 target_dc_id："
                    f"job={job_id}, "
                    f"source={agent_id}"
                )

            # target == source 应该被编码为 Self，
            # 不能被错误标记为 edge_dc。
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
                raise RuntimeError(
                    "Cloud Routing 缺少 target_dc_id："
                    f"job={job_id}"
                )

        elif action_type == "drop":

            if target_dc_id is not None:
                raise RuntimeError(
                    "Drop action 不应该存在 target_dc_id："
                    f"job={job_id}, "
                    f"target={target_dc_id}"
                )

        # ==========================================================
        # Routing Step Sequence
        # ==========================================================

        sequence_index = len(
            trace.routing_steps
        )

        trace.routing_steps.append(
            PendingRoutingStep(
                job_id=job_id,

                sequence_index=(
                    sequence_index
                ),

                agent_id=agent_id,

                agent_index=int(
                    agent_index
                ),

                env_time=float(
                    env_time
                ),

                local_obs=np.asarray(
                    local_obs,
                    dtype=np.float32,
                ).copy(),

                global_state=np.asarray(
                    global_state,
                    dtype=np.float32,
                ).copy(),

                action=int(
                    action
                ),

                action_type=(
                    action_type
                ),

                action_source=(
                    action_source
                ),

                immediate_reward=float(
                    immediate_reward
                ),

                source_dc_id=(
                    agent_id
                ),

                target_dc_id=(
                    target_dc_id
                ),
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
    ) -> bool:

        job_id = str(job_id)

        trace = self._traces.get(
            job_id
        )

        if trace is None:
            return False

        unresolved_steps = [
            step
            for step
            in trace.routing_steps
            if (
                    step.action_type == "edge_dc"
                    and not step.successor_resolved
            )
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

        previous_step = (
            unresolved_steps[-1]
        )

        next_agent_id = str(
            next_agent_id
        )

        next_env_time = float(
            next_env_time
        )

        expected_target_dc_id = (
            previous_step.target_dc_id
        )

        if expected_target_dc_id is None:
            raise RuntimeError(
                "未解析 Edge predecessor 缺少 target_dc_id："
                f"job={job_id}, "
                f"sequence={previous_step.sequence_index}"
            )

        if next_agent_id != str(
                expected_target_dc_id
        ):
            raise RuntimeError(
                "Routing Causal Chain target 不一致："
                f"job={job_id}, "
                f"sequence={previous_step.sequence_index}, "
                f"source={previous_step.source_dc_id}, "
                f"expected_target={expected_target_dc_id}, "
                f"actual_next_agent={next_agent_id}"
            )

        # ==========================================================
        # 时间因果关系必须单调。
        #
        # successor 决策时间不能早于 predecessor 决策时间。
        # ==========================================================

        if (
                next_env_time
                < float(
            previous_step.env_time
        )
        ):
            raise RuntimeError(
                "Routing Causal Chain 时间倒退："
                f"job={job_id}, "
                f"previous_time={previous_step.env_time}, "
                f"next_time={next_env_time}"
            )

        previous_step.next_agent_id = str(
            next_agent_id
        )

        previous_step.next_agent_index = int(
            next_agent_index
        )

        previous_step.next_env_time = float(
            next_env_time
        )

        previous_step.next_local_obs = np.asarray(
            next_local_obs,
            dtype=np.float32,
        ).copy()

        previous_step.next_global_state = (
            np.asarray(
                next_global_state,
                dtype=np.float32,
            ).copy()
        )

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

        job_id = str(
            job_id
        )

        dc_id = str(
            dc_id
        )

        # ==========================================================
        # Host Step 绝不能凭空创建 Job Trace。
        #
        # 正确因果链必须已经存在：
        #
        #   Routing ... -> Self
        #
        # Host 才能发生。
        # ==========================================================

        trace = self.get_trace(
            job_id
        )

        if not trace.routing_steps:
            raise RuntimeError(
                "Host Decision 出现时 Job 没有任何 Routing Step："
                f"job={job_id}, "
                f"dc={dc_id}"
            )

        last_routing_step = (
            trace.routing_steps[-1]
        )

        if (
                last_routing_step.action_type
                != "self"
        ):
            raise RuntimeError(
                "Host Decision 的上一因果节点不是 Self Routing："
                f"job={job_id}, "
                f"last_routing_action="
                f"{last_routing_step.action_type}"
            )

        if (
                last_routing_step.source_dc_id
                != dc_id
        ):
            raise RuntimeError(
                "Host Decision DC 与 Self Routing DC 不一致："
                f"job={job_id}, "
                f"self_dc={last_routing_step.source_dc_id}, "
                f"host_dc={dc_id}"
            )

        if trace.terminal:
            raise RuntimeError(
                "不能为 terminal Job 添加 Host Step："
                f"job={job_id}"
            )

        if trace.host_step is not None:
            raise RuntimeError(
                "同一个 Job 出现了重复 Host Decision："
                f"job={job_id}"
            )

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

            action_source=str(
                action_source
            ),
        )

    def record_host_result(
            self,
            *,
            job_id: str,
            result: str,
            env_time: float,
    ) -> None:

        trace = self._traces.get(
            str(job_id)
        )

        if trace is None:
            raise RuntimeError(
                "找不到 Host Result 对应的 "
                f"Job Trace：{job_id}"
            )

        if trace.host_step is None:
            raise RuntimeError(
                "Job 尚无 Host Step，"
                "却收到 Host execution result："
                f"job={job_id}"
            )

        trace.host_step.execution_result = str(
            result
        )

        trace.host_step.result_time = float(
            env_time
        )

    def record_reward_event(
            self,
            *,
            job_id: str,
            env_time: float,
            reward_delta: float,
            reason: str,
            terminal: bool,
    ) -> None:

        job_id = str(
            job_id
        )

        # ==========================================================
        # Environment terminal/delayed outcome 必须属于
        # 一个已经存在的 Job Causal Trace。
        #
        # 不允许 terminal event 凭空创建 Trace，
        # 否则会掩盖 Routing Decision 丢失问题。
        # ==========================================================

        trace = self.get_trace(
            job_id
        )

        trace.reward_events.append(
            PendingRewardEvent(
                job_id=str(job_id),
                env_time=float(env_time),
                reward_delta=float(
                    reward_delta
                ),
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
            trace.terminal_reason = str(
                reason
            )
            trace.terminal_time = float(
                env_time
            )

    def mark_terminal(
            self,
            *,
            job_id: str,
            env_time: float,
            reason: str,
    ) -> None:

        job_id = str(
            job_id
        )

        trace = self.get_trace(
            job_id
        )

        if trace.terminal:
            raise RuntimeError(
                "Job Trace 重复 terminal："
                f"job={job_id}, "
                f"old={trace.terminal_reason}, "
                f"new={reason}"
            )

        trace.terminal = True
        trace.terminal_reason = str(
            reason
        )
        trace.terminal_time = float(
            env_time
        )

    def _validate_trace_for_finalize(self, trace: PendingJobTrace,) -> None:
        """
        Job Terminal 后，在真正 Finalize 之前，
        对整条 Job Causal Trace 做一次最终结构验证。

        本函数只验证“因果事实是否完整”，
        不进行 Reward Credit Assignment。
        """

        job_id = str(
            trace.job_id
        )

        # ==========================================================
        # 1. Job 必须已经 terminal
        # ==========================================================

        if not trace.terminal:
            raise RuntimeError(
                "不能 Finalize 尚未 terminal 的 Job："
                f"job={job_id}"
            )

        if trace.terminal_reason is None:
            raise RuntimeError(
                "Terminal Job 缺少 terminal_reason："
                f"job={job_id}"
            )

        if trace.terminal_time is None:
            raise RuntimeError(
                "Terminal Job 缺少 terminal_time："
                f"job={job_id}"
            )

        terminal_time = float(
            trace.terminal_time
        )

        # ==========================================================
        # 2. 必须至少发生过一次 Routing Decision
        # ==========================================================

        if not trace.routing_steps:
            raise RuntimeError(
                "Terminal Job 没有任何 Routing Step："
                f"job={job_id}"
            )

        routing_steps = (
            trace.routing_steps
        )

        # ==========================================================
        # 3. 逐条验证 Routing Chain
        # ==========================================================

        for index, step in enumerate(
                routing_steps
        ):
            if (
                    str(step.job_id)
                    != job_id
            ):
                raise RuntimeError(
                    "Routing Step job_id 与 Trace 不一致："
                    f"trace_job={job_id}, "
                    f"step_job={step.job_id}, "
                    f"sequence={index}"
                )

            if (
                    int(step.sequence_index)
                    != index
            ):
                raise RuntimeError(
                    "Routing sequence_index 不连续："
                    f"job={job_id}, "
                    f"expected={index}, "
                    f"actual={step.sequence_index}"
                )

            # ======================================================
            # Edge -> Edge 必须已经拥有真实 same-job successor
            # ======================================================

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

                if (
                        str(step.next_agent_id)
                        != str(step.target_dc_id)
                ):
                    raise RuntimeError(
                        "Edge Routing target 与 successor 不一致："
                        f"job={job_id}, "
                        f"sequence={index}, "
                        f"target={step.target_dc_id}, "
                        f"next_agent={step.next_agent_id}"
                    )

                # Terminal Job 不允许最后一条还是 Edge forwarding。
                if index + 1 >= len(
                        routing_steps
                ):
                    raise RuntimeError(
                        "Terminal Job 的 Routing Chain "
                        "不能以 edge_dc 结束："
                        f"job={job_id}, "
                        f"sequence={index}"
                    )

                next_step = (
                    routing_steps[
                        index + 1
                        ]
                )

                if (
                        str(next_step.agent_id)
                        != str(step.next_agent_id)
                ):
                    raise RuntimeError(
                        "相邻 Routing Step 因果断裂："
                        f"job={job_id}, "
                        f"sequence={index}, "
                        f"resolved_next={step.next_agent_id}, "
                        f"next_step_agent={next_step.agent_id}"
                    )

            else:
                # ==================================================
                # Self / Cloud / Drop 都是 Routing terminal action。
                #
                # 后面不能再有同 Job Routing Step。
                # ==================================================

                if (
                        index
                        != len(routing_steps) - 1
                ):
                    raise RuntimeError(
                        "Routing terminal action 后 "
                        "仍然存在新的 Routing Step："
                        f"job={job_id}, "
                        f"sequence={index}, "
                        f"action_type={step.action_type}"
                    )

        # ==========================================================
        # 4. 最后一条 Routing 必须明确终止 Routing
        # ==========================================================

        final_routing_step = (
            routing_steps[-1]
        )

        final_action_type = str(
            final_routing_step.action_type
        )

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

        # ==========================================================
        # 5. Self -> Host 因果关系
        # ==========================================================

        if final_action_type == "self":

            if trace.host_step is None:
                raise RuntimeError(
                    "Self Routing 后 Job 已 terminal，"
                    "但缺少 Host Step："
                    f"job={job_id}"
                )

            host_step = (
                trace.host_step
            )

            if (
                    str(host_step.job_id)
                    != job_id
            ):
                raise RuntimeError(
                    "Host Step job_id 不一致："
                    f"trace_job={job_id}, "
                    f"host_job={host_step.job_id}"
                )

            if (
                    str(host_step.dc_id)
                    != str(
                final_routing_step
                        .source_dc_id
            )
            ):
                raise RuntimeError(
                    "Host Step DC 与最终 Self DC 不一致："
                    f"job={job_id}, "
                    f"self_dc="
                    f"{final_routing_step.source_dc_id}, "
                    f"host_dc={host_step.dc_id}"
                )

            if (
                    host_step.execution_result
                    is None
            ):
                raise RuntimeError(
                    "Host Step 尚未记录 execution_result："
                    f"job={job_id}"
                )

            if (
                    host_step.result_time
                    is None
            ):
                raise RuntimeError(
                    "Host Step 尚未记录 result_time："
                    f"job={job_id}"
                )

        # ==========================================================
        # 6. Cloud / Drop 不允许存在 Host Step
        # ==========================================================

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

        # ==========================================================
        # 7. Terminal Event 一致性
        #
        # Forced Drop 特殊：
        #   penalty 已经保存在 Routing immediate_reward 中，
        #   所以没有 terminal RewardEvent。
        #
        # 其他完成/失败：
        #   必须恰好有一个 terminal RewardEvent。
        # ==========================================================

        terminal_events = [
            event
            for event
            in trace.reward_events
            if event.terminal
        ]

        if final_action_type == "drop":

            if (
                    str(trace.terminal_reason)
                    != "forced_drop"
            ):
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

            terminal_event = (
                terminal_events[0]
            )

            if (
                    str(terminal_event.reason)
                    != str(trace.terminal_reason)
            ):
                raise RuntimeError(
                    "Terminal Event reason "
                    "与 Trace terminal_reason 不一致："
                    f"job={job_id}, "
                    f"event_reason={terminal_event.reason}, "
                    f"trace_reason={trace.terminal_reason}"
                )

        # ==========================================================
        # 8. 所有 Reward Event 都不能发生在 terminal 以后
        # ==========================================================

        for event in trace.reward_events:

            if (
                    float(event.env_time)
                    > terminal_time + 1e-9
            ):
                raise RuntimeError(
                    "Reward Event 时间晚于 Job terminal："
                    f"job={job_id}, "
                    f"event_time={event.env_time}, "
                    f"terminal_time={terminal_time}"
                )

    def _build_terminal_host_transition(self, trace: PendingJobTrace,) -> Optional[HostTransition]:
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

        job_id = str(
            trace.job_id
        )

        # ==========================================================
        # 1. 没有 Host Decision 的 Job 不产生 Host Transition。
        #
        # 正常对应：
        #   Cloud
        #   Forced Drop
        # ==========================================================

        host_step = trace.host_step

        if host_step is None:
            return None

        # ==========================================================
        # 2. 有 Host Step 时，最终 Routing 必须是 Self。
        #
        # _validate_trace_for_finalize() 已经检查过一次，
        # 这里再次做防御性检查，避免未来调用路径发生改变。
        # ==========================================================

        if not trace.routing_steps:
            raise RuntimeError(
                "构造 HostTransition 时 "
                "Job 没有 Routing Step："
                f"job={job_id}"
            )

        final_routing_step = (
            trace.routing_steps[-1]
        )

        if (
                str(
                    final_routing_step.action_type
                )
                != "self"
        ):
            raise RuntimeError(
                "存在 Host Step，"
                "但最终 Routing action 不是 Self："
                f"job={job_id}, "
                f"final_action="
                f"{final_routing_step.action_type}"
            )

        # ==========================================================
        # 3. Host Step 必须已经获得真实执行结果。
        # ==========================================================

        if host_step.execution_result is None:
            raise RuntimeError(
                "构造 HostTransition 时 "
                "Host execution_result 尚未记录："
                f"job={job_id}"
            )

        # ==========================================================
        # 4. Job 必须已经真正 terminal。
        # ==========================================================

        if not trace.terminal:
            raise RuntimeError(
                "不能从未 terminal Job "
                "构造 HostTransition："
                f"job={job_id}"
            )

        if trace.terminal_reason is None:
            raise RuntimeError(
                "构造 HostTransition 时 "
                "缺少 terminal_reason："
                f"job={job_id}"
            )

        if trace.terminal_time is None:
            raise RuntimeError(
                "构造 HostTransition 时 "
                "缺少 terminal_time："
                f"job={job_id}"
            )

        host_decision_time = float(
            host_step.env_time
        )

        terminal_time = float(
            trace.terminal_time
        )

        if (
                terminal_time
                < host_decision_time
        ):
            raise RuntimeError(
                "Host terminal_time 早于 Host decision_time："
                f"job={job_id}, "
                f"decision_time={host_decision_time}, "
                f"terminal_time={terminal_time}"
            )

        # ==========================================================
        # 5. Host-level Reward
        #
        # Host SAC 只有一个 decision，因此这条 terminal
        # transition 的 reward 应表示：
        #
        #   从 Host decision 开始
        #        ↓
        #   到该 Job terminal
        #
        # 期间产生的全部 delayed Job reward。
        #
        # 特别注意：
        # Host decision 以前由 Routing 产生的 reward
        # 不应该重复归属于 Host SAC。
        # ==========================================================

        host_reward_events = [
            event
            for event
            in trace.reward_events
            if (
                    float(event.env_time)
                    + 1e-9
                    >= host_decision_time
            )
        ]

        # ==========================================================
        # 6. Self -> Host Job 必须恰好有一个 terminal event。
        #
        # 当前环境可能是：
        #
        #   completed
        #   waiting_timeout
        #   local_host_arrival_timeout
        #   local_host_resource_failure
        #
        # ==========================================================

        host_terminal_events = [
            event
            for event
            in host_reward_events
            if bool(
                event.terminal
            )
        ]

        if len(
                host_terminal_events
        ) != 1:
            raise RuntimeError(
                "Host Job 必须恰好存在一个 "
                "terminal RewardEvent："
                f"job={job_id}, "
                f"terminal_event_count="
                f"{len(host_terminal_events)}"
            )

        terminal_event = (
            host_terminal_events[0]
        )

        if (
                str(terminal_event.reason)
                != str(trace.terminal_reason)
        ):
            raise RuntimeError(
                "Host terminal RewardEvent "
                "与 Trace terminal_reason 不一致："
                f"job={job_id}, "
                f"event_reason={terminal_event.reason}, "
                f"trace_reason={trace.terminal_reason}"
            )

        # ==========================================================
        # 7. One-Job Host Return
        #
        # 当前 active 环境基本只有一个 terminal reward event，
        # 因而现在通常等价于：
        #
        #     host_reward = terminal_event.reward_delta
        #
        # 这里仍采用 sum，是为了以后如果重新启用
        # Host decision 后的 waiting cost 等 delayed cost，
        # 不需要改变 HostTransition 定义。
        # ==========================================================

        host_reward = float(
            sum(
                float(event.reward_delta)
                for event
                in host_reward_events
            )
        )

        # ==========================================================
        # 8. 保存 Host 决策时的 Observation。
        # ==========================================================

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

        # ==========================================================
        # 9. Host 层不存在 same-job 下一次 Host decision。
        #
        # 所以禁止使用：
        #
        #   下一个全局 Job 的 Host Observation
        #
        # 或：
        #
        #   started 后重新构造的 Host Observation
        #
        # 作为 next state。
        #
        # 标准 SAC 接口使用全 0 terminal next state。
        # ==========================================================

        next_host_obs = np.zeros_like(
            host_obs,
            dtype=np.float32,
        )

        # ==========================================================
        # 10. 正式创建唯一 Host terminal transition。
        # ==========================================================

        return HostTransition(
            job_id=job_id,

            dc_id=str(
                host_step.dc_id
            ),

            decision_time=(
                host_decision_time
            ),

            host_obs=(
                host_obs
            ),

            action=int(
                host_step.action
            ),

            host_id=str(
                host_step.host_id
            ),

            action_source=str(
                host_step.action_source
            ),

            reward=(
                host_reward
            ),

            next_host_obs=(
                next_host_obs
            ),

            # 一个 Host Decision 对应一个完整 Job lifecycle，
            # 因而 Host-level transition 永远 terminal。
            terminated=True,

            truncated=False,

            done=True,

            execution_result=str(
                host_step.execution_result
            ),

            terminal_reason=str(
                trace.terminal_reason
            ),

            terminal_time=(
                terminal_time
            ),
        )

    def _build_finalized_routing_transitions(self,trace: PendingJobTrace,) -> tuple[RoutingTransition, ...]:
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

        job_id = str(
            trace.job_id
        )

        routing_steps = list(
            trace.routing_steps
        )

        # ==========================================================
        # 找出真正由 Actor 控制的 Routing Decisions。
        #
        # random：
        #     虽然不是 policy 网络采样，
        #     但属于正常探索经验，可以训练。
        #
        # policy：
        #     正常策略经验。
        #
        # forced：
        #     Environment 强制行为，不直接训练。
        # ==========================================================

        actor_step_indices = [
            index
            for index, step
            in enumerate(routing_steps)
            if step.action_source
               in {
                   "random",
                   "policy",
               }
        ]

        if not actor_step_indices:
            return tuple()

        # ==========================================================
        # 初始 reward：
        # 每一步先使用自己真正发生时的 immediate_reward。
        # ==========================================================

        final_rewards = {
            index: float(
                routing_steps[
                    index
                ].immediate_reward
            )
            for index
            in actor_step_indices
        }

        # ==========================================================
        # Environment delayed reward event：
        #
        # 将 reward 归给 event 发生之前最近一次
        # Actor-controlled Routing Decision。
        #
        # Terminal event 因而自然归给最后一次 Actor Decision。
        # ==========================================================

        for event in trace.reward_events:

            event_time = float(
                event.env_time
            )

            candidate_indices = [
                index
                for index
                in actor_step_indices
                if (
                        float(
                            routing_steps[
                                index
                            ].env_time
                        )
                        <= event_time + 1e-9
                )
            ]

            if not candidate_indices:
                continue

            target_index = (
                candidate_indices[-1]
            )

            final_rewards[
                target_index
            ] += float(
                event.reward_delta
            )

        # ==========================================================
        # Forced action 自己不能训练。
        #
        # 但 forced drop 的真实 terminal penalty
        # 必须归因到它之前最近一次 Actor-controlled action。
        #
        # 这实现第十七步已经确定的：
        #
        #   Actor Edge
        #       ↓
        #   Forced Drop
        #
        # 转化成：
        #
        #   Actor Edge
        #   reward += drop penalty
        #   done = True
        # ==========================================================

        for forced_index, forced_step in enumerate(
                routing_steps
        ):

            if (
                    forced_step.action_source
                    != "forced"
            ):
                continue

            forced_reward = float(
                forced_step.immediate_reward
            )

            if abs(forced_reward) <= 1e-12:
                continue

            predecessor_indices = [
                index
                for index
                in actor_step_indices
                if index < forced_index
            ]

            if not predecessor_indices:
                # Job 从一开始就只能 forced drop，
                # 没有 Actor 决策，不产生训练经验。
                continue

            target_index = (
                predecessor_indices[-1]
            )

            final_rewards[
                target_index
            ] += forced_reward

        finalized_transitions = []

        # ==========================================================
        # 构造每一个 Actor-controlled Routing Transition。
        # ==========================================================

        for actor_position, step_index in enumerate(
                actor_step_indices
        ):

            step = routing_steps[
                step_index
            ]

            # ------------------------------------------------------
            # 下一条原始 Routing Step。
            #
            # 注意不是下一个全局 Job。
            # ------------------------------------------------------

            next_raw_step = (
                routing_steps[
                    step_index + 1
                    ]
                if (
                        step_index + 1
                        < len(routing_steps)
                )
                else None
            )

            # ======================================================
            # Edge -> 正常 Actor Routing：
            #
            # 可以通过 same-job successor bootstrap。
            # ======================================================

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

                next_agent_id = str(
                    step.next_agent_id
                )

                next_agent_index = int(
                    step.next_agent_index
                )

                next_env_time = float(
                    step.next_env_time
                )

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

            else:

                # ==================================================
                # Routing terminal：
                #
                #   Self
                #   Cloud
                #   Actor Drop
                #
                # 或：
                #
                #   Edge -> Forced Drop
                #
                # 都不能继续 bootstrap。
                # ==================================================

                next_agent_id = None
                next_agent_index = -1

                next_env_time = float(
                    trace.terminal_time
                )

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

                terminal_reason = str(
                    trace.terminal_reason
                )

            finalized_transitions.append(
                RoutingTransition(
                    job_id=job_id,

                    agent_id=str(
                        step.agent_id
                    ),

                    agent_index=int(
                        step.agent_index
                    ),

                    env_time=float(
                        step.env_time
                    ),

                    local_obs=np.asarray(
                        step.local_obs,
                        dtype=np.float32,
                    ).copy(),

                    global_state=np.asarray(
                        step.global_state,
                        dtype=np.float32,
                    ).copy(),

                    action=int(
                        step.action
                    ),

                    action_type=str(
                        step.action_type
                    ),

                    action_source=str(
                        step.action_source
                    ),

                    reward=float(
                        final_rewards[
                            step_index
                        ]
                    ),

                    next_agent_id=(
                        next_agent_id
                    ),

                    next_agent_index=(
                        next_agent_index
                    ),

                    next_env_time=(
                        next_env_time
                    ),

                    next_local_obs=(
                        next_local_obs
                    ),

                    next_global_state=(
                        next_global_state
                    ),

                    terminated=(
                        terminated
                    ),

                    truncated=(
                        truncated
                    ),

                    done=(
                        done
                    ),

                    terminal_reason=(
                        terminal_reason
                    ),
                )
            )

        return tuple(
            finalized_transitions
        )

    def _build_finalized_trace(self,trace: PendingJobTrace, ) -> FinalizedJobTrace:
        """
        从已经通过结构验证的 PendingJobTrace
        生成完全独立的 FinalizedJobTrace 副本。

        使用 deepcopy 的原因：
            Finalized Trace 不应该继续引用 Pending 对象中的
            numpy array / Step 实例。
        """

        if trace.terminal_reason is None:
            raise RuntimeError(
                "Finalized Trace 缺少 terminal_reason："
                f"job={trace.job_id}"
            )

        if trace.terminal_time is None:
            raise RuntimeError(
                "Finalized Trace 缺少 terminal_time："
                f"job={trace.job_id}"
            )

        finalized_routing_steps = tuple(
            copy.deepcopy(step)
            for step
            in trace.routing_steps
        )

        finalized_host_step = (
            None
            if trace.host_step is None
            else copy.deepcopy(
                trace.host_step
            )
        )

        finalized_reward_events = tuple(
            copy.deepcopy(event)
            for event
            in trace.reward_events
        )

        finalized_routing_transitions = (
            self._build_finalized_routing_transitions(
                trace
            )
        )

        finalized_host_transition = (
            self._build_terminal_host_transition(
                trace
            )
        )

        return FinalizedJobTrace(
            job_id=str(
                trace.job_id
            ),

            routing_steps=(
                finalized_routing_steps
            ),

            # 第十九步：
            # Job terminal 后才生成的正式 Routing Replay experience。
            routing_transitions=(
                finalized_routing_transitions
            ),

            host_step=(
                finalized_host_step
            ),

            reward_events=(
                finalized_reward_events
            ),

            host_transition=(
                finalized_host_transition
            ),

            terminal_reason=str(
                trace.terminal_reason
            ),

            terminal_time=float(
                trace.terminal_time
            ),
        )

    def finalize_terminal_trace(self,job_id: str,) -> FinalizedJobTrace:
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

        job_id = str(
            job_id
        )

        if (
                job_id
                in self._finalized_traces
        ):
            raise RuntimeError(
                "同一个 Job 被重复 Finalize："
                f"job={job_id}"
            )

        trace = self.get_trace(
            job_id
        )

        # ==========================================================
        # 第一阶段：
        # 必须先保证完整链结构正确。
        # ==========================================================

        self._validate_trace_for_finalize(
            trace
        )

        # ==========================================================
        # 第二阶段：
        # 创建 Finalized 副本。
        # ==========================================================

        finalized_trace = (
            self._build_finalized_trace(
                trace
            )
        )

        # ==========================================================
        # 第三阶段：
        # 只有前面全部成功以后才移动。
        #
        # 这样如果 validation / deepcopy 出错，
        # 原 Pending Trace 仍完整保留，便于调试。
        # ==========================================================

        self._finalized_traces[
            job_id
        ] = finalized_trace

        del self._traces[
            job_id
        ]

        return finalized_trace

    def has_finalized_trace(self,job_id: str,) -> bool:

        return (
                str(job_id)
                in self._finalized_traces
        )

    def get_finalized_trace(self,job_id: str,) -> FinalizedJobTrace:

        job_id = str(
            job_id
        )

        trace = self._finalized_traces.get(
            job_id
        )

        if trace is None:
            raise KeyError(
                "找不到 Finalized Job Trace："
                f"job={job_id}"
            )

        return trace

    def pop_finalized_trace( self,job_id: str,) -> FinalizedJobTrace:
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

        job_id = str(
            job_id
        )

        trace = self.get_finalized_trace(
            job_id
        )

        del self._finalized_traces[
            job_id
        ]

        return trace

    def get_finalized_host_transition( self,job_id: str,) -> Optional[HostTransition]:
        """
        返回指定已 Finalize Job 对应的 Local Host SAC
        terminal transition。

        Self Job:
            HostTransition

        Cloud / Drop Job:
            None
        """

        finalized_trace = (
            self.get_finalized_trace(
                job_id
            )
        )

        return (
            finalized_trace.host_transition
        )

    def is_terminal(self, job_id: str,) -> bool:

        job_id = str(
            job_id
        )

        # 已 Finalize 的 Job 必然已经 terminal。
        if (
                job_id
                in self._finalized_traces
        ):
            return True

        trace = self._traces.get(
            job_id
        )

        if trace is None:
            return False

        return bool(
            trace.terminal
        )

    def remove_trace(self,job_id: str,) -> Optional[PendingJobTrace]:

        return self._traces.pop(
            str(job_id),
            None,
        )

    @property
    def open_trace_count(self,) -> int:
        """
        尚未 terminal 的 Pending Trace 数。
        """

        return sum(
            1
            for trace
            in self._traces.values()
            if not trace.terminal
        )

    @property
    def pending_trace_count(self,) -> int:
        """
        尚未完成 Finalize 的 Trace 总数。

        包括：
            open
            terminal-but-not-finalized
        """

        return len(
            self._traces
        )

    @property
    def terminal_trace_count(self,) -> int:
        """
        已经 terminal 的 Job 总数。

        正常情况下 terminal 后会立即 Finalize，
        因而主要来自 _finalized_traces。
        """

        pending_terminal_count = sum(
            1
            for trace
            in self._traces.values()
            if trace.terminal
        )

        return (
                pending_terminal_count
                + len(
            self._finalized_traces
        )
        )

    @property
    def finalized_trace_count(self,) -> int:

        return len(
            self._finalized_traces
        )

    @property
    def finalized_host_transition_count( self,) -> int:
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
            for trace
            in self._finalized_traces.values()
            if (
                    trace.host_transition
                    is not None
            )
        )

    @property
    def total_trace_count(self,) -> int:

        return (
                len(self._traces)
                + len(
            self._finalized_traces
        )
        )

    def assert_no_open_trace(self,) -> None:
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
                "terminal": bool(
                    trace.terminal
                ),
                "terminal_reason": (
                    trace.terminal_reason
                ),
                "routing_steps": len(
                    trace.routing_steps
                ),
                "has_host_step": (
                        trace.host_step
                        is not None
                ),
            }
            for job_id, trace
            in self._traces.items()
        ]

        raise RuntimeError(
            "Episode 已结束，但仍存在未 Finalize 的 "
            "Pending Job Causal Trace："
            f"{remaining_details[:20]}"
        )

    def reset_episode(self,) -> None:
        """
        开始新 Episode 前清理上一 Episode Trace。

        当前第十六步 Finalized Trace 只保留到 Episode 边界。

        后续双 ReplayBuffer 建成以后，
        Finalized Trace 会在这里之前被 pop 并写入经验池。
        """

        # Pending 区只要还有任何内容，都说明上一轮
        # 有 Job 没有完成 Finalize。
        if self._traces:
            self.assert_no_open_trace()

        self._traces.clear()

        # 当前 Finalized Trace 暂时作为本 Episode 的完整结果归档。
        # 下一 Episode 开始时清除。
        self._finalized_traces.clear()

    def assert_no_unflushed_finalized_trace(self,) -> None:

        if not self._finalized_traces:
            return

        raise RuntimeError(
            "Episode 已结束，但仍存在已经 Finalize "
            "却没有写入 ReplayBuffer 的 Job："
            f"{list(self._finalized_traces.keys())[:20]}"
        )


