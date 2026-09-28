"""BGH-MASAC 的历史反馈统计层。

第 1 步职责边界：

* 只从已经 Finalize 的任务因果链维护 source -> target 历史统计；
* 不保存或读取远端 DC 的实时 CPU、GPU、Queue、Host 状态；
* 不负责 Bayesian Posterior、拥塞压力或 Actor 引导。

第 3 步新增的 Evidence Adapter 独立负责 Bayesian 路径；
本 Store 保留原有 EWMA 统计接口，确保历史反馈兼容性和 Zero-Diff。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np

from pending_job_trace import FinalizedJobTrace
from bayesian_congestion_game import BayesianBeliefStore
from bayesian_evidence import build_bayesian_evidence_from_finalized_trace

# 第 1 步：BGH 当前采用“训练脚本目录直接加入 sys.path”的入口方式。
# 因此这里必须使用脚本式顶层导入，不能与 train_bgh_masac.py 混用相对导入。
# Bayesian Evidence 的正式转换由 bayesian_evidence.py 统一负责；
# 本文件的旧 EWMA 统计链保持独立，避免改变已有反馈结果。

__all__ = (
    "ShortWindowFeedbackStore",
    "NeighborPairFeedbackState",
    "NeighborHistoricalFeedbackStore",
)


class ShortWindowFeedbackStore:
    """Same seven observation fields, computed only from current-window outcomes.

Legacy *_ewma names are kept for observation/checkpoint dimensional compatibility;
their values here are window means. No EWMA or cross-episode history is retained.
"""
    def __init__(self, env, *, window_s, confidence_scale_samples=20.0):
        self.env = env
        self.edge_dc_ids = tuple(map(str, env.edge_dc_ids))
        self.store = BayesianBeliefStore(self.edge_dc_ids, window_s=window_s,
                                       confidence_scale=confidence_scale_samples)
        self.completion_time_scale_s = max(float(env.max_job_duration) * float(env.drop_deadline_ratio), 1e-8)
        self.reset_episode_counters()

    def reset_episode_counters(self):
        self.store.reset()
        self.terminal_jobs_seen = 0
        self.pair_samples = 0

    def update_from_finalized_trace(self, trace):
        self.store.advance_time(float(self.env.current_time))
        self.terminal_jobs_seen += 1
        evidence = build_bayesian_evidence_from_finalized_trace(trace, self.env)
        for item in evidence:
            self.store.update(item, update_clock=float(self.env.current_time))
        self.pair_samples += len(evidence)
        return len(evidence)

    def get_feedback(self, source_dc_id, target_dc_id):
        self.store.advance_time(float(self.env.current_time))
        if source_dc_id == target_dc_id or target_dc_id not in self.edge_dc_ids:
            return None
        records = self.store.records(source_dc_id, target_dc_id)
        if not records:
            return None
        count = len(records)
        completed = [e.completion_time_s for e in records if e.completion_time_s is not None]
        # No completed sample is unknown delay, not zero completion time.
        delay = (sum(completed) / len(completed)) if completed else None
        return dict(success_ewma=sum(e.success for e in records) / count,
            sla_success_ewma=sum(e.sla_satisfied for e in records) / count,
            drop_ewma=sum(not e.success for e in records) / count,
            completion_time_ewma=delay / (delay + self.completion_time_scale_s) if delay is not None else 0.5,
            reforward_ewma=sum(e.reforwarded for e in records) / count,
            feedback_age=min(1.0, (self.store.now - max(e.timestamp for e in records)) / self.store.window_s),
            sample_confidence=self.store.get_confidence(source_dc_id, target_dc_id))

    def get_raw_feedback(self, source_dc_id, target_dc_id):
        encoded = self.get_feedback(source_dc_id, target_dc_id)
        if encoded is None:
            return None
        return {**encoded, "sample_count": len(self.store.records(source_dc_id, target_dc_id)),
                "aggregation": "short_time_window"}

    def snapshot_for_source(self, source_dc_id):
        result = {}
        for target in self.edge_dc_ids:
            feedback = self.get_feedback(source_dc_id, target)
            if feedback is not None:
                result[target] = {"raw": self.get_raw_feedback(source_dc_id, target), "encoded": feedback}
        return result

    def source_summary(self, source_dc_id):
        self.store.advance_time(float(self.env.current_time))
        counts = [len(self.store.records(source_dc_id, j)) for j in self.edge_dc_ids if j != source_dc_id]
        return dict(outgoing_pair_count=sum(n > 0 for n in counts), outgoing_sample_count=sum(counts))

    def summary(self):
        counts = [self.source_summary(i) for i in self.edge_dc_ids]
        return dict(terminal_jobs_seen=self.terminal_jobs_seen,
            total_pair_samples=sum(c["outgoing_sample_count"] for c in counts),
            active_pair_count=sum(c["outgoing_pair_count"] for c in counts),
            episode_terminal_jobs_seen=self.terminal_jobs_seen, episode_pair_samples=self.pair_samples)

@dataclass
class NeighborPairFeedbackState:
    """
    保存一个有向 Neighbor Pair：

        source_dc -> target_dc

    的历史反馈状态。

    注意：
        这里只记录历史结果，
        不记录 target DC 当前实时 CPU/GPU/Queue 等状态。

    因而不会破坏 Routing Actor 的部分可观测假设。
    """

    source_dc_id: str
    target_dc_id: str

    # 该有向 pair 一共观察到多少次 Edge -> Edge 转发。
    sample_count: int = 0

    # 其中有多少个 sample 最终成功完成，
    # 用于 completion_time EWMA 的独立初始化。
    completion_sample_count: int = 0

    success_ewma: float = 0.0
    sla_success_ewma: float = 0.0
    drop_ewma: float = 0.0

    # 保存真实秒数。
    # get_feedback() 时再归一化到 [0, 1]。
    completion_time_ewma_s: Optional[float] = None

    reforward_ewma: float = 0.0

    # 使用“已观察 terminal Job 数”作为历史时钟。
    #
    # 不能直接使用 env.current_time，
    # 因为每个 Episode reset 后 simulation time 会归零。
    last_feedback_clock: int = 0

    last_job_id: Optional[str] = None
    last_terminal_reason: Optional[str] = None


class NeighborHistoricalFeedbackStore:
    """
    Neighbor Historical Feedback Store。

    第二十九步职责：

        1. 从已经完整 Finalize 的 Job Causal Trace
           中收集 source -> target 历史结果；

        2. 使用 EWMA 保存历史统计；

        3. 为以后 RoutingObservationBuilder
           提供 get_feedback() 接口；

        4. 当前阶段只收集、不参与策略决策。

    特别注意：

        本 Store 不能读取：

            Remote DC current CPU load
            Remote DC current GPU load
            Remote DC current queue
            Remote Host state

        它只能消费已经结束任务的历史结果。

    因此：

        Historical Feedback != Remote Real-Time State
    """

    def __init__(
            self,
            env: Any,
            ewma_alpha: float = 0.10,
            age_scale_samples: float = 100.0,
            confidence_scale_samples: float = 20.0,
    ) -> None:

        self.env = env

        self.edge_dc_ids = [
            str(dc_id)
            for dc_id
            in env.edge_dc_ids
        ]

        self.edge_dc_id_set = set(
            self.edge_dc_ids
        )

        self.ewma_alpha = float(
            ewma_alpha
        )

        self.age_scale_samples = float(
            age_scale_samples
        )

        self.confidence_scale_samples = float(
            confidence_scale_samples
        )

        if not (
                0.0
                < self.ewma_alpha
                <= 1.0
        ):
            raise ValueError(
                "NEIGHBOR_FEEDBACK_EWMA_ALPHA "
                "必须位于 (0, 1]："
                f"{self.ewma_alpha}"
            )

        if self.age_scale_samples <= 0.0:
            raise ValueError(
                "NEIGHBOR_FEEDBACK_AGE_SCALE_SAMPLES "
                "必须 > 0。"
            )

        if (
                self.confidence_scale_samples
                <= 0.0
        ):
            raise ValueError(
                "NEIGHBOR_FEEDBACK_CONFIDENCE_SCALE_SAMPLES "
                "必须 > 0。"
            )

        # ======================================================
        # Completion Time Normalization Scale
        #
        # 使用 Environment 已经采用的全局时间尺度：
        #
        #   max_job_duration * drop_deadline_ratio
        #
        # 这里只用于把历史 completion EWMA 映射到 [0,1]，
        # 不读取任何 Remote DC 当前状态。
        # ======================================================

        self.completion_time_scale_s = max(
            float(
                env.max_job_duration
            )
            * float(
                env.drop_deadline_ratio
            ),
            1e-8,
        )

        # ======================================================
        # Persistent Pair State
        #
        # 这里不能在每个 Episode reset。
        #
        # Historical Feedback 的意义就是：
        #   Episode N+1 仍然能够保留 Episode N 的历史经验。
        # ======================================================

        self._pair_states: Dict[
            tuple[str, str],
            NeighborPairFeedbackState,
        ] = {}

        # ======================================================
        # Global Historical Clock
        #
        # 每 Finalize 一个 Job +1。
        #
        # 使用 terminal-job count，
        # 而不是 simulation time，
        # 因为后者每 Episode 都会归零。
        # ======================================================

        self._feedback_clock: int = 0

        self._total_terminal_jobs_seen: int = 0
        self._total_pair_samples: int = 0

        # Episode-only counters。
        self._episode_terminal_jobs_seen: int = 0
        self._episode_pair_samples: int = 0

    # ==========================================================
    # Episode Counters
    # ==========================================================

    def reset_episode_counters(
            self,
    ) -> None:
        """
        只清空当前 Episode 的统计计数。

        绝对不能清空 _pair_states。

        否则 Historical Feedback 会退化成：
            “本 Episode Feedback”
        而不再是跨 Episode 历史信息。
        """

        self._episode_terminal_jobs_seen = 0
        self._episode_pair_samples = 0

    # ==========================================================
    # Basic Helpers
    # ==========================================================

    @staticmethod
    def _clip01(
            value: float,
    ) -> float:

        return float(
            np.clip(
                float(value),
                0.0,
                1.0,
            )
        )

    @staticmethod
    def _saturating_ratio(
            value: float,
            scale: float,
    ) -> float:
        """
        将非负量映射到 [0,1)：

            x / (x + scale)

        相比硬 clip：
            min(x / scale, 1)

        不会让大量较大的历史值全部塌缩到 1。
        """

        value = max(
            float(value),
            0.0,
        )

        scale = max(
            float(scale),
            1e-8,
        )

        return float(
            value
            / (
                value
                + scale
            )
        )

    def _update_binary_ewma(
            self,
            previous_value: float,
            sample_value: float,
            previous_sample_count: int,
    ) -> float:

        sample_value = (
            self._clip01(
                sample_value
            )
        )

        # 第一条样本直接作为初始值，
        # 避免从人为的 0 开始产生额外初始化偏置。
        if previous_sample_count <= 0:
            return sample_value

        return float(
            (
                1.0
                - self.ewma_alpha
            )
            * float(
                previous_value
            )
            + self.ewma_alpha
            * sample_value
        )

    def _get_or_create_pair(
            self,
            source_dc_id: str,
            target_dc_id: str,
    ) -> NeighborPairFeedbackState:

        source_dc_id = str(
            source_dc_id
        )

        target_dc_id = str(
            target_dc_id
        )

        if (
                source_dc_id
                not in self.edge_dc_id_set
        ):
            raise ValueError(
                "Neighbor Feedback 收到未知 source DC："
                f"{source_dc_id}"
            )

        if (
                target_dc_id
                not in self.edge_dc_id_set
        ):
            raise ValueError(
                "Neighbor Feedback 收到未知 target DC："
                f"{target_dc_id}"
            )

        if source_dc_id == target_dc_id:
            raise ValueError(
                "Neighbor Feedback 只统计 "
                "Edge -> Edge remote pair，"
                "source 不能等于 target："
                f"{source_dc_id}"
            )

        pair_key = (
            source_dc_id,
            target_dc_id,
        )

        pair_state = (
            self._pair_states.get(
                pair_key
            )
        )

        if pair_state is None:

            pair_state = (
                NeighborPairFeedbackState(
                    source_dc_id=(
                        source_dc_id
                    ),

                    target_dc_id=(
                        target_dc_id
                    ),
                )
            )

            self._pair_states[
                pair_key
            ] = pair_state

        return pair_state

    # ==========================================================
    # Pair Update
    # ==========================================================

    def _update_pair(
            self,
            *,
            source_dc_id: str,
            target_dc_id: str,
            job_id: str,
            terminal_reason: str,
            success_sample: float,
            sla_success_sample: float,
            drop_sample: float,
            completion_time_s: Optional[float],
            reforward_sample: float,
    ) -> None:

        pair_state = (
            self._get_or_create_pair(
                source_dc_id=(
                    source_dc_id
                ),

                target_dc_id=(
                    target_dc_id
                ),
            )
        )

        previous_sample_count = int(
            pair_state.sample_count
        )

        pair_state.success_ewma = (
            self._update_binary_ewma(
                previous_value=(
                    pair_state
                    .success_ewma
                ),

                sample_value=(
                    success_sample
                ),

                previous_sample_count=(
                    previous_sample_count
                ),
            )
        )

        pair_state.sla_success_ewma = (
            self._update_binary_ewma(
                previous_value=(
                    pair_state
                    .sla_success_ewma
                ),

                sample_value=(
                    sla_success_sample
                ),

                previous_sample_count=(
                    previous_sample_count
                ),
            )
        )

        pair_state.drop_ewma = (
            self._update_binary_ewma(
                previous_value=(
                    pair_state
                    .drop_ewma
                ),

                sample_value=(
                    drop_sample
                ),

                previous_sample_count=(
                    previous_sample_count
                ),
            )
        )

        pair_state.reforward_ewma = (
            self._update_binary_ewma(
                previous_value=(
                    pair_state
                    .reforward_ewma
                ),

                sample_value=(
                    reforward_sample
                ),

                previous_sample_count=(
                    previous_sample_count
                ),
            )
        )

        # ------------------------------------------------------
        # Completion Time 只在真正完成时更新。
        #
        # Drop Job 没有“真实完成时间”，
        # 不能用 drop deadline 或 terminal time
        # 冒充 completion sample。
        # ------------------------------------------------------

        if completion_time_s is not None:

            completion_time_s = max(
                float(
                    completion_time_s
                ),
                0.0,
            )

            if (
                    pair_state
                    .completion_sample_count
                    <= 0
                    or pair_state
                    .completion_time_ewma_s
                    is None
            ):
                pair_state.completion_time_ewma_s = (
                    completion_time_s
                )

            else:
                pair_state.completion_time_ewma_s = float(
                    (
                        1.0
                        - self.ewma_alpha
                    )
                    * float(
                        pair_state
                        .completion_time_ewma_s
                    )
                    + self.ewma_alpha
                    * completion_time_s
                )

            pair_state.completion_sample_count += 1

        pair_state.sample_count += 1

        pair_state.last_feedback_clock = int(
            self._feedback_clock
        )

        pair_state.last_job_id = str(
            job_id
        )

        pair_state.last_terminal_reason = str(
            terminal_reason
        )

        self._total_pair_samples += 1
        self._episode_pair_samples += 1

    # ==========================================================
    # Finalized Causal Trace -> Historical Feedback
    # ==========================================================

    def update_from_finalized_trace(
            self,
            finalized_trace: FinalizedJobTrace,
    ) -> int:
        """
        从一个已经 terminal 的完整 Job 因果链
        更新 Neighbor Historical Feedback。

        只处理真实：

            Edge DC -> Edge DC

        Routing Step。

        不处理：
            Self
            Cloud
            Drop

        对多跳链：

            DC1 -> DC3 -> DC5 -> Self

        会分别生成两个历史 sample：

            DC1 -> DC3
            DC3 -> DC5

        如果出现允许的循环：

            DC1 -> DC3 -> DC1

        两次 Edge step 同样分别统计。

        即：
            循环不被屏蔽，
            也不会因为 Pair 已经出现过就跳过 sample。
        """

        self._feedback_clock += 1

        self._total_terminal_jobs_seen += 1

        self._episode_terminal_jobs_seen += 1

        job_id = str(
            finalized_trace.job_id
        )

        job = self.env.job_map.get(
            job_id
        )

        if job is None:
            raise RuntimeError(
                "Neighbor Feedback 无法找到 terminal Job："
                f"{job_id}"
            )

        terminal_reason = str(
            finalized_trace
            .terminal_reason
        )

        completed = bool(
            terminal_reason
            == "completed"
        )

        success_sample = (
            1.0
            if completed
            else 0.0
        )

        drop_sample = (
            0.0
            if completed
            else 1.0
        )

        completion_time_s: Optional[
            float
        ] = None

        sla_success_sample = 0.0

        if completed:

            turnaround_time = (
                job.get_turnaround_time()
            )

            # 正常 completed Job 已有 finish_time。
            # 这里保留 terminal_time fallback，
            # 防止以后环境实现变化。
            if turnaround_time is None:

                turnaround_time = max(
                    float(
                        finalized_trace
                        .terminal_time
                    )
                    - float(
                        job.arrive_time
                    ),
                    0.0,
                )

            completion_time_s = max(
                float(
                    turnaround_time
                ),
                0.0,
            )

            sla_limit_s = (
                float(
                    self.env
                    .sla_deadline_ratio
                )
                * float(
                    job.duration
                )
            )

            sla_success_sample = (
                1.0
                if (
                    completion_time_s
                    <= sla_limit_s
                    + 1e-9
                )
                else 0.0
            )

        routing_steps = list(
            finalized_trace
            .routing_steps
        )

        pair_update_count = 0

        for step_index, routing_step in enumerate(
                routing_steps
        ):

            if (
                    str(
                        routing_step
                        .action_type
                    )
                    != "edge_dc"
            ):
                continue

            source_dc_id = str(
                routing_step
                .source_dc_id
            )

            if (
                    routing_step
                    .target_dc_id
                    is None
            ):
                raise RuntimeError(
                    "Finalized Edge Routing "
                    "缺少 target_dc_id："
                    f"job={job_id}, "
                    f"sequence="
                    f"{routing_step.sequence_index}"
                )

            target_dc_id = str(
                routing_step
                .target_dc_id
            )

            # ==================================================
            # Edge Routing 的后继 Routing Decision
            #
            # PendingJobTrace 已经保证：
            #   Edge -> Edge 必须有同 Job successor。
            #
            # 因此这里使用下一 Routing Step 判断：
            #   target 是否再次向外调度。
            # ==================================================

            next_step_index = (
                step_index + 1
            )

            if (
                    next_step_index
                    >= len(
                        routing_steps
                    )
            ):
                raise RuntimeError(
                    "Neighbor Feedback 找不到 "
                    "Edge Routing 的后继 Routing Step："
                    f"job={job_id}, "
                    f"sequence="
                    f"{routing_step.sequence_index}, "
                    f"source={source_dc_id}, "
                    f"target={target_dc_id}"
                )

            next_routing_step = (
                routing_steps[
                    next_step_index
                ]
            )

            if (
                    str(
                        next_routing_step
                        .agent_id
                    )
                    != target_dc_id
            ):
                raise RuntimeError(
                    "Neighbor Feedback 的 Routing "
                    "因果链 target 不一致："
                    f"job={job_id}, "
                    f"expected={target_dc_id}, "
                    f"actual="
                    f"{next_routing_step.agent_id}"
                )

            next_action_type = str(
                next_routing_step
                .action_type
            )

            # --------------------------------------------------
            # reforward：
            #
            # target 收到任务以后：
            #
            #   Self / Drop
            #       -> 0
            #
            #   Edge / Cloud
            #       -> 1
            #
            # Cloud 这里也视作“继续向外卸载”，
            # 因为 target 没有在本 DC 留下任务。
            # --------------------------------------------------

            reforward_sample = (
                1.0
                if next_action_type
                in {
                    "edge_dc",
                    "cloud",
                }
                else 0.0
            )

            self._update_pair(
                source_dc_id=(
                    source_dc_id
                ),

                target_dc_id=(
                    target_dc_id
                ),

                job_id=(
                    job_id
                ),

                terminal_reason=(
                    terminal_reason
                ),

                success_sample=(
                    success_sample
                ),

                sla_success_sample=(
                    sla_success_sample
                ),

                drop_sample=(
                    drop_sample
                ),

                completion_time_s=(
                    completion_time_s
                ),

                reforward_sample=(
                    reforward_sample
                ),
            )

            pair_update_count += 1

        return int(
            pair_update_count
        )

    # ==========================================================
    # RoutingObservation Provider Interface
    # ==========================================================

    def get_feedback(
            self,
            source_dc_id: str,
            target_dc_id: str,
    ) -> Optional[
        Mapping[str, float]
    ]:
        """
        RoutingObservationBuilder 未来启用 Feedback 后调用。

        返回字段必须与：

            FEEDBACK_FEATURE_NAMES

        完全一致，并且全部已经归一化到 [0,1]。

        第二十九步虽然 Provider 已经安装，
        但 USE_NEIGHBOR_HISTORICAL_FEEDBACK=False，
        所以 ObservationBuilder 当前不会调用本函数。
        """

        source_dc_id = str(
            source_dc_id
        )

        target_dc_id = str(
            target_dc_id
        )

        pair_state = (
            self._pair_states.get(
                (
                    source_dc_id,
                    target_dc_id,
                )
            )
        )

        if pair_state is None:
            return None

        feedback_age_samples = max(
            int(
                self._feedback_clock
            )
            - int(
                pair_state
                .last_feedback_clock
            ),
            0,
        )

        completion_time_ewma = 0.0

        if (
                pair_state
                .completion_time_ewma_s
                is not None
        ):
            completion_time_ewma = (
                self._saturating_ratio(
                    value=(
                        pair_state
                        .completion_time_ewma_s
                    ),

                    scale=(
                        self
                        .completion_time_scale_s
                    ),
                )
            )

        feedback_age = (
            self._saturating_ratio(
                value=(
                    feedback_age_samples
                ),

                scale=(
                    self.age_scale_samples
                ),
            )
        )

        sample_confidence = (
            self._saturating_ratio(
                value=(
                    pair_state
                    .sample_count
                ),

                scale=(
                    self
                    .confidence_scale_samples
                ),
            )
        )

        feedback = {
            "success_ewma":
                self._clip01(
                    pair_state
                    .success_ewma
                ),

            "sla_success_ewma":
                self._clip01(
                    pair_state
                    .sla_success_ewma
                ),

            "drop_ewma":
                self._clip01(
                    pair_state
                    .drop_ewma
                ),

            "completion_time_ewma":
                self._clip01(
                    completion_time_ewma
                ),

            "reforward_ewma":
                self._clip01(
                    pair_state
                    .reforward_ewma
                ),

            "feedback_age":
                self._clip01(
                    feedback_age
                ),

            "sample_confidence":
                self._clip01(
                    sample_confidence
                ),
        }

        return feedback

    # ==========================================================
    # Logging / Analysis Interface
    # ==========================================================

    def get_raw_feedback(
            self,
            source_dc_id: str,
            target_dc_id: str,
    ) -> Optional[
        Dict[str, Any]
    ]:

        pair_state = (
            self._pair_states.get(
                (
                    str(source_dc_id),
                    str(target_dc_id),
                )
            )
        )

        if pair_state is None:
            return None

        return {
            "source_dc_id":
                pair_state.source_dc_id,

            "target_dc_id":
                pair_state.target_dc_id,

            "sample_count":
                int(
                    pair_state.sample_count
                ),

            "completion_sample_count":
                int(
                    pair_state
                    .completion_sample_count
                ),

            "success_ewma":
                float(
                    pair_state.success_ewma
                ),

            "sla_success_ewma":
                float(
                    pair_state
                    .sla_success_ewma
                ),

            "drop_ewma":
                float(
                    pair_state.drop_ewma
                ),

            "completion_time_ewma_s":
                (
                    None
                    if (
                        pair_state
                        .completion_time_ewma_s
                        is None
                    )
                    else float(
                        pair_state
                        .completion_time_ewma_s
                    )
                ),

            "reforward_ewma":
                float(
                    pair_state
                    .reforward_ewma
                ),

            "feedback_age_samples":
                int(
                    max(
                        self._feedback_clock
                        - pair_state
                        .last_feedback_clock,
                        0,
                    )
                ),

            "last_job_id":
                pair_state.last_job_id,

            "last_terminal_reason":
                pair_state
                .last_terminal_reason,
        }

    def snapshot_for_source(
            self,
            source_dc_id: str,
    ) -> Dict[str, Any]:
        """
        返回指定 Source DC 对所有 Neighbor 的历史认识。

        仅用于日志 / 分析。
        """

        source_dc_id = str(
            source_dc_id
        )

        snapshot: Dict[
            str,
            Any,
        ] = {}

        for target_dc_id in (
            self.edge_dc_ids
        ):

            if (
                    target_dc_id
                    == source_dc_id
            ):
                continue

            raw_feedback = (
                self.get_raw_feedback(
                    source_dc_id=(
                        source_dc_id
                    ),

                    target_dc_id=(
                        target_dc_id
                    ),
                )
            )

            encoded_feedback = (
                self.get_feedback(
                    source_dc_id=(
                        source_dc_id
                    ),

                    target_dc_id=(
                        target_dc_id
                    ),
                )
            )

            if raw_feedback is None:
                continue

            snapshot[
                target_dc_id
            ] = {
                "raw":
                    raw_feedback,

                "encoded":
                    (
                        None
                        if encoded_feedback
                        is None
                        else dict(
                            encoded_feedback
                        )
                    ),
            }

        return snapshot

    def source_summary(
            self,
            source_dc_id: str,
    ) -> Dict[str, int]:

        source_dc_id = str(
            source_dc_id
        )

        states = [
            pair_state
            for (
                source_id,
                _
            ), pair_state
            in self._pair_states.items()
            if source_id
            == source_dc_id
        ]

        return {
            "outgoing_pair_count":
                int(
                    len(states)
                ),

            "outgoing_sample_count":
                int(
                    sum(
                        pair_state
                        .sample_count
                        for pair_state
                        in states
                    )
                ),
        }

    def summary(
            self,
    ) -> Dict[str, int]:

        return {
            "terminal_jobs_seen":
                int(
                    self
                    ._total_terminal_jobs_seen
                ),

            "total_pair_samples":
                int(
                    self
                    ._total_pair_samples
                ),

            "active_pair_count":
                int(
                    len(
                        self._pair_states
                    )
                ),

            "episode_terminal_jobs_seen":
                int(
                    self
                    ._episode_terminal_jobs_seen
                ),

            "episode_pair_samples":
                int(
                    self
                    ._episode_pair_samples
                ),
        }


