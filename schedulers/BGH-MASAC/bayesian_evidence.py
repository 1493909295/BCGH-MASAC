"""终止任务到 Bayesian Evidence 的正式转换链。

第 3 步只负责把已经闭合的 ``FinalizedJobTrace`` 转换为
``BayesianHistoricalEvidence``。转换严格发生在 Job terminal 之后，
不读取远端实时资源，也不修改 ReplayBuffer 或 Observation。

一条多跳任务可以产生多条有向 Pair Evidence：

    DC1 -> DC3 -> DC5 -> Self
          |          |
          +----------+
       DC1 -> DC3   DC3 -> DC5

每条 Evidence 都绑定一个 source -> target Edge Routing 选择，
并通过 target 的下一次 Routing Decision 判定是否发生 reforward。
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from bayesian_game import (
    BayesianHistoricalEvidence,
    classify_historical_outcome,
)
from pending_job_trace import FinalizedJobTrace

__all__ = (
    "build_bayesian_evidence_from_finalized_trace",
)


def _get_terminal_outcome_facts(
        finalized_trace: FinalizedJobTrace,
        env: Any,
) -> Tuple[bool, bool, Optional[float], float]:
    """从终止任务和环境中的静态任务元数据提取历史结果。"""

    job_id = str(finalized_trace.job_id)
    job = env.job_map.get(job_id)
    if job is None:
        raise RuntimeError(
            "Bayesian Evidence 无法找到 terminal Job："
            f"{job_id}"
        )

    completed = str(finalized_trace.terminal_reason) == "completed"
    completion_time_s: Optional[float] = None
    normalized_latency_score = 0.0
    sla_satisfied = False

    if completed:
        turnaround_time = job.get_turnaround_time()
        if turnaround_time is None:
            # completed Job 的 finish_time 缺失时，使用已闭合 Trace 的
            # terminal_time 做历史结果 fallback；不访问远端实时状态。
            turnaround_time = max(
                float(finalized_trace.terminal_time)
                - float(job.arrive_time),
                0.0,
            )

        completion_time_s = max(float(turnaround_time), 0.0)
        sla_limit_s = max(
            float(env.sla_deadline_ratio) * float(job.duration),
            1e-9,
        )
        sla_satisfied = bool(
            completion_time_s <= sla_limit_s + 1e-9
        )

        # 这是“历史延迟质量分数”，越快越接近 1；不作为实时负载。
        normalized_latency_score = min(
            1.0,
            max(
                0.0,
                sla_limit_s / max(completion_time_s, 1e-9),
            ),
        )

    return (
        completed,
        sla_satisfied,
        completion_time_s,
        normalized_latency_score,
    )


def build_bayesian_evidence_from_finalized_trace(
        finalized_trace: FinalizedJobTrace,
        env: Any,
) -> Tuple[BayesianHistoricalEvidence, ...]:
    """将一个 FinalizedJobTrace 转换为有向 Pair Evidence 集合。

    转换规则：

    * 只处理 ``action_type == "edge_dc"`` 的 source -> target 选择；
    * 必须存在同一 Job 的下一 Routing Step，且 agent_id 等于 target；
    * 下一步为 ``edge_dc`` 或 ``cloud`` 时保留 reforwarded 诊断字段；
    * terminal 失败或 SLA 违约会生成终局风险证据；
    * reforwarded 由独立吸收信念处理，不重复进入终局拥塞后验；
    * 每条 Evidence 生成后立即执行信息边界校验。
    """

    (
        completed,
        sla_satisfied,
        completion_time_s,
        normalized_latency_score,
    ) = _get_terminal_outcome_facts(
        finalized_trace,
        env,
    )

    routing_steps = tuple(finalized_trace.routing_steps)
    evidences = []

    for step_index, routing_step in enumerate(routing_steps):
        if str(routing_step.action_type) != "edge_dc":
            continue

        source_dc_id = str(routing_step.source_dc_id)
        if routing_step.target_dc_id is None:
            raise RuntimeError(
                "Finalized Edge Routing 缺少 target_dc_id："
                f"job={finalized_trace.job_id}, "
                f"sequence={routing_step.sequence_index}"
            )

        target_dc_id = str(routing_step.target_dc_id)
        next_step_index = step_index + 1
        if next_step_index >= len(routing_steps):
            raise RuntimeError(
                "Bayesian Evidence 找不到 Edge Routing 的后继 Step："
                f"job={finalized_trace.job_id}, "
                f"sequence={routing_step.sequence_index}, "
                f"source={source_dc_id}, target={target_dc_id}"
            )

        next_routing_step = routing_steps[next_step_index]
        actual_target = str(next_routing_step.agent_id)
        if actual_target != target_dc_id:
            raise RuntimeError(
                "Bayesian Evidence 的 Routing 因果链 target 不一致："
                f"job={finalized_trace.job_id}, "
                f"expected={target_dc_id}, actual={actual_target}"
            )

        reforwarded = str(next_routing_step.action_type) in {
            "edge_dc",
            "cloud",
        }
        outcome_type = classify_historical_outcome(
            success=completed,
            sla_satisfied=sla_satisfied,
            normalized_latency_score=normalized_latency_score,
            reforwarded=reforwarded,
        )

        evidence = BayesianHistoricalEvidence(
            source_dc_id=source_dc_id,
            target_dc_id=target_dc_id,
            outcome_type=outcome_type,
            success=completed,
            sla_satisfied=sla_satisfied,
            reforwarded=reforwarded,
            evidence_weight=1.0,
            normalized_latency_score=normalized_latency_score,
            normalized_energy_score=0.0,
            completion_time_s=completion_time_s,
            timestamp=float(finalized_trace.terminal_time),
            routing_time=float(routing_step.env_time),
            evidence_id=f"{finalized_trace.job_id}:{routing_step.sequence_index}",
        )
        evidence.validate_information_boundary()
        evidences.append(evidence)

    return tuple(evidences)
