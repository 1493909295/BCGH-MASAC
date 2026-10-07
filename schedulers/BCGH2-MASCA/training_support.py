"""BCGH2-MASCA Bayesian runtime, state scoring and training support."""

from __future__ import annotations

"""Bayesian routing definition and terminal evidence."""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from .experience import FinalizedJobTrace


@dataclass(frozen=True)
class BayesianInformationPolicy:
    """
    Bayesian Game 的固定信息访问策略。

    注意：
        这不是可训练参数，也不应该放入 config.py。

    它用于明确保证：

        Remote Real-Time State
            X
            |
        Bayesian Belief
    """

    allow_static_routing_topology: bool = True

    allow_finalized_historical_outcomes: bool = True

    allow_neighbor_historical_feedback: bool = True

    allow_current_job_context: bool = True

    allow_source_local_information: bool = True
    allow_static_installed_capacity: bool = True
    allow_local_queue_previous_hop: bool = True

    allow_remote_realtime_cpu: bool = False

    allow_remote_realtime_gpu: bool = False

    allow_remote_realtime_queue: bool = False

    allow_remote_realtime_host_state: bool = False

    allow_remote_available_resources: bool = False

    allow_remote_raw_observation: bool = False

    allow_raw_environment_reference: bool = False

    def validate(self) -> None:
        """
        Fail-fast 验证 Bayesian Information Boundary。

        任何 Remote Real-Time 信息一旦被允许，
        都意味着当前算法已经破坏 Remote Edge
        Partial Observability 的研究假设。
        """

        forbidden_flags = {
            "allow_remote_realtime_cpu": self.allow_remote_realtime_cpu,
            "allow_remote_realtime_gpu": self.allow_remote_realtime_gpu,
            "allow_remote_realtime_queue": self.allow_remote_realtime_queue,
            "allow_remote_realtime_host_state": self.allow_remote_realtime_host_state,
            "allow_remote_available_resources": self.allow_remote_available_resources,
            "allow_remote_raw_observation": self.allow_remote_raw_observation,
            "allow_raw_environment_reference": self.allow_raw_environment_reference,
        }

        enabled_forbidden_flags = [
            flag_name for flag_name, enabled in forbidden_flags.items() if bool(enabled)
        ]

        if enabled_forbidden_flags:
            raise RuntimeError(
                "Bayesian Information Boundary 被破坏："
                "禁止 Bayesian Game 使用 Remote "
                "Real-Time State。"
                f" enabled={enabled_forbidden_flags}"
            )

    def to_metadata(
        self,
    ) -> Dict[str, bool]:
        """
        返回实验可记录的信息边界 metadata。
        """

        return {
            "allow_static_installed_capacity": self.allow_static_installed_capacity,
            "allow_local_queue_previous_hop": self.allow_local_queue_previous_hop,
            "allow_static_routing_topology": bool(self.allow_static_routing_topology),
            "allow_finalized_historical_outcomes": bool(
                self.allow_finalized_historical_outcomes
            ),
            "allow_neighbor_historical_feedback": bool(
                self.allow_neighbor_historical_feedback
            ),
            "allow_current_job_context": bool(self.allow_current_job_context),
            "allow_source_local_information": bool(self.allow_source_local_information),
            "allow_remote_realtime_cpu": bool(self.allow_remote_realtime_cpu),
            "allow_remote_realtime_gpu": bool(self.allow_remote_realtime_gpu),
            "allow_remote_realtime_queue": bool(self.allow_remote_realtime_queue),
            "allow_remote_realtime_host_state": bool(
                self.allow_remote_realtime_host_state
            ),
            "allow_remote_available_resources": bool(
                self.allow_remote_available_resources
            ),
            "allow_remote_raw_observation": bool(self.allow_remote_raw_observation),
            "allow_raw_environment_reference": bool(
                self.allow_raw_environment_reference
            ),
        }


def congestion_observed_from_outcome(
    *,
    success: bool,
    sla_satisfied: bool,
    reforwarded: bool,
) -> bool:
    """将终止任务的历史结果统一映射为拥塞 Bernoulli 样本。

    第 2 步的唯一观测语义是：

    * ``True``  ：观测到拥塞证据，更新 ``alpha_congested``；
    * ``False`` ：观测到非拥塞证据，更新 ``beta_non_congested``。

    失败或 SLA 违约视为终局风险。再转发由独立的四分类吸收信念
    建模，不能在此重复计入拥塞风险。
    """

    del reforwarded
    return bool((not success) or (not sla_satisfied))


@dataclass(frozen=True)
class BayesianHistoricalEvidence:
    """
    Bayesian Evidence。

    表示：

        source DC
            |
            |
            v
        target DC

    历史调度结果对于 Remote Service Suitability
    的一次观测证据。

    注意：

    Evidence 不是 Remote State。

    禁止包含：

        CPU
        GPU
        Queue
        Host state
        utilization
        resource availability
        raw observation

    只允许包含：

        历史动作结果
        SLA outcome
        completion outcome
        latency outcome
        energy outcome
    """

    source_dc_id: str

    target_dc_id: str

    outcome_type: str

    success: bool

    sla_satisfied: bool

    reforwarded: bool = False

    evidence_weight: float = 1.0

    normalized_latency_score: float = 0.0

    normalized_energy_score: float = 0.0

    completion_time_s: Optional[float] = None

    timestamp: Optional[float] = None
    routing_time: Optional[float] = None
    evidence_id: Optional[str] = None

    @property
    def congestion_observed(self) -> bool:
        """返回本条 Evidence 是否属于拥塞侧 Bernoulli 样本。"""

        return congestion_observed_from_outcome(
            success=self.success,
            sla_satisfied=self.sla_satisfied,
            reforwarded=self.reforwarded,
        )

    def validate_information_boundary(
        self,
    ) -> None:
        """
        防止 Evidence 演化成 Remote State。

        当前 Evidence 必须满足：

            Historical Outcome only

        """

        forbidden_fields = [
            "cpu",
            "gpu",
            "queue",
            "host_state",
            "available_resource",
            "raw_observation",
        ]

        for field_name in forbidden_fields:

            if hasattr(
                self,
                field_name,
            ):

                raise RuntimeError(
                    "Bayesian Evidence "
                    "禁止包含 Remote Real-Time State:"
                    f"{field_name}"
                )

        if not self.source_dc_id or not self.target_dc_id:
            raise ValueError(
                "Bayesian Evidence 的 source_dc_id / target_dc_id 不能为空。"
            )

        if self.source_dc_id == self.target_dc_id:
            raise ValueError(
                "Bayesian Evidence 必须表示有向 source -> target DC 对，不能是自环。"
            )

        if self.outcome_type not in {
            BayesianHiddenState.GOOD.value,
            BayesianHiddenState.NORMAL.value,
            BayesianHiddenState.RISKY.value,
        }:
            raise ValueError(
                "outcome_type 必须是 GOOD、NORMAL 或 RISKY；该标签仅用于诊断。"
            )

        if not math.isfinite(self.evidence_weight) or self.evidence_weight <= 0.0:
            raise ValueError("evidence_weight 必须大于 0。")

        for score_name, score in (
            ("normalized_latency_score", self.normalized_latency_score),
            ("normalized_energy_score", self.normalized_energy_score),
        ):
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(f"{score_name} 必须位于 [0, 1]。")


def classify_historical_outcome(
    *,
    success: bool,
    sla_satisfied: bool,
    normalized_latency_score: float,
    reforwarded: bool = False,
) -> str:
    """
    根据历史结果生成 Remote Hidden Type 的诊断标签。

    注意：

    这里不是 Bayesian posterior。

    只是：

        outcome
          |
          v
        evidence label

    注意：GOOD/NORMAL/RISKY 不是 ``alpha`` / ``beta`` 的直接值，
    也不是 posterior。拥塞 posterior 统一由
    ``congestion_observed_from_outcome`` 生成 Bernoulli 样本。
    """

    if success and sla_satisfied and normalized_latency_score >= 0.8:
        return BayesianHiddenState.GOOD.value

    if success and sla_satisfied:
        return BayesianHiddenState.NORMAL.value

    return BayesianHiddenState.RISKY.value


class BayesianHiddenState(str, Enum):
    """
    Remote Service Hidden State θ

    表示:

        source DC
            |
            |
            v

        target DC

    在长期运行过程中的服务可靠性状态。

    """

    GOOD = "good"

    NORMAL = "normal"

    RISKY = "risky"


class BayesianRoutingActionKind(str, Enum):
    """
    Routing Actor 动作在 Bayesian Game 中的语义类型。
    """

    SELF = "self"

    REMOTE_EDGE = "remote_edge"

    CLOUD = "cloud"


@dataclass(frozen=True)
class BayesianRoutingActionSemantic:
    """
    从某个 source DC 的视角描述一条 Routing action。

    只有 REMOTE_EDGE action 才对应未知 Remote Type。
    """

    action_index: int

    target_dc_id: str

    action_kind: BayesianRoutingActionKind

    has_hidden_remote_type: bool


@dataclass(frozen=True)
class BayesianStaticRoutingContext:
    """
    Bayesian Game 从 Environment 中允许获得的
    唯一静态 Routing Context。

    该对象刻意只保存 Routing identity / topology。

    因此这里明确不存在：

        CPU
        GPU
        Queue
        Host
        available resources
        utilization
        local observation
        global state

    完整 Environment 只允许在 whitelist adapter 中出现。
    Bayesian Game Core 后续只能接收本 Context。
    """

    edge_dc_ids: Tuple[str, ...]

    routing_action_target_dc_ids: Tuple[str, ...]

    cloud_enabled: bool

    cloud_id: Optional[str]


def build_bayesian_static_routing_context(
    env: Any,
) -> BayesianStaticRoutingContext:
    """
    从完整 Environment 中提取 Bayesian Game
    唯一允许使用的静态 Routing 信息。

    这是 Bayesian 子系统与完整 Environment 之间
    唯一允许存在的 direct adapter。

    白名单只允许读取：

        edge_dc_ids
        routing_action_target_dc_ids
        enable_cloud_action
        cloud_id

    明确禁止未来在本函数中加入：

        datacenters
        host_list
        CPU / GPU utilization
        waiting queue
        running queue
        available resources
        Remote observation
        Remote load

    如果以后 Bayesian 模块需要新的输入，
    必须先判断它是否违反 Remote Partial Observability。
    """

    edge_dc_ids = tuple(str(dc_id) for dc_id in env.edge_dc_ids)

    routing_action_target_dc_ids = tuple(
        str(dc_id) for dc_id in env.routing_action_target_dc_ids
    )

    cloud_enabled = bool(
        getattr(
            env,
            "enable_cloud_action",
            False,
        )
    )

    raw_cloud_id = getattr(
        env,
        "cloud_id",
        None,
    )

    cloud_id = (
        str(raw_cloud_id) if (cloud_enabled and raw_cloud_id is not None) else None
    )

    return BayesianStaticRoutingContext(
        edge_dc_ids=(edge_dc_ids),
        routing_action_target_dc_ids=(routing_action_target_dc_ids),
        cloud_enabled=(cloud_enabled),
        cloud_id=(cloud_id),
    )


@dataclass(frozen=True)
class BayesianRoutingGameDefinition:
    """
    BCGH2-MASCA Bayesian Routing Game 的静态定义。

    这是 Game Schema，不是 Bayesian State。

    因此：
        - 不维护 alpha / posterior；
        - 不读取 Neighbor Feedback；
        - 不访问 Remote realtime state；
        - 不改变 Routing Observation；
        - 不改变 MASAC policy；
        - 不改变 Reward。
    """

    player_ids: Tuple[str, ...]

    action_target_dc_ids: Tuple[str, ...]

    cloud_enabled: bool

    cloud_id: Optional[str]

    remote_type_space: Tuple[BayesianHiddenState, ...]

    information_policy: BayesianInformationPolicy

    decision_layer: str = "routing_only"

    decision_process: str = "asynchronous_repeated_routing"

    game_objective: str = "cooperative_system_scheduling"

    belief_scope: str = "directed_source_target_pair"

    evidence_scope: str = (
        "short_window_outcomes_and_local_queue_previous_hop_soft_evidence"
    )

    utility_semantics: str = "cooperative_expected_routing_prior_utility"

    equilibrium_solver: str = "none"

    externality_extension: str = "short_window_cpu_gpu_marginal_competition"

    def directed_remote_pairs(
        self,
    ) -> Tuple[Tuple[str, str], ...]:
        """
        返回未来 Bayesian Belief Store 允许维护的全部：

            source Edge -> target Edge

        有向 pair。

        Self 和 Cloud 均不建立 Remote Type Belief。
        """

        return tuple(
            (
                source_dc_id,
                target_dc_id,
            )
            for source_dc_id in self.player_ids
            for target_dc_id in self.player_ids
            if target_dc_id != source_dc_id
        )

    def action_semantics_for(
        self,
        source_dc_id: str,
    ) -> Tuple[BayesianRoutingActionSemantic, ...]:
        """
        返回一个 Routing Agent 对全部 action 的 Bayesian 语义。

        规则：

            source -> source
                Self

            source -> another Edge
                Remote Edge
                存在 Hidden Type

            source -> Cloud
                Cloud
                不建立 Remote Edge Type
        """

        source_dc_id = str(source_dc_id)

        if source_dc_id not in self.player_ids:
            raise ValueError(
                "Bayesian Routing Game 中不存在 player：" f"{source_dc_id}"
            )

        action_semantics = []

        for action_index, target_dc_id in enumerate(self.action_target_dc_ids):

            target_dc_id = str(target_dc_id)

            if target_dc_id == source_dc_id:

                action_kind = BayesianRoutingActionKind.SELF

            elif target_dc_id in self.player_ids:

                action_kind = BayesianRoutingActionKind.REMOTE_EDGE

            elif (
                self.cloud_enabled
                and self.cloud_id is not None
                and target_dc_id == self.cloud_id
            ):

                action_kind = BayesianRoutingActionKind.CLOUD

            else:

                raise RuntimeError(
                    "发现未知 Routing target："
                    f"source={source_dc_id}, "
                    f"target={target_dc_id}, "
                    f"action={action_index}"
                )

            action_semantics.append(
                BayesianRoutingActionSemantic(
                    action_index=int(action_index),
                    target_dc_id=(target_dc_id),
                    action_kind=(action_kind),
                    has_hidden_remote_type=(
                        action_kind == BayesianRoutingActionKind.REMOTE_EDGE
                    ),
                )
            )

        return tuple(action_semantics)

    def to_metadata(
        self,
    ) -> Dict[str, Any]:
        """
        返回可日志化的静态 Game Definition。

        注意：
            这里不包含任何 Bayesian Posterior。
        """

        return {
            "player_ids": list(self.player_ids),
            "action_target_dc_ids": list(self.action_target_dc_ids),
            "cloud_enabled": bool(self.cloud_enabled),
            "cloud_id": self.cloud_id,
            "remote_type_space": [
                remote_type.value for remote_type in self.remote_type_space
            ],
            "decision_layer": self.decision_layer,
            "decision_process": self.decision_process,
            "game_objective": self.game_objective,
            "belief_scope": self.belief_scope,
            "evidence_scope": self.evidence_scope,
            "utility_semantics": self.utility_semantics,
            "equilibrium_solver": self.equilibrium_solver,
            "information_policy": self.information_policy.to_metadata(),
            "externality_extension": self.externality_extension,
            "directed_remote_pair_count": len(self.directed_remote_pairs()),
        }


def build_bayesian_routing_game_definition(
    static_context: BayesianStaticRoutingContext,
) -> BayesianRoutingGameDefinition:
    """
    根据已经脱敏的 BayesianStaticRoutingContext
    构造 Bayesian Routing Game Definition。

    重要：

        本函数从 Step 5 开始不再接收完整 Environment。

    因此本函数在接口层面无法读取：

        Remote CPU
        Remote GPU
        Remote Queue
        Remote Host
        Remote available resources
        Remote raw observation

    本函数只处理：

        Edge DC identity
        Routing action target mapping
        Cloud ON/OFF
        Cloud ID

    它：

        不调用随机数；
        不维护 Bayesian posterior；
        不读取 Historical Feedback；
        不修改 Observation；
        不修改 Reward；
        不修改 Replay；
        不修改 Actor / Critic。
    """

    information_policy = BayesianInformationPolicy()

    information_policy.validate()

    player_ids = tuple(str(dc_id) for dc_id in static_context.edge_dc_ids)

    if not player_ids:
        raise RuntimeError("Bayesian Routing Game 至少需要 " "一个 Edge DC player。")

    if len(set(player_ids)) != len(player_ids):
        raise RuntimeError("Bayesian Routing Game 中存在重复 Edge DC ID。")

    action_target_dc_ids = tuple(
        str(dc_id) for dc_id in static_context.routing_action_target_dc_ids
    )

    if not action_target_dc_ids:
        raise RuntimeError("Routing action target 不能为空。")

    if len(set(action_target_dc_ids)) != len(action_target_dc_ids):
        raise RuntimeError("Routing action target 存在重复。")

    player_set = set(player_ids)

    target_set = set(action_target_dc_ids)

    missing_self_targets = player_set - target_set

    if missing_self_targets:

        raise RuntimeError(
            "Bayesian Routing Game 缺少 Self action：" f"{sorted(missing_self_targets)}"
        )

    cloud_enabled = bool(static_context.cloud_enabled)

    raw_cloud_id = getattr(
        static_context,
        "cloud_id",
        None,
    )

    cloud_id = (
        str(raw_cloud_id) if (cloud_enabled and raw_cloud_id is not None) else None
    )

    non_edge_targets = target_set - player_set

    if cloud_enabled:

        if cloud_id is None:

            raise RuntimeError("Cloud action 已开启，" "但 Environment 缺少 cloud_id。")

        if non_edge_targets != {cloud_id}:

            raise RuntimeError(
                "非 Edge Routing target "
                "必须且只能是 Cloud："
                f"actual={sorted(non_edge_targets)}, "
                f"cloud={cloud_id}"
            )

    elif non_edge_targets:

        raise RuntimeError(
            "Cloud action 已关闭，"
            "但 Routing action 中仍存在非 Edge target："
            f"{sorted(non_edge_targets)}"
        )

    definition = BayesianRoutingGameDefinition(
        player_ids=(player_ids),
        action_target_dc_ids=(action_target_dc_ids),
        cloud_enabled=(cloud_enabled),
        cloud_id=(cloud_id),
        remote_type_space=(
            BayesianHiddenState.GOOD,
            BayesianHiddenState.NORMAL,
            BayesianHiddenState.RISKY,
        ),
        information_policy=(information_policy),
    )

    for player_id in definition.player_ids:

        definition.action_semantics_for(player_id)

    return definition


def _get_terminal_outcome_facts(
    finalized_trace: FinalizedJobTrace,
    env: Any,
) -> Tuple[bool, bool, Optional[float], float]:
    """从终止任务和环境中的静态任务元数据提取历史结果。"""

    job_id = str(finalized_trace.job_id)
    job = env.job_map.get(job_id)
    if job is None:
        raise RuntimeError("Bayesian Evidence 无法找到 terminal Job：" f"{job_id}")

    completed = str(finalized_trace.terminal_reason) == "completed"
    completion_time_s: Optional[float] = None
    normalized_latency_score = 0.0
    sla_satisfied = False

    if completed:
        turnaround_time = job.get_turnaround_time()
        if turnaround_time is None:
            turnaround_time = max(
                float(finalized_trace.terminal_time) - float(job.arrive_time),
                0.0,
            )

        completion_time_s = max(float(turnaround_time), 0.0)
        sla_limit_s = max(
            float(env.sla_deadline_ratio) * float(job.duration),
            1e-9,
        )
        sla_satisfied = bool(completion_time_s <= sla_limit_s + 1e-9)

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


"""Congestion beliefs and short-window runtime."""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping


def positive(value, name, *, zero=False):
    value = float(value)
    if not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(
            f"{name} must be finite and {'nonnegative' if zero else 'positive'}"
        )
    return value


@dataclass(frozen=True)
class BayesianCongestionPairKey:
    source_dc_id: str
    target_dc_id: str


@dataclass(frozen=True)
class StaticDCCapacity:
    cpu: float
    gpu: float
    hosts: tuple[tuple[float, float], ...]

    def __post_init__(self):
        positive(self.cpu, "cpu capacity", zero=True)
        positive(self.gpu, "gpu capacity", zero=True)
        for cpu, gpu in self.hosts:
            positive(cpu, "host cpu", zero=True)
            positive(gpu, "host gpu", zero=True)

    def fits(self, cpu, gpu):
        return any(cpu <= c + 1e-9 and gpu <= g + 1e-9 for c, g in self.hosts)


@dataclass(frozen=True)
class LocalSourceSignal:
    waiting_volume: float = 0.0
    running_volume: float = 0.0
    effective_events: float = 0.0


@dataclass(frozen=True)
class RoutingActionFeatures:
    target_dc_id: str
    success_score: float = 0.5
    sla_score: float = 0.5
    delay_score: float = 0.5
    cpu_request: float = 0.0
    gpu_request: float = 0.0
    source_signal: LocalSourceSignal = LocalSourceSignal()


@dataclass(frozen=True)
class ActionUtilityBreakdown:
    target_dc_id: str
    benefit: float
    congestion_probability: float
    congestion_cost: float
    risk: float
    utility: float
    confidence: float
    bias: float
    outcome_probability: float = 0.5
    source_signal: float = 0.0
    source_pseudo_count: float = 0.0
    source_confidence: float = 0.0
    outcome_confidence: float = 0.0
    resource_cost_raw: float = 0.0
    feasible: bool = True
    guidance_utility: float = 0.0
    outcome_samples: float = 0.0
    cpu_pressure: float = 0.0
    gpu_pressure: float = 0.0
    cpu_request: float = 0.0
    gpu_request: float = 0.0
    absorption_probability: float = 0.0
    edge_reforward_probability: float = 0.0
    cloud_escape_probability: float = 0.0
    absorption_drop_probability: float = 0.0
    absorption_confidence: float = 0.0
    absorption_pair_samples: float = 0.0
    absorption_class_samples: float = 0.0
    absorption_class_mix: float = 0.0
    absorption_resource_class: str = ""
    transit_penalty_raw: float = 0.0
    transit_penalty: float = 0.0


@dataclass
class BetaBeliefState:
    alpha_congested: float = 1.0
    beta_non_congested: float = 1.0
    sample_weight: float = 0.0
    last_update_clock: float | None = None

    @property
    def congestion_probability(self):
        return self.alpha_congested / (self.alpha_congested + self.beta_non_congested)

    def confidence(self, confidence_scale):
        return self.sample_weight / (
            self.sample_weight + positive(confidence_scale, "confidence_scale")
        )


ABSORPTION_OUTCOMES = ("self", "edge", "cloud", "drop")


@dataclass(frozen=True)
class AbsorptionEvidence:
    """Observed successor of one executed source->target edge transfer."""

    event_id: str
    job_id: str
    source_dc_id: str
    target_dc_id: str
    routed_at: float
    observed_at: float
    outcome: str
    cpu_request: float
    gpu_request: float
    resource_class: str
    predecessor_action_source: str
    successor_action_source: str
    informative: bool = True

    def __post_init__(self):
        if not self.event_id or not self.job_id:
            raise ValueError("Absorption evidence requires event_id and job_id")
        if (
            not self.source_dc_id
            or not self.target_dc_id
            or self.source_dc_id == self.target_dc_id
        ):
            raise ValueError("Absorption evidence requires a directed non-self DC pair")
        if self.outcome not in ABSORPTION_OUTCOMES:
            raise ValueError(f"Unknown absorption outcome: {self.outcome}")
        positive(self.routed_at, "routed_at", zero=True)
        positive(self.observed_at, "observed_at", zero=True)
        if self.observed_at + 1e-9 < self.routed_at:
            raise ValueError("Absorption outcome precedes its edge transfer")
        positive(self.cpu_request, "cpu request", zero=True)
        positive(self.gpu_request, "gpu request", zero=True)


@dataclass(frozen=True)
class AbsorptionBeliefState:
    p_absorb: float
    p_edge: float
    p_cloud: float
    p_drop: float
    confidence: float
    pair_samples: float
    class_samples: float
    class_mix: float
    resource_class: str
    counts: Mapping[str, float]


class ShortWindowAbsorptionStore:
    """Dirichlet successor-action beliefs for each directed Edge pair."""

    def __init__(
        self,
        edge_dc_ids,
        capacities,
        *,
        window_s=1200.0,
        priors=None,
        confidence_scale=8.0,
        demand_thresholds=(0.25, 0.60),
        class_confidence_scale=4.0,
    ):
        self.edge_dc_ids = tuple(map(str, edge_dc_ids))
        self.capacities = dict(capacities)
        self.window_s = positive(window_s, "absorption window")
        priors = priors or {name: 1.0 for name in ABSORPTION_OUTCOMES}
        if set(priors) != set(ABSORPTION_OUTCOMES):
            raise ValueError("Absorption priors must cover self/edge/cloud/drop")
        self.priors = {
            name: positive(priors[name], f"{name} prior")
            for name in ABSORPTION_OUTCOMES
        }
        self.confidence_scale = positive(
            confidence_scale, "absorption confidence scale"
        )
        thresholds = tuple(float(x) for x in demand_thresholds)
        if len(thresholds) != 2 or not 0 < thresholds[0] < thresholds[1]:
            raise ValueError("Demand thresholds must be two increasing positive values")
        self.demand_thresholds = thresholds
        self.class_confidence_scale = positive(
            class_confidence_scale, "absorption class confidence scale"
        )
        self.reset()

    def reset(self):
        self.now = 0.0
        self._records = {}
        self._seen_event_ids = set()
        self._episode_counts = {name: 0 for name in ABSORPTION_OUTCOMES}
        self._episode_informative_counts = {name: 0 for name in ABSORPTION_OUTCOMES}
        self._episode_pair_counts = {}

    def advance_time(self, now):
        now = positive(now, "simulation time", zero=True)
        if now < self.now - 1e-9:
            raise ValueError("Simulation clock moved backwards")
        self.now = now
        self._records = {
            key: record
            for key, record in self._records.items()
            if record.observed_at > now - self.window_s
        }

    def resource_class(self, target_dc_id, cpu_request, gpu_request):
        capacity = self.capacities[str(target_dc_id)]
        max_cpu = max((cpu for cpu, _ in capacity.hosts), default=0.0)
        max_gpu = max((gpu for _, gpu in capacity.hosts), default=0.0)
        cpu_ratio = float(cpu_request) / max_cpu if max_cpu > 0 else 0.0
        gpu_ratio = float(gpu_request) / max_gpu if max_gpu > 0 else 0.0
        dominant = max(cpu_ratio, gpu_ratio)
        if dominant < self.demand_thresholds[0]:
            size = "low"
        elif dominant < self.demand_thresholds[1]:
            size = "medium"
        else:
            size = "high"
        return f"{'gpu' if float(gpu_request) > 0 else 'cpu'}:{size}"

    def record(self, evidence):
        if not isinstance(evidence, AbsorptionEvidence):
            raise TypeError("Expected AbsorptionEvidence")
        if (
            evidence.source_dc_id not in self.capacities
            or evidence.target_dc_id not in self.capacities
        ):
            raise KeyError(
                f"Unknown absorption pair: {evidence.source_dc_id}->{evidence.target_dc_id}"
            )
        if evidence.event_id in self._seen_event_ids:
            return False
        self.advance_time(max(self.now, evidence.observed_at))
        self._seen_event_ids.add(evidence.event_id)
        if evidence.observed_at > self.now - self.window_s:
            self._records[evidence.event_id] = evidence
        self._episode_counts[evidence.outcome] += 1
        if evidence.informative:
            self._episode_informative_counts[evidence.outcome] += 1
        pair = f"{evidence.source_dc_id}->{evidence.target_dc_id}"
        pair_counts = self._episode_pair_counts.setdefault(
            pair, {name: 0 for name in ABSORPTION_OUTCOMES}
        )
        pair_counts[evidence.outcome] += 1
        return True

    def records(self, source_dc_id, target_dc_id, *, informative_only=True):
        source_dc_id, target_dc_id = str(source_dc_id), str(target_dc_id)
        return tuple(
            r
            for r in self._records.values()
            if r.source_dc_id == source_dc_id
            and r.target_dc_id == target_dc_id
            and (r.informative or not informative_only)
        )

    def belief(self, source_dc_id, target_dc_id, cpu_request=0.0, gpu_request=0.0):
        target_dc_id = str(target_dc_id)
        resource_class = self.resource_class(target_dc_id, cpu_request, gpu_request)
        pair_records = self.records(source_dc_id, target_dc_id)
        class_records = tuple(
            r for r in pair_records if r.resource_class == resource_class
        )

        def posterior(records):
            counts = {
                name: sum(r.outcome == name for r in records)
                for name in ABSORPTION_OUTCOMES
            }
            total = sum(counts.values())
            denominator = sum(self.priors.values()) + total
            probabilities = {
                name: (self.priors[name] + counts[name]) / denominator
                for name in ABSORPTION_OUTCOMES
            }
            return counts, float(total), probabilities

        pair_counts, pair_n, pair_p = posterior(pair_records)
        _, class_n, class_p = posterior(class_records)
        class_mix = class_n / (class_n + self.class_confidence_scale)
        probabilities = {
            name: class_mix * class_p[name] + (1.0 - class_mix) * pair_p[name]
            for name in ABSORPTION_OUTCOMES
        }
        confidence = pair_n / (pair_n + self.confidence_scale)
        return AbsorptionBeliefState(
            p_absorb=probabilities["self"],
            p_edge=probabilities["edge"],
            p_cloud=probabilities["cloud"],
            p_drop=probabilities["drop"],
            confidence=confidence,
            pair_samples=pair_n,
            class_samples=class_n,
            class_mix=class_mix,
            resource_class=resource_class,
            counts=pair_counts,
        )

    def snapshot(self):
        result = {}
        pairs = {(r.source_dc_id, r.target_dc_id) for r in self._records.values()}
        for source, target in sorted(pairs):
            records = self.records(source, target)
            result[f"{source}->{target}"] = {
                name: sum(r.outcome == name for r in records)
                for name in ABSORPTION_OUTCOMES
            }
        return result

    def episode_summary(self):
        total = sum(self._episode_counts.values())
        informative = sum(self._episode_informative_counts.values())
        return {
            "total": total,
            "informative_total": informative,
            "counts": dict(self._episode_counts),
            "informative_counts": dict(self._episode_informative_counts),
            "pair_counts": {
                key: dict(value) for key, value in self._episode_pair_counts.items()
            },
            "edge_to_cloud_transit_rate": (
                self._episode_counts["cloud"] / total if total else 0.0
            ),
            "edge_to_edge_reforward_rate": (
                self._episode_counts["edge"] / total if total else 0.0
            ),
            "edge_target_absorption_rate": (
                self._episode_counts["self"] / total if total else 0.0
            ),
        }


class BayesianBeliefStore:
    def __init__(
        self,
        edge_dc_ids,
        *,
        prior_alpha=1.0,
        prior_beta=1.0,
        confidence_scale=20.0,
        window_s=1200.0,
    ):
        self.edge_dc_ids = tuple(map(str, edge_dc_ids))
        self.prior_alpha = positive(prior_alpha, "prior_alpha")
        self.prior_beta = positive(prior_beta, "prior_beta")
        self.confidence_scale = positive(confidence_scale, "confidence_scale")
        self.window_s = positive(window_s, "window_s")
        self.reset()

    def reset(self):
        self.now = 0.0
        self._records = {
            (i, j): {} for i in self.edge_dc_ids for j in self.edge_dc_ids if i != j
        }

    def advance_time(self, now):
        now = positive(now, "simulation time", zero=True)
        if now < self.now - 1e-9:
            raise ValueError(
                "Simulation clock moved backwards; reset the episode first"
            )
        self.now = now
        for records in self._records.values():
            for key in tuple(records):
                if records[key].routing_time <= now - self.window_s:
                    del records[key]

    def records(self, source_dc_id, target_dc_id):
        return tuple(self._records[(str(source_dc_id), str(target_dc_id))].values())

    def update(self, evidence: BayesianHistoricalEvidence, *, update_clock=None):
        evidence.validate_information_boundary()
        if (
            evidence.timestamp is None
            or evidence.routing_time is None
            or not evidence.evidence_id
        ):
            raise ValueError(
                "Window evidence requires terminal time, routing time and a unique ID"
            )
        terminal = positive(evidence.timestamp, "terminal time", zero=True)
        routed = positive(evidence.routing_time, "routing time", zero=True)
        if routed > terminal:
            raise ValueError("Outcome precedes routing")
        self.advance_time(
            max(self.now, terminal) if update_clock is None else update_clock
        )
        if terminal > self.now + 1e-9:
            raise ValueError("Future outcome cannot be used")
        records = self._records[(evidence.source_dc_id, evidence.target_dc_id)]
        if routed > self.now - self.window_s:
            records.setdefault(evidence.evidence_id, evidence)
        return self.get_state(evidence.source_dc_id, evidence.target_dc_id)

    def get_state(self, source_dc_id, target_dc_id):
        records = self.records(source_dc_id, target_dc_id)
        bad = sum(e.evidence_weight for e in records if e.congestion_observed)
        good = sum(e.evidence_weight for e in records if not e.congestion_observed)
        return BetaBeliefState(
            self.prior_alpha + bad,
            self.prior_beta + good,
            bad + good,
            max((e.timestamp for e in records), default=None),
        )

    def get_congestion_probability(self, source_dc_id, target_dc_id):
        return self.get_state(source_dc_id, target_dc_id).congestion_probability

    def get_confidence(self, source_dc_id, target_dc_id):
        return self.get_state(source_dc_id, target_dc_id).confidence(
            self.confidence_scale
        )

    def snapshot(self):
        result = {}
        for i, j in self._records:
            state = self.get_state(i, j)
            result[f"{i}->{j}"] = dict(
                alpha_congested=state.alpha_congested,
                beta_non_congested=state.beta_non_congested,
                sample_weight=state.sample_weight,
                congestion_probability=state.congestion_probability,
                confidence=state.confidence(self.confidence_scale),
            )
        return result


class HistoricalPressureStore:
    """Executed edge transfers within (now-T, now], weighted by CPU/GPU demand."""

    def __init__(
        self,
        capacities: Mapping[str, StaticDCCapacity],
        *,
        window_s=1200.0,
        linear_cost_weight=1.0,
        quadratic_cost_weight=1.0,
        cpu_weight=0.5,
        gpu_weight=0.5,
    ):
        self.capacities = dict(capacities)
        self.edge_dc_ids = tuple(self.capacities)
        self.window_s = positive(window_s, "window_s")
        self.linear_cost_weight = positive(linear_cost_weight, "linear cost", zero=True)
        self.quadratic_cost_weight = positive(
            quadratic_cost_weight, "quadratic cost", zero=True
        )
        self.weights = (
            positive(cpu_weight, "cpu weight", zero=True),
            positive(gpu_weight, "gpu weight", zero=True),
        )
        positive(sum(self.weights), "resource weight sum")
        self.reset()

    def reset(self):
        self.now = 0.0
        self._records = {}

    def advance_time(self, now):
        now = positive(now, "simulation time", zero=True)
        if now < self.now - 1e-9:
            raise ValueError("Simulation clock moved backwards")
        self.now = now
        self._records = {
            k: v for k, v in self._records.items() if v[0] > now - self.window_s
        }

    def record_selection(
        self, target_dc_id, *, event_id, event_time, cpu_request, gpu_request
    ):
        if target_dc_id not in self.capacities:
            raise KeyError(target_dc_id)
        cpu = positive(cpu_request, "cpu request", zero=True)
        gpu = positive(gpu_request, "gpu request", zero=True)
        event_time = positive(event_time, "event time", zero=True)
        if event_time > self.now + 1e-9:
            raise ValueError("Transfer has not occurred")
        if event_time > self.now - self.window_s:
            self._records.setdefault(event_id, (event_time, target_dc_id, cpu, gpu))

    def get_pressure(self, target_dc_id):
        capacity = self.capacities[target_dc_id]
        selected = [r for r in self._records.values() if r[1] == target_dc_id]
        return tuple(
            sum(r[index + 2] for r in selected) / cap if cap > 0 else 0.0
            for index, cap in enumerate((capacity.cpu, capacity.gpu))
        )

    def get_congestion_cost(self, target_dc_id, cpu_request=0.0, gpu_request=0.0):
        cpu_request = positive(cpu_request, "cpu request", zero=True)
        gpu_request = positive(gpu_request, "gpu request", zero=True)
        capacity = self.capacities[target_dc_id]
        if not capacity.fits(cpu_request, gpu_request):
            return math.inf
        cost = 0.0
        for x, demand, cap, weight in zip(
            self.get_pressure(target_dc_id),
            (cpu_request, gpu_request),
            (capacity.cpu, capacity.gpu),
            self.weights,
        ):
            delta = demand / cap if cap > 0 else 0.0
            cost += weight * (
                self.linear_cost_weight * delta
                + self.quadratic_cost_weight * (2 * x * delta + delta * delta)
            )
        return cost

    def snapshot(self):
        return {
            j: dict(
                cpu_pressure=self.get_pressure(j)[0],
                gpu_pressure=self.get_pressure(j)[1],
                recent_selection_count=sum(r[1] == j for r in self._records.values()),
            )
            for j in self.edge_dc_ids
        }


class BayesianCongestionGame:
    def __init__(
        self,
        game_definition,
        *,
        capacities,
        prior_alpha=1.0,
        prior_beta=1.0,
        confidence_scale=20.0,
        window_s=1200.0,
        linear_cost_weight=1.0,
        quadratic_cost_weight=1.0,
        benefit_success_weight=0.4,
        benefit_sla_weight=0.4,
        benefit_delay_weight=0.2,
        risk_congestion_cost_weight=1.0,
        source_evidence_weight=2.0,
        source_volume_scale=1.0,
        source_confidence_scale=5.0,
        cpu_weight=0.5,
        gpu_weight=0.5,
        resource_cost_scale=1.0,
        absorption_enabled=True,
        absorption_priors=None,
        absorption_confidence_scale=8.0,
        absorption_edge_weight=0.7,
        absorption_cloud_weight=0.5,
        absorption_drop_weight=1.0,
        absorption_demand_thresholds=(0.25, 0.60),
        absorption_class_confidence_scale=4.0,
        absorption_policy_actions_only=True,
    ):
        self.game_definition = game_definition
        if set(capacities) != set(game_definition.player_ids):
            raise ValueError("Static capacities must cover exactly the edge players")
        self.belief_store = BayesianBeliefStore(
            game_definition.player_ids,
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
            confidence_scale=confidence_scale,
            window_s=window_s,
        )
        self.pressure_store = HistoricalPressureStore(
            capacities,
            window_s=window_s,
            linear_cost_weight=linear_cost_weight,
            quadratic_cost_weight=quadratic_cost_weight,
            cpu_weight=cpu_weight,
            gpu_weight=gpu_weight,
        )
        self.absorption_enabled = bool(absorption_enabled)
        self.absorption_policy_actions_only = bool(absorption_policy_actions_only)
        self.absorption_store = ShortWindowAbsorptionStore(
            game_definition.player_ids,
            capacities,
            window_s=window_s,
            priors=absorption_priors,
            confidence_scale=absorption_confidence_scale,
            demand_thresholds=absorption_demand_thresholds,
            class_confidence_scale=absorption_class_confidence_scale,
        )
        self.absorption_weights = {
            "edge": positive(
                absorption_edge_weight, "absorption edge weight", zero=True
            ),
            "cloud": positive(
                absorption_cloud_weight, "absorption cloud weight", zero=True
            ),
            "drop": positive(
                absorption_drop_weight, "absorption drop weight", zero=True
            ),
        }
        self.benefit_weights = tuple(
            positive(w, "benefit weight", zero=True)
            for w in (benefit_success_weight, benefit_sla_weight, benefit_delay_weight)
        )
        positive(sum(self.benefit_weights), "benefit weight sum")
        self.risk_congestion_cost_weight = positive(
            risk_congestion_cost_weight, "risk weight", zero=True
        )
        self.source_evidence_weight = positive(
            source_evidence_weight, "source weight", zero=True
        )
        self.source_volume_scale = positive(source_volume_scale, "source volume scale")
        self.source_confidence_scale = positive(
            source_confidence_scale, "source confidence scale"
        )
        self.resource_cost_scale = positive(resource_cost_scale, "resource cost scale")

    def reset(self):
        self.belief_store.reset()
        self.pressure_store.reset()
        self.absorption_store.reset()

    def advance_time(self, now):
        self.belief_store.advance_time(now)
        self.pressure_store.advance_time(now)
        self.absorption_store.advance_time(now)

    def update_belief(self, evidence, *, update_clock=None):
        return self.belief_store.update(evidence, update_clock=update_clock)

    def get_congestion_probability(self, source_dc_id, target_dc_id):
        return self.belief_store.get_congestion_probability(source_dc_id, target_dc_id)

    def update_pressure(self, source_dc_id, target_dc_id, **kwargs):
        self.belief_store.get_state(source_dc_id, target_dc_id)
        return self.pressure_store.record_selection(target_dc_id, **kwargs)

    def record_absorption(self, evidence):
        return self.absorption_store.record(evidence)

    def get_absorption_belief(
        self, source_dc_id, target_dc_id, cpu_request=0.0, gpu_request=0.0
    ):
        return self.absorption_store.belief(
            source_dc_id, target_dc_id, cpu_request, gpu_request
        )

    def get_pressure(self, target_dc_id):
        return self.pressure_store.get_pressure(target_dc_id)

    def get_congestion_cost(self, target_dc_id, cpu_request=0.0, gpu_request=0.0):
        return self.pressure_store.get_congestion_cost(
            target_dc_id, cpu_request, gpu_request
        )

    def calculate_benefit(self, action):
        values = (action.success_score, action.sla_score, action.delay_score)
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError("Quality scores must lie in [0,1]")
        return sum(w * v for w, v in zip(self.benefit_weights, values))

    def evaluate_actions(self, source_dc_id, job_context):
        if not job_context:
            raise ValueError("No candidate actions")
        rows = []
        for target, raw in job_context.items():
            action = (
                raw
                if isinstance(raw, RoutingActionFeatures)
                else RoutingActionFeatures(**raw)
            )
            if target != action.target_dc_id:
                raise ValueError("Candidate target mismatch")
            benefit = self.calculate_benefit(action)
            remote = target in self.pressure_store.capacities and target != source_dc_id
            row = dict(
                target_dc_id=target,
                benefit=benefit,
                congestion_probability=0.0,
                congestion_cost=0.0,
                risk=0.0,
                utility=benefit,
                confidence=1.0,
                bias=0.0,
                guidance_utility=benefit,
                cpu_request=action.cpu_request,
                gpu_request=action.gpu_request,
            )
            if remote:
                state = self.belief_store.get_state(source_dc_id, target)
                local = action.source_signal
                volume = positive(
                    local.waiting_volume, "waiting volume", zero=True
                ) + positive(local.running_volume, "running volume", zero=True)
                events = positive(local.effective_events, "source events", zero=True)
                signal = volume / (volume + self.source_volume_scale)
                pseudo = self.source_evidence_weight * signal
                source_conf = (
                    events / (events + self.source_confidence_scale)
                    if pseudo > 0
                    else 0.0
                )
                out_conf = state.confidence(self.belief_store.confidence_scale)
                fused = (state.alpha_congested + pseudo) / (
                    state.alpha_congested + state.beta_non_congested + pseudo
                )
                cost = self.get_congestion_cost(
                    target, action.cpu_request, action.gpu_request
                )
                feasible = math.isfinite(cost)
                normalized_cost = (
                    cost / (cost + self.resource_cost_scale) if feasible else 1.0
                )
                confidence = 1 - (1 - out_conf) * (1 - source_conf)
                effective_probability = (
                    0.5
                    + out_conf * (state.congestion_probability - 0.5)
                    + source_conf * (fused - state.congestion_probability)
                )
                absorption = self.get_absorption_belief(
                    source_dc_id, target, action.cpu_request, action.gpu_request
                )
                transit_raw = (
                    self.absorption_weights["edge"] * absorption.p_edge
                    + self.absorption_weights["cloud"] * absorption.p_cloud
                    + self.absorption_weights["drop"] * absorption.p_drop
                )
                transit_penalty = (
                    absorption.confidence * transit_raw
                    if self.absorption_enabled
                    else 0.0
                )
                risk = (
                    fused
                    + self.risk_congestion_cost_weight * normalized_cost
                    + transit_penalty
                )
                utility = (
                    0.5 + out_conf * (benefit - 0.5) - (effective_probability - 0.5)
                )
                utility -= self.risk_congestion_cost_weight * normalized_cost
                utility -= transit_penalty
                row.update(
                    congestion_probability=fused,
                    congestion_cost=normalized_cost,
                    risk=risk,
                    utility=benefit - risk,
                    guidance_utility=utility,
                    confidence=confidence,
                    outcome_probability=state.congestion_probability,
                    source_signal=signal,
                    source_pseudo_count=pseudo,
                    source_confidence=source_conf,
                    outcome_confidence=out_conf,
                    resource_cost_raw=cost if feasible else 0.0,
                    feasible=feasible,
                    outcome_samples=state.sample_weight,
                    cpu_pressure=self.get_pressure(target)[0],
                    gpu_pressure=self.get_pressure(target)[1],
                    absorption_probability=absorption.p_absorb,
                    edge_reforward_probability=absorption.p_edge,
                    cloud_escape_probability=absorption.p_cloud,
                    absorption_drop_probability=absorption.p_drop,
                    absorption_confidence=absorption.confidence,
                    absorption_pair_samples=absorption.pair_samples,
                    absorption_class_samples=absorption.class_samples,
                    absorption_class_mix=absorption.class_mix,
                    absorption_resource_class=absorption.resource_class,
                    transit_penalty_raw=transit_raw,
                    transit_penalty=transit_penalty,
                )
            rows.append(row)
        return tuple(ActionUtilityBreakdown(**r) for r in rows)


def build_short_window_game(env, settings, definition=None):
    if definition is None:
        definition = build_bayesian_routing_game_definition(
            build_bayesian_static_routing_context(env)
        )
    return BayesianCongestionGame(
        definition,
        capacities=build_static_capacities(env),
        prior_alpha=settings.bayesian_prior_alpha,
        prior_beta=settings.bayesian_prior_beta,
        confidence_scale=settings.bayesian_confidence_scale,
        window_s=settings.short_window_s,
        source_evidence_weight=settings.source_evidence_weight,
        source_volume_scale=settings.source_volume_scale,
        source_confidence_scale=settings.source_confidence_scale,
        cpu_weight=settings.resource_cpu_weight,
        gpu_weight=settings.resource_gpu_weight,
        resource_cost_scale=settings.resource_cost_scale,
        linear_cost_weight=settings.pressure_linear_cost_weight,
        quadratic_cost_weight=settings.pressure_quadratic_cost_weight,
        benefit_success_weight=settings.benefit_success_weight,
        benefit_sla_weight=settings.benefit_sla_weight,
        benefit_delay_weight=settings.benefit_delay_weight,
        risk_congestion_cost_weight=settings.risk_congestion_cost_weight,
        absorption_enabled=settings.absorption_enabled,
        absorption_priors={
            "self": settings.absorption_prior_self,
            "edge": settings.absorption_prior_edge,
            "cloud": settings.absorption_prior_cloud,
            "drop": settings.absorption_prior_drop,
        },
        absorption_confidence_scale=settings.absorption_confidence_scale,
        absorption_edge_weight=settings.absorption_edge_weight,
        absorption_cloud_weight=settings.absorption_cloud_weight,
        absorption_drop_weight=settings.absorption_drop_weight,
        absorption_demand_thresholds=settings.absorption_demand_thresholds,
        absorption_class_confidence_scale=settings.absorption_class_confidence_scale,
        absorption_policy_actions_only=settings.absorption_policy_actions_only,
    )


def build_static_capacities(env):
    """One-time whitelist: installed CPU/GPU only, never used/available resources."""
    result = {}
    base_dc_map = {str(dc.dc_id): dc for dc in env.base_datacenters}
    for dc_id in env.edge_dc_ids:
        hosts = tuple(
            (float(h.cpu_num), float(h.gpu_capacity_num))
            for h in base_dc_map[dc_id].host_list
        )
        result[str(dc_id)] = StaticDCCapacity(
            sum(c for c, _ in hosts), sum(g for _, g in hosts), hosts
        )
    return result


@dataclass(frozen=True)
class LocalReceipt:
    previous_dc_id: str
    received_at: float
    event_id: str
    informative: bool


@dataclass(frozen=True)
class PendingAbsorption:
    job_id: str
    source_dc_id: str
    target_dc_id: str
    event_id: str
    routed_at: float
    cpu_request: float
    gpu_request: float
    predecessor_action_source: str


class ShortWindowRuntime:
    def __init__(self, game):
        self.game = game
        self.policy_actions_only = bool(game.absorption_policy_actions_only)
        self.reset()

    def reset(self):
        self.game.reset()
        self._pending_arrivals = {}
        self._pending_absorption = {}
        self._pending_self = {}
        self._receipts = {i: {} for i in self.game.game_definition.player_ids}
        self._sequence = 0

    def record_routing(self, decision, action_type, target_dc_id, action_source, job):
        """Called after env.step succeeded, before building the next observation."""
        action_type = str(action_type)
        action_source = str(action_source)
        job_id = str(decision.job_id)
        self.game.advance_time(decision.env_time)

        predecessor = self._pending_absorption.pop(job_id, None)
        if predecessor is not None:
            if predecessor.target_dc_id != str(decision.agent_id):
                raise RuntimeError("Absorption successor DC does not match edge target")
            if action_type == "self":
                self._pending_self[job_id] = (
                    predecessor,
                    float(decision.env_time),
                    action_source,
                )
            else:
                outcome = {"edge_dc": "edge", "cloud": "cloud", "drop": "drop"}.get(
                    action_type
                )
                if outcome is None:
                    raise RuntimeError(
                        f"Unknown absorption successor action: {action_type}"
                    )
                self._record_absorption(
                    predecessor,
                    outcome,
                    observed_at=decision.env_time,
                    successor_action_source=action_source,
                )

        if action_type != "edge_dc":
            return
        self._sequence += 1
        event_id = f"{job_id}:{self._sequence}"
        self.game.update_pressure(
            decision.agent_id,
            target_dc_id,
            event_id=event_id,
            event_time=decision.env_time,
            cpu_request=job.cpu_request,
            gpu_request=job.gpu_request,
        )
        pending = PendingAbsorption(
            job_id=job_id,
            source_dc_id=str(decision.agent_id),
            target_dc_id=str(target_dc_id),
            event_id=event_id,
            routed_at=float(decision.env_time),
            cpu_request=float(job.cpu_request),
            gpu_request=float(job.gpu_request),
            predecessor_action_source=action_source,
        )
        self._pending_arrivals[job_id] = pending
        self._pending_absorption[job_id] = pending

    def _record_absorption(
        self, pending, outcome, *, observed_at, successor_action_source
    ):
        informative = not self.policy_actions_only or (
            pending.predecessor_action_source == "policy"
            and str(successor_action_source) == "policy"
        )
        resource_class = self.game.absorption_store.resource_class(
            pending.target_dc_id, pending.cpu_request, pending.gpu_request
        )
        return self.game.record_absorption(
            AbsorptionEvidence(
                event_id=pending.event_id,
                job_id=pending.job_id,
                source_dc_id=pending.source_dc_id,
                target_dc_id=pending.target_dc_id,
                routed_at=pending.routed_at,
                observed_at=float(observed_at),
                outcome=str(outcome),
                cpu_request=pending.cpu_request,
                gpu_request=pending.gpu_request,
                resource_class=resource_class,
                predecessor_action_source=pending.predecessor_action_source,
                successor_action_source=str(successor_action_source),
                informative=informative,
            )
        )

    def record_host_result(self, job_id, dc_id, execution_result, now):
        """Finalize a provisional SELF successor after the real host admission result."""
        self.game.advance_time(now)
        item = self._pending_self.pop(str(job_id), None)
        if item is None:
            return False
        pending, observed_at, successor_action_source = item
        if pending.target_dc_id != str(dc_id):
            raise RuntimeError("Host result DC does not match absorption target")
        execution_result = str(execution_result)
        if execution_result not in {"started", "queued", "dropped"}:
            raise ValueError(f"Unknown host execution result: {execution_result}")
        outcome = "self" if execution_result in {"started", "queued"} else "drop"
        return self._record_absorption(
            pending,
            outcome,
            observed_at=max(float(observed_at), float(now)),
            successor_action_source=successor_action_source,
        )

    def record_arrival(self, dc_id, job_id, now):
        """Only a real current routing decision makes the receipt visible locally."""
        self.game.advance_time(now)
        pending = self._pending_arrivals.get(str(job_id))
        if pending is not None:
            if pending.target_dc_id != dc_id:
                raise RuntimeError("Incoming task target does not match current DC")
            self._receipts[dc_id][str(job_id)] = LocalReceipt(
                pending.source_dc_id,
                now,
                pending.event_id,
                pending.predecessor_action_source == "policy",
            )
            del self._pending_arrivals[str(job_id)]
        cutoff = now - self.game.belief_store.window_s
        for receipts in self._receipts.values():
            for key in tuple(receipts):
                if receipts[key].received_at <= cutoff:
                    del receipts[key]

    def local_source_signals(self, source_dc_id, local_dc, now):
        """Caller supplies exactly one local DC; no access to an environment here."""
        self.game.advance_time(now)
        capacity = self.game.pressure_store.capacities[source_dc_id]
        volumes = {}
        seen = set()
        for host in local_dc.host_list:
            for queue_name, index in (("waiting_queue", 0), ("running_queue", 1)):
                for job in getattr(host, queue_name)._queue:
                    job_id = str(job.job_id)
                    if job_id in seen:
                        continue
                    seen.add(job_id)
                    receipt = self._receipts[source_dc_id].get(job_id)
                    if receipt is None or not receipt.informative:
                        continue
                    age = now - receipt.received_at
                    if not 0 <= age < self.game.belief_store.window_s:
                        continue
                    freshness = 1.0 - age / self.game.belief_store.window_s
                    volume = max(
                        float(job.cpu_request) / capacity.cpu if capacity.cpu else 0.0,
                        float(job.gpu_request) / capacity.gpu if capacity.gpu else 0.0,
                    )
                    entry = volumes.setdefault(receipt.previous_dc_id, [0.0, 0.0, 0.0])
                    entry[index] += freshness * volume
                    entry[2] += freshness
        return {j: LocalSourceSignal(*v) for j, v in volumes.items()}

    def build_action_context(self, policy, env, decision, feedback_provider):
        self.game.advance_time(decision.env_time)
        signals = self.local_source_signals(
            decision.agent_id, env.dc_map[decision.agent_id], decision.env_time
        )
        job = env.job_map[decision.job_id]
        return policy.build_action_context(
            decision.agent_id,
            feedback_provider,
            cpu_request=job.cpu_request,
            gpu_request=job.gpu_request,
            source_signals=signals,
        )

    def absorption_summary(self):
        return self.game.absorption_store.episode_summary()


"""Training, evaluation, logs, and checkpoint helpers."""

import csv
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple, Union

import numpy as np
import torch
from numpy.typing import NDArray

import config as conf
from environment.cloud_edge_env import CloudEdgeEnv

from .experience import (
    FinalizedJobTrace,
    HostReplayBuffer,
    PendingJobTraceStore,
    RoutingReplayBuffer,
)
from .h_masac_agent import (
    HostSACConfig,
    LocalHostSAC,
    RoutingMASAC,
    RoutingMASACConfig,
    atomic_checkpoint_text_write,
)
from .observations import (
    HostObservationBuilder,
    NeighborHistoricalFeedbackStore,
    RoutingCentralizedStateBuilder,
    RoutingObservationBuilder,
    ShortWindowFeedbackStore,
)


@dataclass(frozen=True)
class TrainingRewardConfig:
    """
    H-MASAC Training Reward 参数。

    注意：
        这里的参数属于“学习目标”，
        不属于 CloudEdgeEnv 的物理环境状态。

    Environment 只负责产生事实：
        latency
        elapsed time
        completion time
        energy
        terminal reason
        ...

    本类负责把这些事实映射成训练 Reward。
    """

    task_completion_reward: float

    completion_time_cost_weight: float
    sla_violation_cost_weight: float

    remote_offload_base_penalty: float
    remote_latency_cost_weight: float
    sla_risk_cost_weight: float

    timeout_drop_penalty: float
    resource_drop_penalty: float

    energy_normalization_j: float
    energy_cost_weight: float

    max_latency_s: float
    max_job_duration_s: float

    sla_deadline_ratio: float
    drop_deadline_ratio: float

    norm_eps: float = 1e-8


class HMasacTrainingRewardModel:
    """
    H-MASAC 两层训练系统使用的 Reward Model。

    本类不：
        - 推进 Environment；
        - 修改 Job；
        - 修改 Queue；
        - 修改 DataCenter；
        - 产生 Event。

    它只完成：

        Physical / Outcome Facts
                  ↓
             Training Reward

    第三十一步暂时保持现有 Reward 公式不变，
    只把 Reward calculation 从 Environment 中移出来。
    """

    def __init__(
        self,
        config: TrainingRewardConfig,
    ) -> None:

        self.config = config

        if float(config.energy_normalization_j) <= 0.0:
            raise ValueError("energy_normalization_j 必须 > 0。")

        if float(config.max_latency_s) <= 0.0:
            raise ValueError("max_latency_s 必须 > 0。")

        if float(config.max_job_duration_s) <= 0.0:
            raise ValueError("max_job_duration_s 必须 > 0。")

    def _normalize(
        self,
        value: float,
        scale: float,
    ) -> float:

        scale = max(
            float(scale),
            float(self.config.norm_eps),
        )

        return float(
            np.clip(
                max(
                    float(value),
                    0.0,
                )
                / scale,
                0.0,
                1.0,
            )
        )

    def _calculate_sla_violation_degree(
        self,
        *,
        job_duration_s: float,
        completion_time_s: float,
    ) -> float:

        job_duration_s = max(
            float(job_duration_s),
            float(self.config.norm_eps),
        )

        completion_time_s = max(
            float(completion_time_s),
            0.0,
        )

        sla_limit_s = float(self.config.sla_deadline_ratio) * job_duration_s

        drop_limit_s = float(self.config.drop_deadline_ratio) * job_duration_s

        violation_window_s = max(
            drop_limit_s - sla_limit_s,
            float(self.config.norm_eps),
        )

        violation_degree = (completion_time_s - sla_limit_s) / violation_window_s

        return float(
            np.clip(
                violation_degree,
                0.0,
                1.0,
            )
        )

    def _calculate_energy_penalty(
        self,
        attributable_energy_j: float,
    ) -> float:

        energy_cost = max(
            float(attributable_energy_j),
            0.0,
        ) / float(self.config.energy_normalization_j)

        return float(float(self.config.energy_cost_weight) * energy_cost)

    def calculate_routing_immediate_reward(
        self,
        facts: Mapping[
            str,
            Any,
        ],
    ) -> float:
        """
        根据一次 Routing action 已经真实发生的物理事实，
        生成 Routing MASAC immediate reward。

        当前保持第三十一步之前的 Reward 数值语义：

        Self:
            0

        Edge / Cloud:
            remote base penalty
            + latency cost
            + predicted SLA risk

        Forced Drop:
            timeout/resource penalty
            + 已产生 attributable energy
        """

        action_type = str(facts["action_type"])

        if action_type == "self":
            return 0.0

        if action_type == "drop":

            failure_reason = str(
                facts.get(
                    "failure_reason",
                    "",
                )
            )

            energy_penalty = self._calculate_energy_penalty(
                facts.get(
                    "total_attributable_energy_j",
                    0.0,
                )
            )

            if failure_reason in {
                "waiting_timeout",
                "等待超时",
            }:
                return -float(float(self.config.timeout_drop_penalty) + energy_penalty)

            if failure_reason in {
                "resource_failure",
                "资源不足",
            }:
                return -float(float(self.config.resource_drop_penalty) + energy_penalty)

            raise ValueError("未知 Routing drop failure_reason：" f"{failure_reason}")

        if action_type in {
            "edge_dc",
            "cloud",
        }:

            transfer_latency_s = max(
                float(
                    facts.get(
                        "transfer_latency_s",
                        0.0,
                    )
                ),
                0.0,
            )

            elapsed_service_time_s = max(
                float(
                    facts.get(
                        "elapsed_service_time_s",
                        0.0,
                    )
                ),
                0.0,
            )

            job_duration_s = max(
                float(facts["job_duration_s"]),
                float(self.config.norm_eps),
            )

            remote_latency_cost = self._normalize(
                value=(transfer_latency_s),
                scale=(self.config.max_latency_s),
            )

            predicted_completion_time_s = (
                elapsed_service_time_s + transfer_latency_s + job_duration_s
            )

            predicted_sla_violation_degree = self._calculate_sla_violation_degree(
                job_duration_s=(job_duration_s),
                completion_time_s=(predicted_completion_time_s),
            )

            return -float(
                float(self.config.remote_offload_base_penalty)
                + float(self.config.remote_latency_cost_weight) * remote_latency_cost
                + float(self.config.sla_risk_cost_weight)
                * predicted_sla_violation_degree
            )

        raise ValueError("未知 Routing action_type：" f"{action_type}")

    def calculate_outcome_reward(
        self,
        facts: Mapping[
            str,
            Any,
        ],
    ) -> float:
        """
        Environment Outcome Fact
                ↓
        H-MASAC delayed/terminal training reward。

        当前第三十一步仍保持原有 terminal Reward 数值公式。
        """

        reason = str(facts["reason"])

        attributable_energy_j = float(
            facts.get(
                "total_attributable_energy_j",
                0.0,
            )
        )

        energy_penalty = self._calculate_energy_penalty(attributable_energy_j)

        if reason == "completed":

            completion_time_s = max(
                float(facts["completion_time_s"]),
                0.0,
            )

            job_duration_s = max(
                float(facts["job_duration_s"]),
                float(self.config.norm_eps),
            )

            global_completion_scale_s = max(
                float(self.config.drop_deadline_ratio)
                * float(self.config.max_job_duration_s),
                float(self.config.norm_eps),
            )

            completion_time_cost = self._normalize(
                value=(completion_time_s),
                scale=(global_completion_scale_s),
            )

            sla_violation_degree = self._calculate_sla_violation_degree(
                job_duration_s=(job_duration_s),
                completion_time_s=(completion_time_s),
            )

            completion_time_penalty = (
                float(self.config.completion_time_cost_weight) * completion_time_cost
            )

            sla_violation_penalty = (
                float(self.config.sla_violation_cost_weight) * sla_violation_degree
            )

            return float(
                float(self.config.task_completion_reward)
                - completion_time_penalty
                - sla_violation_penalty
                - energy_penalty
            )

        if reason in {
            "waiting_timeout",
            "local_host_arrival_timeout",
            "cloud_arrival_timeout",
        }:
            return -float(float(self.config.timeout_drop_penalty) + energy_penalty)

        if reason in {
            "local_host_resource_failure",
            "cloud_resource_failure",
        }:
            return -float(float(self.config.resource_drop_penalty) + energy_penalty)

        raise ValueError("TrainingRewardModel " "收到未知 outcome reason：" f"{reason}")


FloatArray = NDArray[np.float32]


class CloudEdgeEnvLike(Protocol):

    possible_agents: Sequence[str]
    agents: Sequence[str]

    agent_selection: Optional[str]
    current_agent_id: Optional[str]

    current_job_id: Optional[str]
    current_time: float
    action_dim: int
    drop_action: int
    has_reset: bool

    agent_name_mapping: Mapping[str, int]

    terminations: Mapping[str, bool]
    truncations: Mapping[str, bool]

    def step(self, action: Optional[int]) -> None: ...

    def pop_last_routing_action_facts(
        self,
    ) -> Dict[str, Any]: ...


@dataclass(frozen=True)
class DecisionSnapshot:
    agent_id: str
    agent_index: int
    job_id: str
    env_time: float
    local_obs: FloatArray

    global_state: FloatArray

    forced_action: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_index": self.agent_index,
            "job_id": self.job_id,
            "env_time": self.env_time,
            "local_obs": self.local_obs.copy(),
            "global_state": self.global_state.copy(),
            "forced_action": self.forced_action,
        }


@dataclass(frozen=True)
class RoutingActionResult:
    """
    一次 Routing action 执行完成后的即时结果。

    注意：
        RoutingActionResult 不是 Replay Transition。

    它只服务于 Trainer：
        - Episode statistics；
        - 判断当前 Job 是否已经 Finalize；
        - 识别本次动作类型；
        - 继续环境控制流。

    正式 RoutingTransition 只能在 Job terminal 后，
    由 PendingJobTraceStore / FinalizedJobTrace 构造。
    """

    agent_id: str
    agent_index: int

    job_id: str

    decision_time: float

    result_time: float

    action: int
    action_type: str
    action_source: str

    immediate_reward: float

    target_dc_id: Optional[str]

    job_finalized: bool

    def as_dict(
        self,
    ) -> Dict[str, Any]:

        return {
            "agent_id": self.agent_id,
            "agent_index": self.agent_index,
            "job_id": self.job_id,
            "decision_time": self.decision_time,
            "result_time": self.result_time,
            "action": self.action,
            "action_type": self.action_type,
            "action_source": self.action_source,
            "immediate_reward": self.immediate_reward,
            "target_dc_id": self.target_dc_id,
            "job_finalized": self.job_finalized,
        }


class TransitionCollector:
    def __init__(
        self,
        env: CloudEdgeEnvLike,
        routing_observation_builder: RoutingObservationBuilder,
        routing_state_builder: RoutingCentralizedStateBuilder,
        pending_trace_store: PendingJobTraceStore,
        training_reward_model: HMasacTrainingRewardModel,
        short_window_runtime=None,
        state_heuristic=None,
    ) -> None:

        self.env = env
        self.short_window_runtime = short_window_runtime
        self.state_heuristic = state_heuristic

        self.routing_observation_builder = routing_observation_builder

        self.routing_state_builder = routing_state_builder

        self.pending_trace_store = pending_trace_store
        self.training_reward_model = training_reward_model

        self.total_routing_action_count = 0
        self.episode_routing_action_count = 0

        self._validate_environment_interface()

    def reset_episode(self) -> None:
        self.episode_routing_action_count = 0

    def _build_current_decision_snapshot(
        self,
    ) -> DecisionSnapshot:
        """
        根据环境当前时刻构造 Routing DecisionSnapshot。

        本函数只服务于 Routing 层 PettingZoo Agent。

        主要职责：
            1. 获取当前真正需要做 Routing 决策的 Edge DC；
            2. 构造 Routing Actor 的 local observation；
            3. 构造 Routing centralized critic state；
            4. 构造当前 DecisionSnapshot；
            5. 如果当前 Job 是之前某个 Edge->Edge Routing action
               到达下一 Edge 后产生的 same-job successor，
               则在 PendingJobTraceStore 中完成上一跳 next_state 回填。

        注意：
            Host 层不会调用本函数。
        """

        self._require_reset()

        agent_id = self._get_live_selected_agent()

        if self.env.current_job_id is None:
            raise RuntimeError(
                "构造 Routing DecisionSnapshot 时 "
                "current_job_id 为 None："
                f"agent={agent_id}"
            )

        job_id = str(self.env.current_job_id)
        if self.short_window_runtime is not None:
            self.short_window_runtime.record_arrival(
                agent_id, job_id, float(self.env.current_time)
            )

        local_obs = np.asarray(
            self.routing_observation_builder.build(agent_id),
            dtype=np.float32,
        ).copy()

        global_state = np.asarray(
            self.routing_state_builder.build(),
            dtype=np.float32,
        ).copy()

        snapshot = DecisionSnapshot(
            agent_id=agent_id,
            agent_index=int(self.env.agent_name_mapping[agent_id]),
            job_id=job_id,
            env_time=float(self.env.current_time),
            local_obs=local_obs,
            global_state=global_state,
            forced_action=(self._get_forced_action(job_id)),
        )

        score = (
            self.state_heuristic.capture(self.env, snapshot)
            if self.state_heuristic is not None
            else 0.0
        )
        self.pending_trace_store.resolve_routing_successor(
            next_heuristic_score=score,
            heuristic_valid=self.state_heuristic is not None,
            job_id=snapshot.job_id,
            next_agent_id=(snapshot.agent_id),
            next_agent_index=(snapshot.agent_index),
            next_env_time=(snapshot.env_time),
            next_local_obs=(snapshot.local_obs),
            next_global_state=(snapshot.global_state),
        )

        return snapshot

    def capture_decision(self) -> DecisionSnapshot:
        self._require_reset()
        return self._build_current_decision_snapshot()

    def _validate_decision_is_current(
        self,
        decision: DecisionSnapshot,
    ) -> None:
        """
        检查 Trainer 保存的 Routing DecisionSnapshot
        是否仍然对应 Environment 当前真正等待执行的
        Routing decision。

        主要防止：

            1. Host Phase 后错误复用旧 Routing snapshot；
            2. 当前 Routing Agent 已经改变；
            3. 当前 Job 已经改变；
            4. 对过期状态执行 Actor action。

        注意：
            本函数只验证 Routing decision 的 identity，
            不负责构造 Observation 或执行 action。
        """

        current_agent_id = self._get_live_selected_agent()

        if current_agent_id != str(decision.agent_id):
            raise RuntimeError(
                "Routing DecisionSnapshot 已过期："
                f"snapshot_agent={decision.agent_id}, "
                f"current_agent={current_agent_id}, "
                f"job={decision.job_id}"
            )

        if self.env.current_job_id is None:
            raise RuntimeError(
                "执行 Routing action 时 "
                "Environment current_job_id 为 None："
                f"snapshot_agent={decision.agent_id}, "
                f"snapshot_job={decision.job_id}"
            )

        current_job_id = str(self.env.current_job_id)

        if current_job_id != str(decision.job_id):
            raise RuntimeError(
                "Routing DecisionSnapshot Job 已过期："
                f"snapshot_job={decision.job_id}, "
                f"current_job={current_job_id}, "
                f"agent={current_agent_id}"
            )

    def execute_and_record(
        self,
        decision: DecisionSnapshot,
        action: int,
        action_source: str,
    ) -> Tuple[
        RoutingActionResult,
        Optional[DecisionSnapshot],
    ]:
        """
        执行一次 PettingZoo Routing action，
        并把已经真实发生的 Routing 因果事实写入
        PendingJobTraceStore。

        第二十一步以后，本函数不再生成任何正式训练 Transition。

        正式 RoutingTransition 只能在：

            Job terminal
                ↓
            FinalizedJobTrace
                ↓
            _build_finalized_routing_transitions()

        阶段生成。

        本函数只负责：

            DecisionSnapshot
                ↓
            Routing action
                ↓
            Environment.step()
                ↓
            immediate reward
                ↓
            PendingRoutingStep

        如果是 forced drop：

            mark terminal
                ↓
            finalize complete Job trace

        返回的 next_decision 只是 Trainer 环境时间线上的
        下一条 Routing Decision。

        它不一定属于当前 Job，
        因而绝不在本函数中被当作当前 Job 的 Replay next_state。
        """

        self._require_reset()

        self._validate_decision_is_current(decision)

        action = int(action)

        action_source = str(action_source)

        if action_source not in {
            "forced",
            "random",
            "policy",
            "orchestrator",
        }:
            raise ValueError(
                "TransitionCollector 收到未知 action_source：" f"{action_source}"
            )

        if decision.forced_action is not None:

            if action_source != "forced":
                raise RuntimeError(
                    "当前 Routing Decision 必须执行 forced action，"
                    "但 Trainer 给出的 action_source 不是 forced："
                    f"job={decision.job_id}, "
                    f"action_source={action_source}"
                )

            if int(decision.forced_action) != action:
                raise RuntimeError(
                    "Trainer 执行的 forced action "
                    "与 DecisionSnapshot 不一致："
                    f"job={decision.job_id}, "
                    f"expected={decision.forced_action}, "
                    f"actual={action}"
                )

        elif action_source == "forced":

            raise RuntimeError(
                "DecisionSnapshot 没有 forced action，"
                "但 Trainer 将当前动作标记为 forced："
                f"job={decision.job_id}, "
                f"action={action}"
            )

        (
            action_type,
            target_dc_id,
        ) = self._decode_routing_action(
            agent_id=(decision.agent_id),
            action=action,
        )

        self.env.step(action)

        routing_action_facts = self.env.pop_last_routing_action_facts()

        immediate_reward = float(
            self.training_reward_model.calculate_routing_immediate_reward(
                routing_action_facts
            )
        )

        self.pending_trace_store.record_routing_step(
            job_id=(decision.job_id),
            agent_id=(decision.agent_id),
            agent_index=(decision.agent_index),
            env_time=(decision.env_time),
            local_obs=(decision.local_obs),
            global_state=(decision.global_state),
            action=action,
            action_type=(action_type),
            action_source=(action_source),
            immediate_reward=(immediate_reward),
            target_dc_id=(target_dc_id),
        )

        if action_type == "drop":
            self.pending_trace_store.mark_terminal(
                job_id=(decision.job_id),
                env_time=float(self.env.current_time),
                reason="forced_drop",
            )

            self.pending_trace_store.finalize_terminal_trace(job_id=(decision.job_id))

        if self.short_window_runtime is not None:
            self.short_window_runtime.record_routing(
                decision,
                action_type,
                target_dc_id,
                action_source,
                self.env.job_map[decision.job_id],
            )

        environment_done = self._is_episode_done()

        next_decision: Optional[DecisionSnapshot] = None

        if not environment_done and self.env.agent_selection is not None:
            next_decision = self._build_current_decision_snapshot()

        routing_result = RoutingActionResult(
            agent_id=str(decision.agent_id),
            agent_index=int(decision.agent_index),
            job_id=str(decision.job_id),
            decision_time=float(decision.env_time),
            result_time=float(self.env.current_time),
            action=action,
            action_type=(action_type),
            action_source=(action_source),
            immediate_reward=(immediate_reward),
            target_dc_id=(target_dc_id),
            job_finalized=(
                self.pending_trace_store.has_finalized_trace(decision.job_id)
            ),
        )

        self.total_routing_action_count += 1
        self.episode_routing_action_count += 1

        return (
            routing_result,
            next_decision,
        )

    def drain_one_dead_agent(
        self,
    ) -> bool:
        """
        按 PettingZoo AEC 约定，
        对当前已经 terminal / truncated 的 Routing Agent
        执行一次 step(None)。

        Host Phase 下 agent_selection=None，
        此时本函数什么都不做。
        """

        if not self.env.agents:
            return False

        if self.env.agent_selection is None:
            return False

        agent_id = str(self.env.agent_selection)

        is_dead = bool(
            self.env.terminations.get(
                agent_id,
                False,
            )
            or self.env.truncations.get(
                agent_id,
                False,
            )
        )

        if not is_dead:
            return False

        self.env.step(None)

        return True

    def drain_all_dead_agents(self) -> int:
        count = 0
        while self.env.agents:
            if not self.drain_one_dead_agent():
                break
            count += 1
        return count

    def _validate_environment_interface(self) -> None:
        required_attributes = (
            "possible_agents",
            "agents",
            "agent_selection",
            "current_job_id",
            "current_time",
            "action_dim",
            "drop_action",
            "has_reset",
            "agent_name_mapping",
            "rewards",
            "terminations",
            "truncations",
        )

        for name in required_attributes:
            if not hasattr(self.env, name):
                raise AttributeError(f"环境缺少 TransitionCollector 所需属性：{name}。")

        for method_name in (
            "step",
            "_decode_action",
            "_should_drop_arrival_job",
        ):
            method = getattr(
                self.env,
                method_name,
                None,
            )

            if not callable(method):
                raise AttributeError(
                    "Environment 缺少 TransitionCollector "
                    "所需可调用方法："
                    f"{method_name}()"
                )

        if int(self.env.action_dim) <= 0:
            raise ValueError("action_dim 必须大于 0。")
        if int(self.routing_observation_builder.obs_dim) <= 0:
            raise ValueError("routing_obs_dim 必须大于 0。")

        if int(self.routing_state_builder.state_dim) <= 0:
            raise ValueError("routing_global_state_dim " "必须大于 0。")

    def _require_reset(self) -> None:
        if not bool(self.env.has_reset):
            raise RuntimeError("环境尚未 reset，不能采集 Transition。")

    def _get_live_selected_agent(
        self,
    ) -> str:
        """
        返回当前真正处于 PettingZoo Routing Phase 的 Agent。

        Host Phase 下：
            agent_selection == None

        因此必须立即拒绝，
        不能把 None 转换成字符串 "None"。
        """

        if self.env.agent_selection is None:
            raise RuntimeError(
                "当前不存在 PettingZoo Routing Agent。"
                "如果存在 pending Host decision，"
                "应由 Local Host SAC 分支处理，"
                "不能调用 Routing TransitionCollector。"
            )

        agent_id = str(self.env.agent_selection)

        if agent_id not in self.env.possible_agents:
            raise RuntimeError(
                "当前 agent_selection 不是合法 Routing Agent："
                f"agent={agent_id}, "
                f"possible_agents={self.env.possible_agents}"
            )

        if agent_id not in self.env.agents:
            raise RuntimeError(
                "当前 Routing Agent 已不在活跃 agents 中："
                f"agent={agent_id}, "
                f"agents={self.env.agents}"
            )

        return agent_id

    def _is_episode_done(self) -> bool:
        if len(self.env.possible_agents) == 0:
            return True
        return all(
            bool(
                self.env.terminations.get(agent_id, False)
                or self.env.truncations.get(agent_id, False)
            )
            for agent_id in self.env.possible_agents
        )

    def _get_forced_action(self, job_id: str) -> Optional[int]:
        should_drop_fn = getattr(self.env, "_should_drop_arrival_job", None)
        if bool(should_drop_fn(job_id)):
            return int(self.env.drop_action)
        return None

    def _decode_routing_action(
        self,
        *,
        agent_id: str,
        action: int,
    ) -> Tuple[
        str,
        Optional[str],
    ]:
        """
        使用 Environment 的唯一动作语义定义解析 Routing action。

        Collector 不复制：
            Self index
            Edge index
            Cloud index

        DROP_ACTION=-1 同样由 Environment 解码。
        """

        decode_action = getattr(
            self.env,
            "_decode_action",
            None,
        )

        if not callable(decode_action):
            raise AttributeError(
                "TransitionCollector 需要 Environment 提供 " "_decode_action()。"
            )

        decoded_action = decode_action(
            agent_id=str(agent_id),
            action=int(action),
        )

        action_type = str(
            decoded_action.get(
                "action_type",
                "",
            )
        )

        if action_type not in {
            "self",
            "edge_dc",
            "cloud",
            "drop",
        }:
            raise RuntimeError(
                "Environment 返回未知 Routing action_type："
                f"agent={agent_id}, "
                f"action={action}, "
                f"action_type={action_type}"
            )

        raw_target_dc_id = decoded_action.get(
            "target_dc_id",
            None,
        )

        target_dc_id = None if raw_target_dc_id is None else str(raw_target_dc_id)

        return (
            action_type,
            target_dc_id,
        )


class StateScoreLog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("x", encoding="utf-8")
        self.episode = 0

    def record(self, decision, score, rows, settings):
        self.file.write(
            json.dumps(
                {
                    "episode": self.episode,
                    "time": decision.env_time,
                    "job_id": decision.job_id,
                    "agent_id": decision.agent_id,
                    "score": score,
                    "score_version": settings.heuristic_version,
                    "candidates": [asdict(row) for row in rows],
                },
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )

    def flush(self):
        self.file.flush()
        os.fsync(self.file.fileno())

    def close(self):
        self.file.close()


class RoutingStateHeuristic:
    """Capture-time state quality from uncentered confidence-adjusted utility."""

    def __init__(self, runtime, feedback, settings, targets, log=None):
        validate_bcgh_feature_config(settings)
        self.runtime, self.feedback, self.settings = runtime, feedback, settings
        self.action_target_dc_ids = tuple(targets)
        self.log = log

    def build_action_context(
        self,
        source_dc_id,
        historical_feedback_provider=None,
        *,
        cpu_request=0.0,
        gpu_request=0.0,
        source_signals=None,
    ):
        context = {}
        for target in self.action_target_dc_ids:
            feedback = None
            if target != source_dc_id and historical_feedback_provider is not None:
                feedback = historical_feedback_provider.get_feedback(
                    source_dc_id, target
                )
            success = 0.5 if feedback is None else feedback.get("success_ewma", 0.5)
            sla = 0.5 if feedback is None else feedback.get("sla_success_ewma", 0.5)
            delay = (
                0.5
                if feedback is None
                else 1 - feedback.get("completion_time_ewma", 0.5)
            )
            context[target] = RoutingActionFeatures(
                target,
                float(np.clip(success, 0, 1)),
                float(np.clip(sla, 0, 1)),
                float(np.clip(delay, 0, 1)),
                cpu_request,
                gpu_request,
                (source_signals or {}).get(target, LocalSourceSignal()),
            )
        return context

    def capture(self, env, decision):
        if (
            not self.settings.enable_heuristic_guidance
            or decision.forced_action is not None
        ):
            return 0.0
        context = self.runtime.build_action_context(self, env, decision, self.feedback)
        rows = self.runtime.game.evaluate_actions(decision.agent_id, context)
        score = state_score_from_utilities(rows, self.settings)
        if self.log is not None:
            self.log.record(decision, score, rows, self.settings)
        return score


def state_score_from_utilities(rows, settings):
    feasible = [row for row in rows if row.feasible]
    if not feasible:
        raise ValueError("No reference candidate for routing state score")
    values = [row.guidance_utility for row in feasible]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Non-finite candidate utility")
    positive(settings.heuristic_scale, "heuristic_scale", zero=True)
    if not math.isfinite(settings.heuristic_neutral_utility):
        raise ValueError("heuristic_neutral_utility must be finite")
    return (
        settings.heuristic_scale
        * sum(math.tanh(value - settings.heuristic_neutral_utility) for value in values)
        / len(values)
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BCGH_MASAC_DIR = Path(__file__).resolve().parent

CHECKPOINT_SCHEMA_VERSION = 2

CHECKPOINT_ARCHITECTURE = "bcgh2_masca_two_layer_routing_host_v1"

TERMINAL_FAILURE_REASONS = frozenset(
    {
        "waiting_timeout",
        "cloud_arrival_timeout",
        "cloud_resource_failure",
        "local_host_arrival_timeout",
        "local_host_resource_failure",
    }
)

UPDATE_TENSOR_METRIC_NAMES = (
    "critic_loss",
    "q1_loss",
    "q2_loss",
    "mean_q1",
    "mean_q2",
    "mean_target_q",
    "actor_loss",
    "alpha_loss",
    "alpha",
    "policy_entropy",
    "target_entropy",
    "heuristic_reward_mean",
    "modified_reward_mean",
    "raw_reward_mean",
    "effective_gamma",
)

ENVIRONMENT_DIR = PROJECT_ROOT / "environment"


@dataclass(frozen=True)
class TrainConfig:

    num_episodes: int = conf.Episodes

    routing_replay_capacity: int = conf.ROUTING_REPLAY_CAPACITY
    host_replay_capacity: int = conf.HOST_REPLAY_CAPACITY
    routing_batch_size: int = conf.ROUTING_BATCH_SIZE
    routing_random_warmup_steps: int = conf.ROUTING_RANDOM_WARMUP_STEPS
    routing_learning_starts: int = conf.ROUTING_LEARNING_STARTS
    routing_train_every: int = conf.ROUTING_TRAIN_EVERY
    routing_updates_per_train: int = conf.ROUTING_UPDATES_PER_TRAIN

    host_batch_size: int = conf.HOST_BATCH_SIZE
    host_random_warmup_steps: int = conf.HOST_RANDOM_WARMUP_STEPS
    host_learning_starts: int = conf.HOST_LEARNING_STARTS
    host_train_every: int = conf.HOST_TRAIN_EVERY
    host_updates_per_train: int = conf.HOST_UPDATES_PER_TRAIN

    host_pretrain_episodes: int = conf.HOST_PRETRAIN_EPISODES
    routing_train_episodes: int = conf.ROUTING_TRAIN_EPISODES
    joint_finetune_episodes: int = conf.JOINT_FINETUNE_EPISODES
    log_interval: int = conf.Log_interval
    checkpoint_interval: int = (
        conf.Checkpoint_Interval
    )  # 兼容旧调用，双层训练不按间隔保存
    seed: int = conf.Seed

    checkpoint_dir: str = "model/BCGH2-MASCA/checkpoints"
    episode_log_csv_path: str = "result/BCGH2-MASCA/episode_log.csv"
    dc_log_csv_path: str = "result/BCGH2-MASCA/dc_log.csv"
    old_env_path: Optional[str] = conf.Old_Env_Path
    resume_checkpoint: Optional[str] = None
    routing_init_checkpoint: Optional[str] = None

    vary_episode_seed: bool = conf.Vary_Episode_Seed
    collect_neighbor_historical_feedback: bool = (
        conf.COLLECT_NEIGHBOR_HISTORICAL_FEEDBACK
    )
    use_neighbor_historical_feedback: bool = conf.USE_NEIGHBOR_HISTORICAL_FEEDBACK
    neighbor_feedback_ewma_alpha: float = conf.NEIGHBOR_FEEDBACK_EWMA_ALPHA
    neighbor_feedback_age_scale_samples: float = (
        conf.NEIGHBOR_FEEDBACK_AGE_SCALE_SAMPLES
    )
    neighbor_feedback_confidence_scale_samples: float = (
        conf.NEIGHBOR_FEEDBACK_CONFIDENCE_SCALE_SAMPLES
    )
    enable_bayesian_game: bool = True
    enable_heuristic_guidance: bool = True
    enable_reward_shaping: bool = True
    enable_discount_reduction: bool = True
    heuristic_beta: float = 0.98
    heuristic_scale: float = 1.0
    heuristic_neutral_utility: float = 0.5
    heuristic_reference_distribution: str = "uniform_static_feasible"
    heuristic_version: int = 1
    bayesian_prior_alpha: float = conf.BCGH_BAYESIAN_PRIOR_ALPHA
    bayesian_prior_beta: float = conf.BCGH_BAYESIAN_PRIOR_BETA
    bayesian_confidence_scale: float = conf.BCGH_BAYESIAN_CONFIDENCE_SCALE
    short_window_s: float = conf.BCGH_SHORT_WINDOW_S
    source_evidence_weight: float = conf.BCGH_SOURCE_EVIDENCE_WEIGHT
    source_volume_scale: float = conf.BCGH_SOURCE_VOLUME_SCALE
    source_confidence_scale: float = conf.BCGH_SOURCE_CONFIDENCE_SCALE
    resource_cpu_weight: float = conf.BCGH_RESOURCE_CPU_WEIGHT
    resource_gpu_weight: float = conf.BCGH_RESOURCE_GPU_WEIGHT
    resource_cost_scale: float = conf.BCGH_RESOURCE_COST_SCALE
    absorption_enabled: bool = conf.BCGH_ABSORPTION_ENABLED
    absorption_prior_self: float = conf.BCGH_ABSORPTION_PRIOR_SELF
    absorption_prior_edge: float = conf.BCGH_ABSORPTION_PRIOR_EDGE
    absorption_prior_cloud: float = conf.BCGH_ABSORPTION_PRIOR_CLOUD
    absorption_prior_drop: float = conf.BCGH_ABSORPTION_PRIOR_DROP
    absorption_confidence_scale: float = conf.BCGH_ABSORPTION_CONFIDENCE_SCALE
    absorption_edge_weight: float = conf.BCGH_ABSORPTION_EDGE_WEIGHT
    absorption_cloud_weight: float = conf.BCGH_ABSORPTION_CLOUD_WEIGHT
    absorption_drop_weight: float = conf.BCGH_ABSORPTION_DROP_WEIGHT
    absorption_demand_thresholds: tuple = conf.BCGH_ABSORPTION_DEMAND_THRESHOLDS
    absorption_class_confidence_scale: float = (
        conf.BCGH_ABSORPTION_CLASS_CONFIDENCE_SCALE
    )
    absorption_policy_actions_only: bool = conf.BCGH_ABSORPTION_POLICY_ACTIONS_ONLY
    pressure_linear_cost_weight: float = conf.BCGH_PRESSURE_LINEAR_WEIGHT
    pressure_quadratic_cost_weight: float = conf.BCGH_PRESSURE_QUADRATIC_WEIGHT
    benefit_success_weight: float = conf.BCGH_BENEFIT_SUCCESS_WEIGHT
    benefit_sla_weight: float = conf.BCGH_BENEFIT_SLA_WEIGHT
    benefit_delay_weight: float = conf.BCGH_BENEFIT_DELAY_WEIGHT
    risk_congestion_cost_weight: float = conf.BCGH_RISK_CONGESTION_COST_WEIGHT


def bcgh_guidance_metadata(train_config):
    """Runtime semantics are part of the checkpoint even when tensor shapes match."""
    names = (
        "enable_bayesian_game",
        "enable_heuristic_guidance",
        "enable_reward_shaping",
        "enable_discount_reduction",
        "heuristic_beta",
        "heuristic_scale",
        "heuristic_neutral_utility",
        "heuristic_reference_distribution",
        "heuristic_version",
        "short_window_s",
        "bayesian_prior_alpha",
        "bayesian_prior_beta",
        "bayesian_confidence_scale",
        "source_evidence_weight",
        "source_volume_scale",
        "source_confidence_scale",
        "resource_cpu_weight",
        "resource_gpu_weight",
        "resource_cost_scale",
        "absorption_enabled",
        "absorption_prior_self",
        "absorption_prior_edge",
        "absorption_prior_cloud",
        "absorption_prior_drop",
        "absorption_confidence_scale",
        "absorption_edge_weight",
        "absorption_cloud_weight",
        "absorption_drop_weight",
        "absorption_demand_thresholds",
        "absorption_class_confidence_scale",
        "absorption_policy_actions_only",
        "pressure_linear_cost_weight",
        "pressure_quadratic_cost_weight",
        "benefit_success_weight",
        "benefit_sla_weight",
        "benefit_delay_weight",
        "risk_congestion_cost_weight",
        "collect_neighbor_historical_feedback",
        "use_neighbor_historical_feedback",
    )
    parameters = {}
    for name in names:
        value = getattr(train_config, name)
        parameters[name] = list(value) if isinstance(value, tuple) else value
    return {
        "version": 1,
        "scheduler": "BCGH2-MASCA",
        "method": "same_job_next_state_reward_discount_reformulation",
        "data_augmentation": False,
        "actor_logit_bias": False,
        "clock": "episode_simulation_seconds",
        "source_semantics": "local_queue_direct_previous_hop_bounded_soft_evidence",
        "pressure_semantics": "executed_edge_transfer_cpu_gpu_marginal_cost",
        "absorption_semantics": "directed_edge_successor_self_edge_cloud_drop",
        "parameters": parameters,
    }


def is_bcgh_zero_diff_mode(
    train_config: TrainConfig,
) -> bool:
    """
    判断贝叶斯运行状态与后继状态引导是否都关闭。

    Zero-Diff Mode：
        Bayesian Game        = False
        Heuristic Guidance   = False
    """

    return not bool(train_config.enable_bayesian_game) and not bool(
        train_config.enable_heuristic_guidance
    )


def get_bcgh_runtime_mode(
    train_config: TrainConfig,
) -> str:
    """
    返回当前 BCGH2-MASCA 的实验模式名称。

    历史反馈输入由独立配置控制，因此关闭这两个功能不意味着
    与所有其他调度器的观测完全相同。
    """

    bayesian_enabled = bool(train_config.enable_bayesian_game)

    heuristic_enabled = bool(train_config.enable_heuristic_guidance)

    if not bayesian_enabled and not heuristic_enabled:
        return "SAC-without-Bayesian-guidance"

    if bayesian_enabled and not heuristic_enabled:
        return "Bayesian-only"

    if not bayesian_enabled and heuristic_enabled:
        return "Invalid-guidance-without-Bayesian-runtime"

    return "Bayesian+NextStateRewardDiscount"


def validate_bcgh_feature_config(train_config):
    if train_config.enable_heuristic_guidance and not train_config.enable_bayesian_game:
        raise ValueError("BCGH2 state guidance requires Bayesian runtime")
    if (
        not math.isfinite(train_config.heuristic_beta)
        or not 0 <= train_config.heuristic_beta <= 1
    ):
        raise ValueError("heuristic_beta must be in [0, 1]")
    positive(train_config.heuristic_scale, "heuristic_scale", zero=True)
    if not math.isfinite(train_config.heuristic_neutral_utility):
        raise ValueError("heuristic_neutral_utility must be finite")
    if (
        train_config.heuristic_reference_distribution != "uniform_static_feasible"
        or train_config.heuristic_version != 1
    ):
        raise ValueError("Unsupported heuristic definition")
    if train_config.resume_checkpoint and train_config.routing_init_checkpoint:
        raise ValueError("Resume and weight initialization are mutually exclusive")


class TrainingStage(
    str,
    Enum,
):
    HOST_PRETRAIN = "host_pretrain"
    ROUTING_TRAIN = "routing_train"
    JOINT_FINETUNE = "joint_finetune"


@dataclass
class EpisodeStatistics:

    episode: int
    episode_seed: int
    training_stage: str

    per_agent_returns: Dict[str, float]
    episode_return: float = 0.0

    routing_decision_count: int = 0
    routing_forced_action_count: int = 0
    routing_orchestrator_action_count: int = 0
    routing_random_action_count: int = 0
    routing_policy_action_count: int = 0
    routing_self_count: int = 0
    routing_edge_count: int = 0
    routing_cloud_count: int = 0
    routing_drop_count: int = 0

    host_decision_count: int = 0
    host_random_action_count: int = 0
    host_policy_action_count: int = 0
    host_started_count: int = 0
    host_queued_count: int = 0
    host_dropped_count: int = 0

    routing_update_count: int = 0
    host_update_count: int = 0

    routing_update_metric_sums: Dict[
        str,
        float,
    ] = field(default_factory=dict)
    routing_update_metric_counts: Dict[
        str,
        int,
    ] = field(default_factory=dict)

    host_update_metric_sums: Dict[
        str,
        float,
    ] = field(default_factory=dict)
    host_update_metric_counts: Dict[
        str,
        int,
    ] = field(default_factory=dict)

    host_update_metric_sums_by_dc: Dict[
        str,
        Dict[str, float],
    ] = field(default_factory=dict)
    host_update_metric_counts_by_dc: Dict[
        str,
        Dict[str, int],
    ] = field(default_factory=dict)
    pending_routing_update_infos: list = field(default_factory=list)
    pending_host_update_infos: list = field(default_factory=list)

    dc_counters: Dict[
        str,
        Dict[str, int],
    ] = field(default_factory=dict)

    routing_source_target_counts: Dict[
        str,
        Dict[str, int],
    ] = field(default_factory=dict)

    terminal_trace_flushed_count: int = 0
    routing_transition_flushed_count: int = 0
    host_transition_flushed_count: int = 0
    routing_edge_hop_total: int = 0
    multi_hop_job_count: int = 0
    max_routing_hops: int = 0
    routing_edge_hops_by_job: list = field(default_factory=list)
    edge_successor_total: int = 0
    edge_successor_self_count: int = 0
    edge_successor_edge_count: int = 0
    edge_successor_cloud_count: int = 0
    edge_successor_drop_count: int = 0
    first_edge_successor_total: int = 0
    first_edge_absorbed_count: int = 0
    absorption_pair_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    routing_layer_reward_sum: float = 0.0
    host_layer_reward_sum: float = 0.0

    bayesian_game_enabled: bool = True
    heuristic_beta: float = 1.0
    routing_gamma: float = 0.99
    reward_shaping_enabled: bool = False
    discount_reduction_enabled: bool = False
    heuristic_reward_sum: float = 0.0
    heuristic_score_sum: float = 0.0
    heuristic_nonterminal_count: int = 0
    heuristic_valid_score_count: int = 0

    def _inc_dc(
        self,
        dc_id: str,
        metric_name: str,
        delta: int = 1,
    ) -> None:
        dc_id = str(dc_id)
        dc_metrics = self.dc_counters.setdefault(
            dc_id,
            {},
        )
        dc_metrics[metric_name] = int(
            dc_metrics.get(
                metric_name,
                0,
            )
            + int(delta)
        )

    @staticmethod
    def _accumulate_metric(
        sums: Dict[str, float],
        counts: Dict[str, int],
        metric_name: str,
        metric_value: float,
    ) -> None:

        metric_value = float(metric_value)

        if not np.isfinite(metric_value):
            return

        sums[metric_name] = float(
            sums.get(
                metric_name,
                0.0,
            )
            + metric_value
        )
        counts[metric_name] = int(
            counts.get(
                metric_name,
                0,
            )
            + 1
        )

    def record_routing_decision(
        self,
        agent_id: str,
        reward: float,
        action_type: str,
        action_source: str,
        target_dc_id: Optional[str],
    ) -> None:
        agent_id = str(agent_id)
        action_type = str(action_type)
        action_source = str(action_source)
        reward = float(reward)
        target_dc_id = None if target_dc_id is None else str(target_dc_id)

        self.episode_return += reward
        self.per_agent_returns[agent_id] = float(
            self.per_agent_returns.get(
                agent_id,
                0.0,
            )
            + reward
        )

        self.routing_decision_count += 1
        self._inc_dc(
            agent_id,
            "routing_decisions",
        )

        if action_source == "forced":
            self.routing_forced_action_count += 1
        elif action_source == "orchestrator":
            self.routing_orchestrator_action_count += 1
        elif action_source == "random":
            self.routing_random_action_count += 1
        elif action_source == "policy":
            self.routing_policy_action_count += 1
        else:
            raise ValueError("未知 Routing action_source：" f"{action_source}")

        if action_type == "self":
            self.routing_self_count += 1
            self._inc_dc(
                agent_id,
                "route_self_count",
            )
        elif action_type == "edge_dc":
            if target_dc_id is None:
                raise RuntimeError("统计 Edge Routing 时缺少 target_dc_id。")
            self.routing_edge_count += 1
            self._inc_dc(
                agent_id,
                "route_out_edge_count",
            )
            self._inc_dc(
                target_dc_id,
                "route_in_edge_count",
            )
        elif action_type == "cloud":
            self.routing_cloud_count += 1
            self._inc_dc(
                agent_id,
                "route_cloud_count",
            )
        elif action_type == "drop":
            self.routing_drop_count += 1
            self._inc_dc(
                agent_id,
                "route_drop_count",
            )
        else:
            raise ValueError("未知 Routing action_type：" f"{action_type}")

        target_key = str(target_dc_id) if target_dc_id is not None else "__drop__"
        source_targets = self.routing_source_target_counts.setdefault(
            agent_id,
            {},
        )

        source_targets[target_key] = int(
            source_targets.get(
                target_key,
                0,
            )
            + 1
        )

    def record_host_decision(
        self,
        dc_id: str,
        action_source: str,
    ) -> None:

        dc_id = str(dc_id)

        action_source = str(action_source)

        self.host_decision_count += 1

        self._inc_dc(
            dc_id,
            "host_decisions",
        )

        if action_source == "random":

            self.host_random_action_count += 1

            self._inc_dc(
                dc_id,
                "host_random_count",
            )

        elif action_source == "policy":

            self.host_policy_action_count += 1

            self._inc_dc(
                dc_id,
                "host_policy_count",
            )

        else:
            raise ValueError("未知 Host action_source：" f"{action_source}")

    def record_host_result(
        self,
        dc_id: str,
        execution_result: str,
    ) -> None:

        dc_id = str(dc_id)

        execution_result = str(execution_result)

        if execution_result == "started":

            self.host_started_count += 1

            self._inc_dc(
                dc_id,
                "host_started_count",
            )

        elif execution_result == "queued":

            self.host_queued_count += 1

            self._inc_dc(
                dc_id,
                "host_queued_count",
            )

        elif execution_result == "dropped":

            self.host_dropped_count += 1

            self._inc_dc(
                dc_id,
                "host_dropped_count",
            )

        else:
            raise ValueError("未知 Host execution_result：" f"{execution_result}")

    def record_delayed_training_reward(
        self,
        agent_id: str,
        reward_delta: float,
    ) -> None:

        agent_id = str(agent_id)

        reward_delta = float(reward_delta)

        self.episode_return += reward_delta

        self.per_agent_returns[agent_id] = float(
            self.per_agent_returns.get(
                agent_id,
                0.0,
            )
            + reward_delta
        )

    def record_routing_update(
        self,
        update_info: Dict[str, float],
    ) -> None:

        self.routing_update_count += 1

        for metric_name, metric_value in update_info.items():

            self._accumulate_metric(
                sums=(self.routing_update_metric_sums),
                counts=(self.routing_update_metric_counts),
                metric_name=(metric_name),
                metric_value=float(metric_value),
            )

    def record_host_update(
        self,
        dc_id: str,
        update_info: Dict[str, float],
    ) -> None:

        dc_id = str(dc_id)

        self.host_update_count += 1

        self._inc_dc(
            dc_id,
            "host_updates",
        )

        dc_sums = self.host_update_metric_sums_by_dc.setdefault(
            dc_id,
            {},
        )

        dc_counts = self.host_update_metric_counts_by_dc.setdefault(
            dc_id,
            {},
        )

        for metric_name, metric_value in update_info.items():

            metric_value = float(metric_value)

            self._accumulate_metric(
                sums=(self.host_update_metric_sums),
                counts=(self.host_update_metric_counts),
                metric_name=(metric_name),
                metric_value=(metric_value),
            )

            self._accumulate_metric(
                sums=dc_sums,
                counts=dc_counts,
                metric_name=(metric_name),
                metric_value=(metric_value),
            )

    def mean_routing_metric(
        self,
        metric_name: str,
    ) -> float:

        count = self.routing_update_metric_counts.get(
            metric_name,
            0,
        )

        if count <= 0:
            return float("nan")

        return float(self.routing_update_metric_sums[metric_name] / count)

    def mean_host_metric(
        self,
        metric_name: str,
        dc_id: Optional[str] = None,
    ) -> float:

        if dc_id is None:

            sums = self.host_update_metric_sums

            counts = self.host_update_metric_counts

        else:

            dc_id = str(dc_id)

            sums = self.host_update_metric_sums_by_dc.get(
                dc_id,
                {},
            )

            counts = self.host_update_metric_counts_by_dc.get(
                dc_id,
                {},
            )

        count = counts.get(
            metric_name,
            0,
        )

        if count <= 0:
            return float("nan")

        return float(sums[metric_name] / count)

    def record_finalized_trace(
        self,
        finalized_trace: FinalizedJobTrace,
    ) -> None:

        self.terminal_trace_flushed_count += 1

        routing_transition_count = len(finalized_trace.routing_transitions)
        for transition in finalized_trace.routing_transitions:
            if not transition.done:
                self.heuristic_nonterminal_count += 1
                self.heuristic_score_sum += transition.next_heuristic_score
                self.heuristic_valid_score_count += int(transition.heuristic_valid)
            if self.reward_shaping_enabled:
                self.heuristic_reward_sum += (
                    self.routing_gamma
                    * (1 - self.heuristic_beta)
                    * transition.next_heuristic_score
                )

        self.routing_transition_flushed_count += routing_transition_count

        if finalized_trace.host_transition is not None:
            self.host_transition_flushed_count += 1

        edge_hops = sum(
            1
            for routing_step in finalized_trace.routing_steps
            if (routing_step.action_type == "edge_dc")
        )

        self.routing_edge_hop_total += int(edge_hops)
        self.routing_edge_hops_by_job.append(int(edge_hops))

        self.max_routing_hops = max(
            self.max_routing_hops,
            int(edge_hops),
        )

        if edge_hops >= 2:
            self.multi_hop_job_count += 1

        first_edge_seen = False
        routing_steps = tuple(finalized_trace.routing_steps)
        for index, step in enumerate(routing_steps):
            if str(step.action_type) != "edge_dc":
                continue
            if index + 1 >= len(routing_steps):
                raise RuntimeError(
                    f"Edge step has no successor: job={finalized_trace.job_id}, sequence={step.sequence_index}"
                )
            successor = routing_steps[index + 1]
            if str(successor.agent_id) != str(step.target_dc_id):
                raise RuntimeError(
                    f"Edge successor target mismatch: job={finalized_trace.job_id}, "
                    f"expected={step.target_dc_id}, actual={successor.agent_id}"
                )
            successor_type = str(successor.action_type)
            if successor_type == "self":
                host_result = (
                    None
                    if finalized_trace.host_step is None
                    else str(finalized_trace.host_step.execution_result)
                )
                outcome = "self" if host_result in {"started", "queued"} else "drop"
            elif successor_type == "edge_dc":
                outcome = "edge"
            elif successor_type == "cloud":
                outcome = "cloud"
            elif successor_type == "drop":
                outcome = "drop"
            else:
                raise RuntimeError(f"Unknown Edge successor action: {successor_type}")

            self.edge_successor_total += 1
            setattr(
                self,
                f"edge_successor_{outcome}_count",
                int(getattr(self, f"edge_successor_{outcome}_count")) + 1,
            )
            target_dc_id = str(step.target_dc_id)
            self._inc_dc(target_dc_id, "incoming_edge_successor_count")
            self._inc_dc(target_dc_id, f"incoming_edge_{outcome}_count")
            pair = f"{step.source_dc_id}->{target_dc_id}"
            pair_counts = self.absorption_pair_counts.setdefault(
                pair, {name: 0 for name in ("self", "edge", "cloud", "drop")}
            )
            pair_counts[outcome] += 1
            if not first_edge_seen:
                self.first_edge_successor_total += 1
                self.first_edge_absorbed_count += int(outcome == "self")
                first_edge_seen = True

        self.routing_layer_reward_sum += float(
            sum(
                float(transition.reward)
                for transition in finalized_trace.routing_transitions
            )
        )

        if finalized_trace.host_transition is not None:
            self.host_layer_reward_sum += float(finalized_trace.host_transition.reward)


def validate_training_stage_config(
    train_config: TrainConfig,
) -> None:

    host_pretrain = int(train_config.host_pretrain_episodes)

    routing_train = int(train_config.routing_train_episodes)

    joint_finetune = int(train_config.joint_finetune_episodes)

    if (
        min(
            host_pretrain,
            routing_train,
            joint_finetune,
        )
        < 0
    ):
        raise ValueError("三阶段 Episode 数不能为负数。")

    configured_total = host_pretrain + routing_train + joint_finetune

    if configured_total != int(train_config.num_episodes):
        raise ValueError(
            "三阶段 Episode 总数与 "
            "num_episodes 不一致："
            f"stages={configured_total}, "
            f"num_episodes="
            f"{train_config.num_episodes}"
        )


def resolve_training_stage(
    episode: int,
    train_config: TrainConfig,
) -> TrainingStage:

    episode = int(episode)

    host_end = int(train_config.host_pretrain_episodes)

    routing_end = host_end + int(train_config.routing_train_episodes)

    if episode <= host_end:
        return TrainingStage.HOST_PRETRAIN

    if episode <= routing_end:
        return TrainingStage.ROUTING_TRAIN

    return TrainingStage.JOINT_FINETUNE


def training_stage_start_episode(
    stage: TrainingStage,
    train_config: TrainConfig,
) -> int:
    """
    返回某个 Training Stage 的第一个 Episode。

    Episode 使用 1-based 编号：
        Stage 1: 1
        Stage 2: HOST_PRETRAIN_EPISODES + 1
        Stage 3: HOST_PRETRAIN_EPISODES
                 + ROUTING_TRAIN_EPISODES + 1
    """

    host_end = int(train_config.host_pretrain_episodes)

    routing_end = host_end + int(train_config.routing_train_episodes)

    if stage == TrainingStage.HOST_PRETRAIN:
        return 1

    if stage == TrainingStage.ROUTING_TRAIN:
        return host_end + 1

    return routing_end + 1


def training_stage_end_episode(
    stage: TrainingStage,
    train_config: TrainConfig,
) -> int:
    """
    返回某个 Training Stage 的最后一个 Episode。
    """

    host_end = int(train_config.host_pretrain_episodes)

    routing_end = host_end + int(train_config.routing_train_episodes)

    if stage == TrainingStage.HOST_PRETRAIN:
        return host_end

    if stage == TrainingStage.ROUTING_TRAIN:
        return routing_end

    return int(train_config.num_episodes)


def stage_trains_routing(
    stage: TrainingStage,
) -> bool:

    return stage in {
        TrainingStage.ROUTING_TRAIN,
        TrainingStage.JOINT_FINETUNE,
    }


def stage_trains_host(
    stage: TrainingStage,
) -> bool:

    return stage in {
        TrainingStage.HOST_PRETRAIN,
        TrainingStage.JOINT_FINETUNE,
    }


def apply_training_stage_modes(
    stage: TrainingStage,
    routing_masac: RoutingMASAC,
    host_sac_agents: Dict[str, LocalHostSAC],
) -> None:
    """
    根据 Training Stage 明确设置两层网络模式。

    真正是否更新参数仍由 Orchestrator
    是否调用 update() 决定。
    """

    if stage == TrainingStage.HOST_PRETRAIN:

        routing_masac.eval_mode()

        for host_agent in host_sac_agents.values():
            host_agent.train_mode()

        return

    if stage == TrainingStage.ROUTING_TRAIN:

        routing_masac.train_mode()

        for host_agent in host_sac_agents.values():
            host_agent.eval_mode()

        return

    routing_masac.train_mode()

    for host_agent in host_sac_agents.values():
        host_agent.train_mode()


def get_self_routing_action(
    env: CloudEdgeEnv,
    agent_id: str,
) -> int:
    """
    Stage 1 Host Pretrain 时，
    Orchestrator 直接让当前 Job 进入当前 DC Host 层。

    不是 action mask，
    也不是 Environment forced action。
    """

    agent_id = str(agent_id)

    action = env.routing_dc_id_to_action.get(agent_id)

    if action is None:
        raise RuntimeError(
            "找不到当前 DC 对应的 Self " "Routing action：" f"dc={agent_id}"
        )

    decoded = env._decode_action(
        agent_id=agent_id,
        action=int(action),
    )

    if str(decoded["action_type"]) != "self":
        raise RuntimeError(
            "Stage 1 Self bypass "
            "动作编码错误："
            f"dc={agent_id}, "
            f"action={action}"
        )

    return int(action)


def choose_random_host_action(
    action_dim: int,
    rng: np.random.Generator,
) -> int:

    action_dim = int(action_dim)

    if action_dim <= 0:
        raise ValueError("Host action_dim 必须 > 0")

    return int(
        rng.integers(
            low=0,
            high=action_dim,
        )
    )


def _convert_update_block_to_cpu(
    update_infos: list[
        Dict[
            str,
            Union[
                float,
                torch.Tensor,
            ],
        ]
    ],
) -> list[Dict[str, float]]:
    """
    批量把一次连续 update block 的 GPU Tensor
    转换成 CPU float。

    Routing / Host 共用这一“数据转换”逻辑，
    但转换后的训练指标分别进入自己的统计容器。
    """

    if not update_infos:
        return []

    update_rows = []

    for update_info in update_infos:

        reference_value = update_info["critic_loss"]

        reference_device = (
            reference_value.device
            if torch.is_tensor(reference_value)
            else torch.device("cpu")
        )

        metric_tensors = []

        for metric_name in UPDATE_TENSOR_METRIC_NAMES:

            if metric_name in {
                "heuristic_reward_mean",
                "modified_reward_mean",
                "raw_reward_mean",
                "effective_gamma",
            }:
                metric_value = update_info.get(metric_name, float("nan"))
            else:
                metric_value = update_info[metric_name]

            if torch.is_tensor(metric_value):
                metric_tensor = (
                    metric_value.detach()
                    .to(
                        device=reference_device,
                        dtype=torch.float32,
                    )
                    .reshape(())
                )

            else:
                metric_tensor = torch.tensor(
                    float(metric_value),
                    dtype=torch.float32,
                    device=reference_device,
                )

            metric_tensors.append(metric_tensor)

        update_rows.append(
            torch.stack(
                metric_tensors,
                dim=0,
            )
        )

    update_matrix = torch.stack(
        update_rows,
        dim=0,
    )

    update_matrix_cpu = update_matrix.detach().cpu().numpy()

    return [
        {
            metric_name: float(row[metric_index])
            for metric_index, metric_name in enumerate(UPDATE_TENSOR_METRIC_NAMES)
        }
        for row in update_matrix_cpu
    ]


def record_routing_update_block(
    stats: EpisodeStatistics,
    update_infos: list[
        Dict[
            str,
            Union[
                float,
                torch.Tensor,
            ],
        ]
    ],
) -> None:
    """
    只记录 Routing MASAC update 指标。
    """

    stats.pending_routing_update_infos.extend(
        {
            name: value.detach() if torch.is_tensor(value) else value
            for name, value in update_info.items()
        }
        for update_info in update_infos
    )


def record_host_update_block(
    stats: EpisodeStatistics,
    dc_id: str,
    update_infos: list[
        Dict[
            str,
            Union[
                float,
                torch.Tensor,
            ],
        ]
    ],
) -> None:
    """
    只记录指定 DC 的 Local Host SAC update 指标。
    """

    stats.pending_host_update_infos.extend(
        (
            dc_id,
            {
                name: value.detach() if torch.is_tensor(value) else value
                for name, value in update_info.items()
            },
        )
        for update_info in update_infos
    )


def flush_pending_update_metrics(stats: EpisodeStatistics) -> None:
    """Read detached update metrics once per episode, before building logs."""
    for update_info in _convert_update_block_to_cpu(stats.pending_routing_update_infos):
        stats.record_routing_update(update_info)
    stats.pending_routing_update_infos.clear()

    host_updates = stats.pending_host_update_infos
    for (dc_id, _), update_info in zip(
        host_updates,
        _convert_update_block_to_cpu([info for _, info in host_updates]),
    ):
        stats.record_host_update(dc_id=dc_id, update_info=update_info)
    host_updates.clear()


def flush_finalized_trace_to_replay(
    finalized_trace: FinalizedJobTrace,
    routing_replay_buffer: RoutingReplayBuffer,
    host_replay_buffers: Dict[str, HostReplayBuffer],
    stats: EpisodeStatistics,
    neighbor_feedback_store: NeighborHistoricalFeedbackStore,
    collect_neighbor_historical_feedback: bool,
    bayesian_game: Optional[BayesianCongestionGame] = None,
    env: Optional[CloudEdgeEnv] = None,
) -> None:
    """
    把一个已经完整 Finalize 的 Job
    一次性写入两个正式 Replay System。

    顺序：

        FinalizedJobTrace
            ↓
        RoutingTransitions
            ↓
        RoutingReplayBuffer

    如果该 Job 最终 Routing=Self：

        HostTransition
            ↓
        对应 DC 的 HostReplayBuffer

        Cloud / Drop：
        不产生 HostTransition。

    Bayesian Evidence（可选接入）：
        FinalizedJobTrace
            ↓
        BayesianHistoricalEvidence
            ↓
        BayesianBeliefStore

    ``bayesian_game`` 默认保持 None，确保当前 H-MASAC-equivalent
    Zero-Diff 路径不会额外启用 Bayesian 状态更新。
    """

    host_transition = finalized_trace.host_transition

    host_replay_buffer = None

    if host_transition is not None:

        host_dc_id = str(host_transition.dc_id)

        host_replay_buffer = host_replay_buffers.get(host_dc_id)

        if host_replay_buffer is None:
            raise RuntimeError(
                "找不到 HostTransition 对应的 "
                "HostReplayBuffer："
                f"job={finalized_trace.job_id}, "
                f"dc={host_dc_id}"
            )

    bayesian_evidences = tuple()
    if bayesian_game is not None:
        if env is None:
            raise RuntimeError("启用 Bayesian Evidence 转换时必须提供 Environment。")
        bayesian_evidences = build_bayesian_evidence_from_finalized_trace(
            finalized_trace,
            env,
        )

    for routing_transition in finalized_trace.routing_transitions:

        routing_replay_buffer.add(routing_transition)

    if host_transition is not None and host_replay_buffer is not None:
        host_replay_buffer.add(host_transition)

    stats.record_finalized_trace(finalized_trace)
    if collect_neighbor_historical_feedback:
        neighbor_feedback_store.update_from_finalized_trace(finalized_trace)

    if bayesian_game is not None:
        for evidence in bayesian_evidences:
            bayesian_game.update_belief(
                evidence,
                update_clock=float(env.current_time),
            )


def consume_environment_outcome_events(
    env: CloudEdgeEnv,
    pending_trace_store: PendingJobTraceStore,
    routing_replay_buffer: RoutingReplayBuffer,
    host_replay_buffers: Dict[str, HostReplayBuffer],
    stats: EpisodeStatistics,
    neighbor_feedback_store: NeighborHistoricalFeedbackStore,
    collect_neighbor_historical_feedback: bool,
    training_reward_model: HMasacTrainingRewardModel,
    bayesian_game: Optional[BayesianCongestionGame] = None,
) -> None:
    """
    第十九步以后：

        Environment delayed outcome
                ↓
        Pending Job Causal Trace
                ↓
        Job terminal
                ↓
        FinalizedJobTrace
                ↓
        ┌──────────────────────┐
        │                      │
        ▼                      ▼
    RoutingReplay        HostReplay

    Job 没有 terminal 前：
        不允许修改正式 ReplayBuffer。
    """

    outcome_events = env.pop_job_outcome_events()

    if not outcome_events:
        return

    for outcome_event in outcome_events:

        job_id = str(outcome_event["job_id"])

        reason = str(outcome_event["reason"])

        env_time = float(outcome_event["env_time"])

        is_terminal = bool(
            outcome_event.get(
                "terminal",
                False,
            )
        )

        reward_delta = float(
            training_reward_model.calculate_outcome_reward(outcome_event)
        )

        pending_trace_store.record_reward_event(
            job_id=job_id,
            env_time=env_time,
            reward_delta=reward_delta,
            reason=reason,
            terminal=is_terminal,
        )

        if is_terminal:

            finalized_trace = pending_trace_store.finalize_terminal_trace(job_id=job_id)

            correction_agent_id = None

            if finalized_trace.routing_transitions:
                correction_agent_id = str(
                    finalized_trace.routing_transitions[-1].agent_id
                )

            flush_finalized_trace_to_replay(
                finalized_trace=(finalized_trace),
                routing_replay_buffer=(routing_replay_buffer),
                host_replay_buffers=(host_replay_buffers),
                stats=stats,
                neighbor_feedback_store=(neighbor_feedback_store),
                collect_neighbor_historical_feedback=(
                    collect_neighbor_historical_feedback
                ),
                bayesian_game=bayesian_game,
                env=env,
            )

            pending_trace_store.pop_finalized_trace(job_id)

            if correction_agent_id is not None:

                stats.record_delayed_training_reward(
                    agent_id=(correction_agent_id),
                    reward_delta=(reward_delta),
                )

            continue

        pending_trace = pending_trace_store.get_trace(job_id)

        correction_agent_id = None

        for routing_step in reversed(pending_trace.routing_steps):

            if routing_step.action_source != "forced":
                correction_agent_id = str(routing_step.agent_id)
                break

        if correction_agent_id is not None:

            stats.record_delayed_training_reward(
                agent_id=(correction_agent_id),
                reward_delta=(reward_delta),
            )


def set_global_random_seeds(seed: int) -> None:
    """统一设置 Python、NumPy 和 PyTorch 随机种子。"""

    seed = int(seed)

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_environment(
    seed: int,
    old_env_path: Optional[str] = None,
) -> Any:
    if old_env_path is None:
        return CloudEdgeEnv(seed=int(seed))
    return CloudEdgeEnv(
        old_env_path=str(old_env_path),
        seed=int(seed),
    )


def choose_random_routing_action(
    action_dim: int,
    rng: np.random.Generator,
) -> int:

    action_dim = int(action_dim)

    if action_dim <= 0:
        raise ValueError("Routing action_dim 必须大于 0。")

    return int(
        rng.integers(
            low=0,
            high=action_dim,
        )
    )


def count_completed_jobs(env: Any) -> int:
    completed_count = 0
    for datacenter in getattr(env, "datacenters", []):
        for host in getattr(datacenter, "host_list", []):
            completed_queue = getattr(host, "completed_queue", None)

            if completed_queue is None:
                continue

            completed_count += len(completed_queue)
    return int(completed_count)


def calculate_service_metrics(env: Any) -> Dict[str, float]:
    completed_jobs = []
    for job in getattr(env, "jobs", []):
        if getattr(job, "finish_time", None) is not None:
            completed_jobs.append(job)

    total_jobs = int(len(getattr(env, "jobs", [])))

    dropped_jobs = int(
        len(
            getattr(
                env,
                "dropped_jobs_info",
                [],
            )
        )
    )

    total_completion_time = 0.0
    sla_satisfied_jobs = 0
    sla_violated_completed_jobs = 0
    total_violation_degree = 0.0

    sla_ratio = float(env.sla_deadline_ratio)
    drop_ratio = float(env.drop_deadline_ratio)

    for job in completed_jobs:
        turnaround_time = job.get_turnaround_time()
        if turnaround_time is None:
            continue

        turnaround_time = float(turnaround_time)
        duration = max(
            float(job.duration),
            1e-8,
        )
        sla_limit = sla_ratio * duration
        drop_limit = drop_ratio * duration
        total_completion_time += turnaround_time

        if turnaround_time <= sla_limit:
            sla_satisfied_jobs += 1

        else:
            sla_violated_completed_jobs += 1
            violation_degree = (turnaround_time - sla_limit) / max(
                drop_limit - sla_limit,
                1e-8,
            )

            total_violation_degree += float(
                np.clip(
                    violation_degree,
                    0.0,
                    1.0,
                )
            )

    completed_count = len(completed_jobs)

    avg_completion_time = (
        total_completion_time / completed_count if completed_count > 0 else 0.0
    )

    total_violation_degree += float(dropped_jobs)

    sla_satisfaction_rate = sla_satisfied_jobs / total_jobs if total_jobs > 0 else 0.0

    sla_violation_rate = (
        (sla_violated_completed_jobs + dropped_jobs) / total_jobs
        if total_jobs > 0
        else 0.0
    )

    mean_sla_violation_degree = (
        total_violation_degree / total_jobs if total_jobs > 0 else 0.0
    )

    return {
        "sla_satisfied_jobs": int(sla_satisfied_jobs),
        "sla_violated_completed_jobs": int(sla_violated_completed_jobs),
        "sla_satisfaction_rate": float(sla_satisfaction_rate),
        "sla_violation_rate": float(sla_violation_rate),
        "mean_sla_violation_degree": float(mean_sla_violation_degree),
        "total_completion_time": float(total_completion_time),
        "avg_completion_time": float(avg_completion_time),
    }


def count_waiting_jobs(env: Any) -> int:
    waiting_count = 0
    for datacenter in getattr(
        env,
        "datacenters",
        [],
    ):
        for host in getattr(
            datacenter,
            "host_list",
            [],
        ):
            waiting_queue = getattr(
                host,
                "waiting_queue",
                None,
            )
            if waiting_queue is None:
                continue
            waiting_count += len(waiting_queue)
    return int(waiting_count)


def safe_ratio(
    numerator: float,
    denominator: float,
) -> float:
    denominator = float(denominator)
    if denominator <= 0.0:
        return 0.0
    return float(float(numerator) / denominator)


def calculate_resource_timeline_metrics(
    jobs: list,
    episode_duration_s: float,
) -> Dict[str, float]:
    episode_duration_s = max(
        float(episode_duration_s),
        0.0,
    )
    cpu_resource_seconds = 0.0
    gpu_resource_seconds = 0.0
    running_job_seconds = 0.0
    events: Dict[float, list] = {}
    valid_running_jobs = 0

    for job in jobs:
        start_time = getattr(
            job,
            "start_time",
            None,
        )
        finish_time = getattr(
            job,
            "finish_time",
            None,
        )

        if start_time is None or finish_time is None:
            continue

        start_time = float(start_time)
        finish_time = float(finish_time)
        run_time_s = max(
            finish_time - start_time,
            0.0,
        )
        cpu_request = float(
            getattr(
                job,
                "cpu_request",
                0.0,
            )
        )
        gpu_request = float(
            getattr(
                job,
                "gpu_request",
                0.0,
            )
        )

        cpu_resource_seconds += cpu_request * run_time_s
        gpu_resource_seconds += gpu_request * run_time_s
        running_job_seconds += run_time_s
        valid_running_jobs += 1

        if start_time not in events:
            events[start_time] = [
                0.0,
                0.0,
                0,
            ]
        if finish_time not in events:
            events[finish_time] = [
                0.0,
                0.0,
                0,
            ]

        events[start_time][0] += cpu_request
        events[start_time][1] += gpu_request
        events[start_time][2] += 1

        events[finish_time][0] -= cpu_request
        events[finish_time][1] -= gpu_request
        events[finish_time][2] -= 1

    current_used_cpu = 0.0
    current_used_gpu = 0.0
    current_running_jobs = 0

    peak_used_cpu = 0.0
    peak_used_gpu = 0.0
    peak_running_jobs = 0

    busy_time_s = 0.0
    previous_event_time = 0.0

    for event_time in sorted(events.keys()):
        event_time = float(event_time)
        interval_s = max(
            event_time - previous_event_time,
            0.0,
        )

        if current_running_jobs > 0:
            busy_time_s += interval_s

        cpu_delta, gpu_delta, running_delta = events[event_time]

        current_used_cpu += float(cpu_delta)
        current_used_gpu += float(gpu_delta)
        current_running_jobs += int(running_delta)
        current_used_cpu = max(
            current_used_cpu,
            0.0,
        )
        current_used_gpu = max(
            current_used_gpu,
            0.0,
        )
        current_running_jobs = max(
            current_running_jobs,
            0,
        )

        peak_used_cpu = max(
            peak_used_cpu,
            current_used_cpu,
        )
        peak_used_gpu = max(
            peak_used_gpu,
            current_used_gpu,
        )
        peak_running_jobs = max(
            peak_running_jobs,
            current_running_jobs,
        )
        previous_event_time = event_time

    return {
        "valid_running_jobs": int(valid_running_jobs),
        "cpu_resource_seconds": float(cpu_resource_seconds),
        "gpu_resource_seconds": float(gpu_resource_seconds),
        "avg_used_cpu": safe_ratio(
            cpu_resource_seconds,
            episode_duration_s,
        ),
        "avg_used_gpu": safe_ratio(
            gpu_resource_seconds,
            episode_duration_s,
        ),
        "peak_used_cpu": float(peak_used_cpu),
        "peak_used_gpu": float(peak_used_gpu),
        "avg_running_jobs": safe_ratio(
            running_job_seconds,
            episode_duration_s,
        ),
        "peak_running_jobs": int(peak_running_jobs),
        "busy_time_s": float(busy_time_s),
        "busy_ratio": safe_ratio(
            busy_time_s,
            episode_duration_s,
        ),
    }


def calculate_episode_load_metrics(
    env: Any,
) -> Dict[str, Any]:

    episode_duration_s = max(
        float(
            getattr(
                env,
                "current_time",
                0.0,
            )
        ),
        0.0,
    )
    all_jobs = list(
        getattr(
            env,
            "jobs",
            [],
        )
    )
    cloud_id = str(
        getattr(
            env,
            "cloud_id",
            "cloud",
        )
    )

    cpu_requests = np.asarray(
        [
            float(
                getattr(
                    job,
                    "cpu_request",
                    0.0,
                )
            )
            for job in all_jobs
        ],
        dtype=np.float64,
    )
    gpu_requests = np.asarray(
        [
            float(
                getattr(
                    job,
                    "gpu_request",
                    0.0,
                )
            )
            for job in all_jobs
        ],
        dtype=np.float64,
    )
    durations = np.asarray(
        [
            float(
                getattr(
                    job,
                    "duration",
                    0.0,
                )
            )
            for job in all_jobs
        ],
        dtype=np.float64,
    )
    arrival_times = np.asarray(
        [
            float(
                getattr(
                    job,
                    "arrive_time",
                    0.0,
                )
            )
            for job in all_jobs
        ],
        dtype=np.float64,
    )

    edge_dc_ids = [str(dc_id) for dc_id in getattr(env, "edge_dc_ids", [])]
    dc_arrival_mode = str(
        getattr(
            env,
            "dc_arrival_mode",
            "uniform",
        )
    )
    configured_arrival_profile = dict(
        getattr(
            env,
            "dc_arrival_profile",
            {},
        )
    )
    dc_exogenous_arrival_times = {dc_id: [] for dc_id in edge_dc_ids}
    missing_origin_count = 0
    unknown_origin_count = 0

    for job in all_jobs:
        origin_dc_id_raw = getattr(
            job,
            "origin_datacenter",
            None,
        )
        if origin_dc_id_raw is None:
            missing_origin_count += 1
            continue

        origin_dc_id = str(origin_dc_id_raw)
        if origin_dc_id not in dc_exogenous_arrival_times:
            unknown_origin_count += 1
            continue

        dc_exogenous_arrival_times[origin_dc_id].append(
            float(
                getattr(
                    job,
                    "arrive_time",
                    0.0,
                )
            )
        )

    uniform_probability = safe_ratio(
        1.0,
        len(edge_dc_ids),
    )
    dc_exogenous_arrival_details = {}
    for dc_id in edge_dc_ids:
        dc_arrival_times = np.asarray(
            dc_exogenous_arrival_times[dc_id],
            dtype=np.float64,
        )
        dc_arrival_count = int(dc_arrival_times.size)

        if dc_arrival_count >= 2:
            dc_arrival_span_s = float(
                np.max(dc_arrival_times) - np.min(dc_arrival_times)
            )
            dc_observed_arrival_rate = safe_ratio(
                dc_arrival_count - 1,
                dc_arrival_span_s,
            )
        else:
            dc_arrival_span_s = 0.0
            dc_observed_arrival_rate = 0.0

        profile = dict(
            configured_arrival_profile.get(
                dc_id,
                {},
            )
        )
        configured_probability = float(
            profile.get(
                "probability",
                uniform_probability,
            )
        )
        configured_arrival_rate = float(
            profile.get(
                "configured_arrival_rate",
                float(getattr(env, "lambda_rate", 0.0)) * configured_probability,
            )
        )

        dc_exogenous_arrival_details[dc_id] = {
            "arrival_mode": dc_arrival_mode,
            "arrival_role": str(
                profile.get(
                    "role",
                    "uniform",
                )
            ),
            "configured_arrival_weight": float(
                profile.get(
                    "weight",
                    1.0,
                )
            ),
            "configured_arrival_probability": (configured_probability),
            "configured_arrival_rate": (configured_arrival_rate),
            "expected_exogenous_arrival_count": float(
                len(all_jobs) * configured_probability
            ),
            "exogenous_arrival_count": dc_arrival_count,
            "exogenous_arrival_share": safe_ratio(
                dc_arrival_count,
                len(all_jobs),
            ),
            "exogenous_arrival_span_s": dc_arrival_span_s,
            "observed_exogenous_arrival_rate": (dc_observed_arrival_rate),
        }

    edge_jobs = []
    cloud_jobs = []
    edge_total_cpu_capacity = 0.0
    edge_total_gpu_capacity = 0.0
    edge_host_cpu_avg_loads = []
    edge_host_gpu_avg_loads = []
    edge_host_busy_ratios = []
    edge_host_details = {}
    dc_load_details = {}

    for dc in getattr(
        env,
        "datacenters",
        [],
    ):
        dc_id = str(dc.dc_id)
        is_cloud = dc_id == cloud_id
        dc_jobs = []
        dc_cpu_capacity = 0.0
        dc_gpu_capacity = 0.0

        for host_idx, host in enumerate(
            getattr(
                dc,
                "host_list",
                [],
            )
        ):
            completed_queue = getattr(
                host,
                "completed_queue",
                None,
            )
            host_jobs = list(
                getattr(
                    completed_queue,
                    "_queue",
                    [],
                )
            )
            dc_jobs.extend(host_jobs)
            host_timeline = calculate_resource_timeline_metrics(
                jobs=host_jobs,
                episode_duration_s=episode_duration_s,
            )

            cpu_capacity = float(
                getattr(
                    host,
                    "cpu_num",
                    0.0,
                )
            )
            gpu_capacity = float(
                getattr(
                    host,
                    "gpu_capacity_num",
                    0.0,
                )
            )

            if not is_cloud:
                dc_cpu_capacity += cpu_capacity
                dc_gpu_capacity += gpu_capacity
                edge_total_cpu_capacity += cpu_capacity
                edge_total_gpu_capacity += gpu_capacity
                avg_cpu_load = safe_ratio(
                    host_timeline["avg_used_cpu"],
                    cpu_capacity,
                )
                peak_cpu_load = safe_ratio(
                    host_timeline["peak_used_cpu"],
                    cpu_capacity,
                )
                avg_gpu_load = safe_ratio(
                    host_timeline["avg_used_gpu"],
                    gpu_capacity,
                )
                peak_gpu_load = safe_ratio(
                    host_timeline["peak_used_gpu"],
                    gpu_capacity,
                )
                edge_host_cpu_avg_loads.append(avg_cpu_load)

                if gpu_capacity > 0.0:
                    edge_host_gpu_avg_loads.append(avg_gpu_load)

                edge_host_busy_ratios.append(float(host_timeline["busy_ratio"]))
                edge_host_details[f"{dc_id}/{host.host_id}"] = {
                    "cpu_capacity": cpu_capacity,
                    "gpu_capacity": gpu_capacity,
                    "completed_jobs": len(host_jobs),
                    "avg_cpu_load": avg_cpu_load,
                    "peak_cpu_load": peak_cpu_load,
                    "avg_gpu_load": avg_gpu_load,
                    "peak_gpu_load": peak_gpu_load,
                    "avg_running_jobs": host_timeline["avg_running_jobs"],
                    "peak_running_jobs": host_timeline["peak_running_jobs"],
                    "busy_ratio": host_timeline["busy_ratio"],
                }

        dc_timeline = calculate_resource_timeline_metrics(
            jobs=dc_jobs,
            episode_duration_s=episode_duration_s,
        )

        if is_cloud:
            cloud_jobs.extend(dc_jobs)
            dc_load_details[dc_id] = {
                "completed_jobs": len(dc_jobs),
                "avg_used_cpu": dc_timeline["avg_used_cpu"],
                "peak_used_cpu": dc_timeline["peak_used_cpu"],
                "avg_used_gpu": dc_timeline["avg_used_gpu"],
                "peak_used_gpu": dc_timeline["peak_used_gpu"],
                "avg_running_jobs": dc_timeline["avg_running_jobs"],
                "peak_running_jobs": dc_timeline["peak_running_jobs"],
                "busy_ratio": dc_timeline["busy_ratio"],
            }

        else:
            edge_jobs.extend(dc_jobs)
            dc_load_details[dc_id] = {
                "cpu_capacity": dc_cpu_capacity,
                "gpu_capacity": dc_gpu_capacity,
                "completed_jobs": len(dc_jobs),
                "avg_cpu_load": safe_ratio(
                    dc_timeline["avg_used_cpu"],
                    dc_cpu_capacity,
                ),
                "peak_cpu_load": safe_ratio(
                    dc_timeline["peak_used_cpu"],
                    dc_cpu_capacity,
                ),
                "avg_gpu_load": safe_ratio(
                    dc_timeline["avg_used_gpu"],
                    dc_gpu_capacity,
                ),
                "peak_gpu_load": safe_ratio(
                    dc_timeline["peak_used_gpu"],
                    dc_gpu_capacity,
                ),
                "avg_running_jobs": dc_timeline["avg_running_jobs"],
                "peak_running_jobs": dc_timeline["peak_running_jobs"],
                "busy_ratio": dc_timeline["busy_ratio"],
            }
            dc_load_details[dc_id].update(
                dc_exogenous_arrival_details.get(
                    dc_id,
                    {},
                )
            )

    edge_timeline = calculate_resource_timeline_metrics(
        jobs=edge_jobs,
        episode_duration_s=episode_duration_s,
    )
    cloud_timeline = calculate_resource_timeline_metrics(
        jobs=cloud_jobs,
        episode_duration_s=episode_duration_s,
    )
    edge_cpu_host_mean = (
        float(np.mean(edge_host_cpu_avg_loads)) if edge_host_cpu_avg_loads else 0.0
    )
    edge_cpu_host_std = (
        float(np.std(edge_host_cpu_avg_loads)) if edge_host_cpu_avg_loads else 0.0
    )
    edge_cpu_host_p95 = (
        float(
            np.percentile(
                edge_host_cpu_avg_loads,
                95,
            )
        )
        if edge_host_cpu_avg_loads
        else 0.0
    )
    edge_gpu_host_mean = (
        float(np.mean(edge_host_gpu_avg_loads)) if edge_host_gpu_avg_loads else 0.0
    )
    edge_gpu_host_std = (
        float(np.std(edge_host_gpu_avg_loads)) if edge_host_gpu_avg_loads else 0.0
    )

    if len(arrival_times) >= 2:
        arrival_span_s = float(np.max(arrival_times) - np.min(arrival_times))
        observed_arrival_rate = safe_ratio(
            len(arrival_times) - 1,
            arrival_span_s,
        )

    else:
        arrival_span_s = 0.0
        observed_arrival_rate = 0.0

    return {
        "dc_arrival_mode": dc_arrival_mode,
        "configured_total_arrival_rate": float(
            getattr(
                env,
                "lambda_rate",
                0.0,
            )
        ),
        "workload_missing_origin_count": int(missing_origin_count),
        "workload_unknown_origin_count": int(unknown_origin_count),
        "dc_exogenous_arrival_details": json.dumps(
            dc_exogenous_arrival_details,
            ensure_ascii=False,
            sort_keys=True,
        ),
        "workload_total_cpu_request": (
            float(np.sum(cpu_requests)) if cpu_requests.size else 0.0
        ),
        "workload_mean_cpu_request": (
            float(np.mean(cpu_requests)) if cpu_requests.size else 0.0
        ),
        "workload_max_cpu_request": (
            float(np.max(cpu_requests)) if cpu_requests.size else 0.0
        ),
        "workload_total_gpu_request": (
            float(np.sum(gpu_requests)) if gpu_requests.size else 0.0
        ),
        "workload_mean_gpu_request": (
            float(np.mean(gpu_requests)) if gpu_requests.size else 0.0
        ),
        "workload_max_gpu_request": (
            float(np.max(gpu_requests)) if gpu_requests.size else 0.0
        ),
        "workload_gpu_job_ratio": (
            float(np.mean(gpu_requests > 0.0)) if gpu_requests.size else 0.0
        ),
        "workload_total_duration_s": (
            float(np.sum(durations)) if durations.size else 0.0
        ),
        "workload_mean_duration_s": (
            float(np.mean(durations)) if durations.size else 0.0
        ),
        "workload_p95_duration_s": (
            float(
                np.percentile(
                    durations,
                    95,
                )
            )
            if durations.size
            else 0.0
        ),
        "workload_max_duration_s": float(np.max(durations)) if durations.size else 0.0,
        "workload_arrival_span_s": arrival_span_s,
        "workload_observed_arrival_rate": observed_arrival_rate,
        "edge_completed_jobs": int(len(edge_jobs)),
        "edge_total_cpu_capacity": float(edge_total_cpu_capacity),
        "edge_total_gpu_capacity": float(edge_total_gpu_capacity),
        "edge_cpu_resource_seconds": float(edge_timeline["cpu_resource_seconds"]),
        "edge_gpu_resource_seconds": float(edge_timeline["gpu_resource_seconds"]),
        "edge_avg_cpu_load": safe_ratio(
            edge_timeline["avg_used_cpu"],
            edge_total_cpu_capacity,
        ),
        "edge_peak_cpu_load": safe_ratio(
            edge_timeline["peak_used_cpu"],
            edge_total_cpu_capacity,
        ),
        "edge_avg_gpu_load": safe_ratio(
            edge_timeline["avg_used_gpu"],
            edge_total_gpu_capacity,
        ),
        "edge_peak_gpu_load": safe_ratio(
            edge_timeline["peak_used_gpu"],
            edge_total_gpu_capacity,
        ),
        "edge_avg_running_jobs": float(edge_timeline["avg_running_jobs"]),
        "edge_peak_running_jobs": int(edge_timeline["peak_running_jobs"]),
        "edge_host_avg_cpu_load_mean": edge_cpu_host_mean,
        "edge_host_avg_cpu_load_std": edge_cpu_host_std,
        "edge_host_avg_cpu_load_p95": edge_cpu_host_p95,
        "edge_host_avg_gpu_load_mean": edge_gpu_host_mean,
        "edge_host_avg_gpu_load_std": edge_gpu_host_std,
        "edge_host_busy_ratio_mean": (
            float(np.mean(edge_host_busy_ratios)) if edge_host_busy_ratios else 0.0
        ),
        "edge_host_busy_ratio_max": (
            float(np.max(edge_host_busy_ratios)) if edge_host_busy_ratios else 0.0
        ),
        "cloud_completed_jobs": int(len(cloud_jobs)),
        "cloud_cpu_resource_seconds": float(cloud_timeline["cpu_resource_seconds"]),
        "cloud_gpu_resource_seconds": float(cloud_timeline["gpu_resource_seconds"]),
        "cloud_avg_used_cpu": float(cloud_timeline["avg_used_cpu"]),
        "cloud_peak_used_cpu": float(cloud_timeline["peak_used_cpu"]),
        "cloud_avg_used_gpu": float(cloud_timeline["avg_used_gpu"]),
        "cloud_peak_used_gpu": float(cloud_timeline["peak_used_gpu"]),
        "cloud_avg_running_jobs": float(cloud_timeline["avg_running_jobs"]),
        "cloud_peak_running_jobs": int(cloud_timeline["peak_running_jobs"]),
        "cloud_busy_ratio": float(cloud_timeline["busy_ratio"]),
        "dc_load_details": json.dumps(
            dc_load_details,
            ensure_ascii=False,
            sort_keys=True,
        ),
        "edge_host_load_details": json.dumps(
            edge_host_details,
            ensure_ascii=False,
            sort_keys=True,
        ),
    }


def calculate_episode_energy_metrics(
    env: Any,
) -> Dict[str, float]:

    simulation_end_time_s = float(
        getattr(
            env,
            "current_time",
            0.0,
        )
    )

    energy_end_time_s = float(
        getattr(
            env,
            "last_energy_update_time",
            simulation_end_time_s,
        )
    )

    energy_time_s = max(
        energy_end_time_s,
        0.0,
    )

    edge_idle_energy_j = float(
        getattr(
            env,
            "edge_idle_energy_j",
            0.0,
        )
    )

    edge_cpu_dynamic_energy_j = float(
        getattr(
            env,
            "edge_cpu_dynamic_energy_j",
            0.0,
        )
    )

    edge_gpu_dynamic_energy_j = float(
        getattr(
            env,
            "edge_gpu_dynamic_energy_j",
            0.0,
        )
    )

    cloud_compute_energy_j = float(
        getattr(
            env,
            "cloud_compute_energy_j",
            0.0,
        )
    )

    transfer_energy_j = float(
        getattr(
            env,
            "transfer_energy_j",
            0.0,
        )
    )

    edge_edge_transfer_energy_j = float(
        getattr(
            env,
            "edge_edge_transfer_energy_j",
            0.0,
        )
    )

    edge_cloud_transfer_energy_j = float(
        getattr(
            env,
            "edge_cloud_transfer_energy_j",
            0.0,
        )
    )

    edge_total_energy_j = (
        edge_idle_energy_j + edge_cpu_dynamic_energy_j + edge_gpu_dynamic_energy_j
    )

    system_compute_energy_j = edge_total_energy_j + cloud_compute_energy_j

    system_dynamic_compute_energy_j = (
        edge_cpu_dynamic_energy_j + edge_gpu_dynamic_energy_j + cloud_compute_energy_j
    )

    total_system_energy_j = system_compute_energy_j + transfer_energy_j

    all_jobs = list(
        getattr(
            env,
            "jobs",
            [],
        )
    )

    task_compute_energy_j = sum(
        float(
            getattr(
                job,
                "compute_energy_j",
                0.0,
            )
        )
        for job in all_jobs
    )

    task_transfer_energy_j = sum(
        float(
            getattr(
                job,
                "transfer_energy_j",
                0.0,
            )
        )
        for job in all_jobs
    )

    task_edge_edge_transfer_energy_j = sum(
        float(
            getattr(
                job,
                "edge_edge_transfer_energy_j",
                0.0,
            )
        )
        for job in all_jobs
    )

    task_edge_cloud_transfer_energy_j = sum(
        float(
            getattr(
                job,
                "edge_cloud_transfer_energy_j",
                0.0,
            )
        )
        for job in all_jobs
    )

    task_attributable_energy_j = task_compute_energy_j + task_transfer_energy_j

    total_jobs = len(all_jobs)

    completed_jobs = count_completed_jobs(env)

    transfer_split_gap_j = (
        transfer_energy_j - edge_edge_transfer_energy_j - edge_cloud_transfer_energy_j
    )

    transfer_job_accounting_gap_j = transfer_energy_j - task_transfer_energy_j

    energy_time_gap_s = simulation_end_time_s - energy_end_time_s

    return {
        "episode_energy_time_s": energy_time_s,
        "energy_time_gap_s": float(energy_time_gap_s),
        "edge_idle_energy_j": edge_idle_energy_j,
        "edge_cpu_dynamic_energy_j": edge_cpu_dynamic_energy_j,
        "edge_gpu_dynamic_energy_j": edge_gpu_dynamic_energy_j,
        "edge_total_energy_j": edge_total_energy_j,
        "cloud_compute_energy_j": cloud_compute_energy_j,
        "system_compute_energy_j": system_compute_energy_j,
        "system_dynamic_compute_energy_j": system_dynamic_compute_energy_j,
        "transfer_energy_j": transfer_energy_j,
        "edge_edge_transfer_energy_j": edge_edge_transfer_energy_j,
        "edge_cloud_transfer_energy_j": edge_cloud_transfer_energy_j,
        "total_system_energy_j": total_system_energy_j,
        "total_system_energy_kwh": total_system_energy_j / 3_600_000.0,
        "edge_idle_avg_power_w": safe_ratio(
            edge_idle_energy_j,
            energy_time_s,
        ),
        "edge_cpu_dynamic_avg_power_w": safe_ratio(
            edge_cpu_dynamic_energy_j,
            energy_time_s,
        ),
        "edge_gpu_dynamic_avg_power_w": safe_ratio(
            edge_gpu_dynamic_energy_j,
            energy_time_s,
        ),
        "edge_total_avg_power_w": safe_ratio(
            edge_total_energy_j,
            energy_time_s,
        ),
        "cloud_compute_avg_power_w": safe_ratio(
            cloud_compute_energy_j,
            energy_time_s,
        ),
        "system_compute_avg_power_w": safe_ratio(
            system_compute_energy_j,
            energy_time_s,
        ),
        "transfer_equivalent_avg_power_w": safe_ratio(
            transfer_energy_j,
            energy_time_s,
        ),
        "system_total_equivalent_avg_power_w": safe_ratio(
            total_system_energy_j,
            energy_time_s,
        ),
        "edge_idle_energy_share": safe_ratio(
            edge_idle_energy_j,
            total_system_energy_j,
        ),
        "edge_cpu_dynamic_energy_share": safe_ratio(
            edge_cpu_dynamic_energy_j,
            total_system_energy_j,
        ),
        "edge_gpu_dynamic_energy_share": safe_ratio(
            edge_gpu_dynamic_energy_j,
            total_system_energy_j,
        ),
        "cloud_compute_energy_share": safe_ratio(
            cloud_compute_energy_j,
            total_system_energy_j,
        ),
        "transfer_energy_share": safe_ratio(
            transfer_energy_j,
            total_system_energy_j,
        ),
        "system_energy_per_completed_job_j": safe_ratio(
            total_system_energy_j,
            completed_jobs,
        ),
        "system_energy_per_total_job_j": safe_ratio(
            total_system_energy_j,
            total_jobs,
        ),
        "task_compute_energy_j": float(task_compute_energy_j),
        "task_transfer_energy_j": float(task_transfer_energy_j),
        "task_attributable_energy_j": float(task_attributable_energy_j),
        "task_compute_energy_per_completed_job_j": safe_ratio(
            task_compute_energy_j,
            completed_jobs,
        ),
        "task_attributable_energy_per_total_job_j": safe_ratio(
            task_attributable_energy_j,
            total_jobs,
        ),
        "transfer_split_gap_j": float(transfer_split_gap_j),
        "transfer_job_accounting_gap_j": float(transfer_job_accounting_gap_j),
        "task_edge_edge_transfer_energy_j": float(task_edge_edge_transfer_energy_j),
        "task_edge_cloud_transfer_energy_j": float(task_edge_cloud_transfer_energy_j),
    }


def build_checkpoint_structure_metadata(
    env: CloudEdgeEnv,
    routing_masac: RoutingMASAC,
    host_sac_agents: Dict[
        str,
        LocalHostSAC,
    ],
) -> Dict[str, Any]:
    """
    构造 checkpoint 的结构身份信息。

    这些信息不是训练指标，而是判断：
        “当前运行环境是否仍然与该 checkpoint 兼容”

    的硬结构约束。

    特别需要保护：
        1. Cloud ON/OFF；
        2. Edge DC 数量及顺序；
        3. Routing action 语义及维度；
        4. Routing Observation / Global State 维度；
        5. 每个 DC 的 Host 数量及顺序；
        6. 每个 Host SAC 的 observation/action dimension。
    """

    edge_dc_ids = [str(dc_id) for dc_id in env.edge_dc_ids]

    base_dc_map = {str(dc.dc_id): dc for dc in env.base_datacenters}

    host_ids_per_dc: Dict[
        str,
        list,
    ] = {}

    host_count_per_dc: Dict[
        str,
        int,
    ] = {}

    for dc_id in edge_dc_ids:
        dc = base_dc_map.get(dc_id)

        if dc is None:
            raise RuntimeError(
                "构造 checkpoint metadata 时 " "找不到 Edge DC：" f"{dc_id}"
            )

        host_ids = [str(host.host_id) for host in dc.host_list]

        host_ids_per_dc[dc_id] = host_ids

        host_count_per_dc[dc_id] = len(host_ids)

    host_model_metadata = {
        str(dc_id): {
            "obs_dim": int(host_agent.obs_dim),
            "action_dim": int(host_agent.action_dim),
        }
        for dc_id, host_agent in host_sac_agents.items()
    }

    return {
        "cloud_enabled": bool(env.enable_cloud_action),
        "edge_dc_ids": edge_dc_ids,
        "routing_action_target_dc_ids": [
            str(dc_id) for dc_id in env.routing_action_target_dc_ids
        ],
        "host_count_per_dc": host_count_per_dc,
        "host_ids_per_dc": host_ids_per_dc,
        "host_capacities_per_dc": {
            dc_id: [
                [float(host.cpu_num), float(host.gpu_capacity_num)]
                for host in base_dc_map[dc_id].host_list
            ]
            for dc_id in edge_dc_ids
        },
        "routing_links": [
            [source, target, float(env.base_graph[source][target]["weight"])]
            for source in edge_dc_ids
            for target in env.routing_action_target_dc_ids
            if target != source
        ],
        "routing": {
            "local_obs_dim": int(routing_masac.local_obs_dim),
            "global_state_dim": int(routing_masac.global_state_dim),
            "action_dim": int(routing_masac.action_dim),
            "num_agents": int(routing_masac.num_agents),
        },
        "hosts": host_model_metadata,
    }


def validate_checkpoint_structure_metadata(
    checkpoint_metadata: Dict[str, Any],
    env: CloudEdgeEnv,
    routing_masac: RoutingMASAC,
    host_sac_agents: Dict[str, LocalHostSAC],
) -> None:
    """
    在真正加载任何网络参数以前，
    检查 checkpoint 与当前双层调度结构是否兼容。

    结构不一致时立即 fail-fast，
    禁止把语义不同的权重强行加载进当前模型。
    """

    schema_version = int(
        checkpoint_metadata.get(
            "schema_version",
            -1,
        )
    )

    if schema_version != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(
            "Checkpoint schema version 不兼容："
            f"saved={schema_version}, "
            f"current="
            f"{CHECKPOINT_SCHEMA_VERSION}"
        )

    architecture = str(
        checkpoint_metadata.get(
            "architecture",
            "",
        )
    )

    if architecture != CHECKPOINT_ARCHITECTURE:
        raise RuntimeError(
            "Checkpoint architecture 不兼容："
            f"saved={architecture!r}, "
            f"current="
            f"{CHECKPOINT_ARCHITECTURE!r}"
        )

    saved_structure = checkpoint_metadata.get("structure", {})

    current_structure = build_checkpoint_structure_metadata(
        env=env,
        routing_masac=(routing_masac),
        host_sac_agents=(host_sac_agents),
    )

    saved_cloud_enabled = bool(
        saved_structure.get(
            "cloud_enabled",
            False,
        )
    )

    current_cloud_enabled = bool(current_structure["cloud_enabled"])

    if saved_cloud_enabled != current_cloud_enabled:
        raise RuntimeError(
            "Checkpoint Cloud 配置不兼容："
            f"saved={saved_cloud_enabled}, "
            f"current={current_cloud_enabled}"
        )

    saved_edge_dc_ids = list(
        saved_structure.get(
            "edge_dc_ids",
            [],
        )
    )

    current_edge_dc_ids = list(current_structure["edge_dc_ids"])

    if saved_edge_dc_ids != current_edge_dc_ids:
        raise RuntimeError(
            "Checkpoint Edge DC 列表或顺序不兼容："
            f"saved={saved_edge_dc_ids}, "
            f"current={current_edge_dc_ids}"
        )

    saved_targets = list(
        saved_structure.get(
            "routing_action_target_dc_ids",
            [],
        )
    )

    current_targets = list(current_structure["routing_action_target_dc_ids"])

    if saved_targets != current_targets:
        raise RuntimeError(
            "Checkpoint Routing action mapping 不兼容："
            f"saved={saved_targets}, "
            f"current={current_targets}"
        )

    saved_routing = saved_structure.get("routing", {})

    current_routing = current_structure["routing"]

    for field_name in (
        "local_obs_dim",
        "global_state_dim",
        "action_dim",
        "num_agents",
    ):
        if int(
            saved_routing.get(
                field_name,
                -1,
            )
        ) != int(current_routing[field_name]):
            raise RuntimeError(
                "Checkpoint Routing 结构不兼容："
                f"field={field_name}, "
                f"saved="
                f"{saved_routing.get(field_name)}, "
                f"current="
                f"{current_routing[field_name]}"
            )

    saved_host_ids = saved_structure.get("host_ids_per_dc", {})

    current_host_ids = current_structure["host_ids_per_dc"]

    if saved_host_ids != current_host_ids:
        raise RuntimeError("Checkpoint Host ID / action mapping " "与当前环境不兼容。")

    saved_hosts = saved_structure.get("hosts", {})

    current_hosts = current_structure["hosts"]

    if set(saved_hosts.keys()) != set(current_hosts.keys()):
        raise RuntimeError(
            "Checkpoint Host SAC DC 集合不兼容："
            f"saved={sorted(saved_hosts.keys())}, "
            f"current={sorted(current_hosts.keys())}"
        )

    for dc_id in current_hosts.keys():
        for field_name in (
            "obs_dim",
            "action_dim",
        ):
            if int(
                saved_hosts[dc_id].get(
                    field_name,
                    -1,
                )
            ) != int(current_hosts[dc_id][field_name]):
                raise RuntimeError(
                    "Checkpoint Local Host SAC "
                    "结构不兼容："
                    f"dc={dc_id}, "
                    f"field={field_name}, "
                    f"saved="
                    f"{saved_hosts[dc_id].get(field_name)}, "
                    f"current="
                    f"{current_hosts[dc_id][field_name]}"
                )


def checkpoint_state_path(model_path: Path) -> Path:
    return model_path.with_suffix(".trainer.json")


def host_checkpoint_dir(
    routing_model_path: Path,
) -> Path:

    return routing_model_path.with_name(f"{routing_model_path.stem}" "_hosts")


def save_two_layer_checkpoint(
    env: CloudEdgeEnv,
    routing_masac: RoutingMASAC,
    host_sac_agents: Dict[str, LocalHostSAC],
    model_path: Path,
    training_stage: TrainingStage,
    train_config: TrainConfig,
    next_episode: int,
    global_decision_steps: int,
    routing_normal_action_steps: int,
    host_training_action_steps: Dict[str, int],
    best_episode_return: float,
) -> None:

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    routing_masac.save(model_path)

    host_dir = host_checkpoint_dir(model_path)

    host_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for dc_id, host_agent in host_sac_agents.items():
        host_agent.save(host_dir / f"{dc_id}.pt")

    checkpoint_metadata = {
        "schema_version": int(CHECKPOINT_SCHEMA_VERSION),
        "architecture": CHECKPOINT_ARCHITECTURE,
        "bcgh2_guidance": bcgh_guidance_metadata(train_config),
        "saved_training_stage": str(training_stage.value),
        "structure": build_checkpoint_structure_metadata(
            env=env,
            routing_masac=(routing_masac),
            host_sac_agents=(host_sac_agents),
        ),
        "training_schedule": {
            "num_episodes": int(train_config.num_episodes),
            "host_pretrain_episodes": int(train_config.host_pretrain_episodes),
            "routing_train_episodes": int(train_config.routing_train_episodes),
            "joint_finetune_episodes": int(train_config.joint_finetune_episodes),
        },
        "trainer_state": {
            "next_episode": int(next_episode),
            "global_decision_steps": int(global_decision_steps),
            "routing_normal_action_steps": int(routing_normal_action_steps),
            "routing_global_steps": int(routing_normal_action_steps),
            "host_training_action_steps": {
                str(dc_id): int(step_count)
                for dc_id, step_count in host_training_action_steps.items()
            },
            "host_global_steps": int(
                sum(
                    int(step_count)
                    for step_count in host_training_action_steps.values()
                )
            ),
            "best_episode_return": float(best_episode_return),
        },
    }

    state_path = checkpoint_state_path(model_path)

    atomic_checkpoint_text_write(
        json.dumps(
            checkpoint_metadata,
            ensure_ascii=False,
            indent=2,
        ),
        state_path,
    )


def load_two_layer_checkpoint_if_needed(
    env: CloudEdgeEnv,
    train_config: TrainConfig,
    routing_masac: RoutingMASAC,
    host_sac_agents: Dict[
        str,
        LocalHostSAC,
    ],
    resume_checkpoint: Optional[str],
) -> Tuple[
    int,
    int,
    int,
    Dict[str, int],
    float,
]:
    """
    恢复完整 Two-Level Scheduler checkpoint。

    新的 Resume 顺序严格为：

        1. 检查 Routing checkpoint 是否存在；
        2. 检查 Host checkpoint 目录是否存在；
        3. 检查每个 DC 的 Host checkpoint 是否完整；
        4. 检查 trainer metadata 是否存在；
        5. 读取 checkpoint metadata；
        6. 检查 checkpoint schema / architecture；
        7. 检查当前 Environment 与 checkpoint 的结构兼容性；
        8. 检查 Trainer State；
        9. 所有检查通过后，才真正加载 Routing MASAC；
       10. 加载每个 DC 的 Local Host SAC；
       11. 恢复训练计数。

    这样可以避免：

        - Cloud ON/OFF 不一致；
        - Edge DC 数量或顺序变化；
        - Routing action mapping 变化；
        - Observation / Global State 维度变化；
        - Host 数量或顺序变化；
        - Host SAC action_dim 变化；
        - Trainer metadata 丢失；

    时仍然静默恢复旧 checkpoint。

    返回：
        start_episode
        global_decision_steps
        routing_normal_action_steps
        host_training_action_steps
        best_episode_return
    """

    default_host_steps = {str(dc_id): 0 for dc_id in host_sac_agents.keys()}

    if resume_checkpoint is None:
        return (
            1,  # start_episode
            0,  # global_decision_steps
            0,  # routing_normal_action_steps
            default_host_steps,
            float("-inf"),  # best_episode_return
        )

    model_path = Path(resume_checkpoint)

    if not model_path.exists():
        raise FileNotFoundError("找不到 Routing MASAC checkpoint：" f"{model_path}")

    if not model_path.is_file():
        raise FileNotFoundError(
            "Routing MASAC checkpoint 不是有效文件：" f"{model_path}"
        )

    host_dir = host_checkpoint_dir(model_path)

    if not host_dir.exists():
        raise FileNotFoundError(
            "恢复 Two-Level checkpoint 时找不到 "
            "Local Host SAC checkpoint 目录："
            f"{host_dir}"
        )

    if not host_dir.is_dir():
        raise FileNotFoundError(
            "Local Host SAC checkpoint 路径不是目录：" f"{host_dir}"
        )

    expected_host_paths: Dict[
        str,
        Path,
    ] = {}

    for dc_id in host_sac_agents.keys():

        dc_id = str(dc_id)

        host_path = host_dir / f"{dc_id}.pt"

        expected_host_paths[dc_id] = host_path

        if not host_path.exists():
            raise FileNotFoundError(
                "缺少 Local Host SAC checkpoint：" f"dc={dc_id}, " f"path={host_path}"
            )

        if not host_path.is_file():
            raise FileNotFoundError(
                "Local Host SAC checkpoint "
                "不是有效文件："
                f"dc={dc_id}, "
                f"path={host_path}"
            )

    state_path = checkpoint_state_path(model_path)

    if not state_path.exists():
        raise FileNotFoundError(
            "Two-Level checkpoint 缺少必要的 " "trainer metadata：" f"{state_path}"
        )

    if not state_path.is_file():
        raise FileNotFoundError("Trainer metadata 不是有效文件：" f"{state_path}")

    try:
        checkpoint_metadata = json.loads(state_path.read_text(encoding="utf-8"))
    except (
        json.JSONDecodeError,
        OSError,
    ) as exc:
        raise RuntimeError(
            "无法读取 Two-Level checkpoint " "trainer metadata：" f"{state_path}"
        ) from exc

    if not isinstance(
        checkpoint_metadata,
        dict,
    ):
        raise RuntimeError(
            "Two-Level checkpoint metadata " "根节点必须是 dict：" f"{state_path}"
        )

    if checkpoint_metadata.get("bcgh2_guidance") != bcgh_guidance_metadata(
        train_config
    ):
        raise RuntimeError(
            "Checkpoint BCGH2 scoring/training semantics differ from this run"
        )

    validate_checkpoint_structure_metadata(
        checkpoint_metadata=(checkpoint_metadata),
        env=env,
        routing_masac=(routing_masac),
        host_sac_agents=(host_sac_agents),
    )

    saved_training_stage_raw = checkpoint_metadata.get("saved_training_stage")

    if saved_training_stage_raw is None:
        raise RuntimeError("Checkpoint metadata 缺少 " "saved_training_stage。")

    try:
        saved_training_stage = TrainingStage(str(saved_training_stage_raw))
    except ValueError as exc:
        raise RuntimeError(
            "Checkpoint 中存在未知 Training Stage：" f"{saved_training_stage_raw!r}"
        ) from exc

    saved_training_schedule = checkpoint_metadata.get("training_schedule")

    if not isinstance(
        saved_training_schedule,
        dict,
    ):
        raise RuntimeError("Checkpoint metadata 缺少有效的 " "training_schedule。")

    current_training_schedule = {
        "num_episodes": int(train_config.num_episodes),
        "host_pretrain_episodes": int(train_config.host_pretrain_episodes),
        "routing_train_episodes": int(train_config.routing_train_episodes),
        "joint_finetune_episodes": int(train_config.joint_finetune_episodes),
    }

    saved_schedule_normalized = {
        "num_episodes": int(
            saved_training_schedule.get(
                "num_episodes",
                -1,
            )
        ),
        "host_pretrain_episodes": int(
            saved_training_schedule.get(
                "host_pretrain_episodes",
                -1,
            )
        ),
        "routing_train_episodes": int(
            saved_training_schedule.get(
                "routing_train_episodes",
                -1,
            )
        ),
        "joint_finetune_episodes": int(
            saved_training_schedule.get(
                "joint_finetune_episodes",
                -1,
            )
        ),
    }

    if saved_schedule_normalized != current_training_schedule:
        print(
            "\n"
            "============================================================\n"
            "⚠️ Checkpoint Training Schedule 与当前配置不同\n"
            f"Saved   : {saved_schedule_normalized}\n"
            f"Current : {current_training_schedule}\n"
            "\n"
            "模型结构兼容，因此允许继续恢复；\n"
            "但请确认这是有意修改三阶段 Episode 配置。\n"
            "============================================================\n",
            flush=True,
        )

    trainer_state = checkpoint_metadata.get("trainer_state")

    if not isinstance(
        trainer_state,
        dict,
    ):
        raise RuntimeError("Checkpoint metadata 缺少有效的 " "trainer_state。")

    required_trainer_fields = {
        "next_episode",
        "global_decision_steps",
        "routing_normal_action_steps",
        "host_training_action_steps",
        "best_episode_return",
    }

    missing_trainer_fields = required_trainer_fields - set(trainer_state.keys())

    if missing_trainer_fields:
        raise RuntimeError(
            "Checkpoint trainer_state 缺少必要字段："
            f"{sorted(missing_trainer_fields)}"
        )

    start_episode = int(trainer_state["next_episode"])

    global_decision_steps = int(trainer_state["global_decision_steps"])

    routing_normal_action_steps = int(trainer_state["routing_normal_action_steps"])

    best_episode_return = float(trainer_state["best_episode_return"])

    if start_episode < 1:
        raise RuntimeError("Checkpoint next_episode 非法：" f"{start_episode}")

    if start_episode > int(train_config.num_episodes) + 1:
        raise RuntimeError(
            "Checkpoint next_episode 超出当前训练范围："
            f"next_episode={start_episode}, "
            f"num_episodes="
            f"{train_config.num_episodes}"
        )

    if global_decision_steps < 0:
        raise RuntimeError(
            "Checkpoint global_decision_steps "
            "不能为负数："
            f"{global_decision_steps}"
        )

    if routing_normal_action_steps < 0:
        raise RuntimeError(
            "Checkpoint routing_normal_action_steps "
            "不能为负数："
            f"{routing_normal_action_steps}"
        )

    saved_host_steps = trainer_state["host_training_action_steps"]

    if not isinstance(
        saved_host_steps,
        dict,
    ):
        raise RuntimeError("Checkpoint " "host_training_action_steps " "必须是 dict。")

    expected_host_dc_ids = {str(dc_id) for dc_id in host_sac_agents.keys()}

    saved_host_dc_ids = {str(dc_id) for dc_id in saved_host_steps.keys()}

    if saved_host_dc_ids != expected_host_dc_ids:
        raise RuntimeError(
            "Checkpoint Host training counter "
            "的 DC 集合与当前环境不一致："
            f"saved="
            f"{sorted(saved_host_dc_ids)}, "
            f"current="
            f"{sorted(expected_host_dc_ids)}"
        )

    host_training_action_steps: Dict[
        str,
        int,
    ] = {}

    for dc_id in sorted(expected_host_dc_ids):

        step_count = int(saved_host_steps[dc_id])

        if step_count < 0:
            raise RuntimeError(
                "Checkpoint Host training step "
                "不能为负数："
                f"dc={dc_id}, "
                f"steps={step_count}"
            )

        host_training_action_steps[dc_id] = step_count

    routing_masac.load(
        file_path=(model_path),
        load_optimizers=True,
    )

    for dc_id, host_agent in host_sac_agents.items():

        dc_id = str(dc_id)

        host_path = expected_host_paths[dc_id]

        host_agent.load(
            host_path,
            load_optimizers=True,
        )

    print(
        "\n"
        "============================================================\n"
        "✅ Two-Level BCGH2-MASCA checkpoint 恢复成功\n"
        f"Routing checkpoint : {model_path}\n"
        f"Host directory     : {host_dir}\n"
        f"Trainer metadata   : {state_path}\n"
        f"Saved stage        : {saved_training_stage.value}\n"
        f"Next episode       : {start_episode}\n"
        f"Global decisions   : {global_decision_steps}\n"
        f"Routing steps      : {routing_normal_action_steps}\n"
        f"Host steps         : {host_training_action_steps}\n"
        f"Best return        : {best_episode_return}\n"
        "============================================================\n",
        flush=True,
    )

    return (
        start_episode,
        global_decision_steps,
        routing_normal_action_steps,
        host_training_action_steps,
        best_episode_return,
    )


def build_timestamped_log_path(
    base_log_path: str,
    run_start_time: str,
) -> Path:
    """
    根据统一 run timestamp 生成一份日志路径。
    """

    base_path = Path(base_log_path)

    if not base_path.is_absolute():

        base_path = PROJECT_ROOT / base_path

    base_path = base_path.resolve()

    log_file_name = f"{base_path.stem}_" f"{run_start_time}" f"{base_path.suffix}"

    return base_path.parent / log_file_name


def build_run_log_paths(
    episode_base_log_path: str,
    dc_base_log_path: str,
) -> Tuple[Path, Path]:
    """
    为同一次训练生成 Episode Log 和 DC Log。

    两个 CSV 必须共用完全相同的 timestamp，
    才能保证离线分析时可以一一匹配。
    """

    run_start_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    episode_log_path = build_timestamped_log_path(
        base_log_path=(episode_base_log_path),
        run_start_time=(run_start_time),
    )

    dc_log_path = build_timestamped_log_path(
        base_log_path=(dc_base_log_path),
        run_start_time=(run_start_time),
    )

    return (
        episode_log_path,
        dc_log_path,
    )


def append_csv_log(
    csv_path: Path,
    row: Dict[str, Any],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    with csv_path.open(
        mode="a",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(row.keys()),
        )

        if write_header:
            writer.writeheader()

        writer.writerow(row)
        csv_file.flush()
        os.fsync(csv_file.fileno())


def build_episode_log_row(
    stats: EpisodeStatistics,
    env: Any,
    routing_replay_buffer: RoutingReplayBuffer,
    host_replay_buffers: Dict[str, HostReplayBuffer],
    routing_masac: RoutingMASAC,
    host_sac_agents: Dict[str, LocalHostSAC],
    host_training_action_steps: Dict[str, int],
    pending_trace_store: PendingJobTraceStore,
    neighbor_feedback_store: NeighborHistoricalFeedbackStore,
    global_decision_steps: int,
    routing_normal_action_steps: int,
    wall_time_seconds: float,
    service_metrics: Dict[str, Any],
    energy_metrics: Dict[str, Any],
    load_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """
    构造 BCGH2-MASCA Episode-level CSV。

    Episode Log 只保存：
        - System outcome
        - Workload / Energy / aggregate load
        - Routing aggregate
        - Host aggregate
        - Causal Trace health

    每个 DC / Host 的细节移动到 dc_log.csv。
    """

    total_jobs = int(
        len(
            getattr(
                env,
                "jobs",
                [],
            )
        )
    )

    completed_jobs = int(count_completed_jobs(env))

    dropped_jobs = int(
        len(
            getattr(
                env,
                "dropped_jobs_info",
                [],
            )
        )
    )

    unresolved_jobs = int(total_jobs - completed_jobs - dropped_jobs)

    queued_jobs = int(
        getattr(
            env,
            "queued_jobs",
            0,
        )
    )

    started_from_waiting_jobs = int(
        getattr(
            env,
            "started_from_waiting_jobs",
            0,
        )
    )

    waiting_timeout_drops = int(
        getattr(
            env,
            "waiting_timeout_drops",
            0,
        )
    )

    max_waiting_queue_length = int(
        getattr(
            env,
            "max_waiting_queue_length",
            0,
        )
    )

    remaining_waiting_jobs = int(count_waiting_jobs(env))

    host_alpha_values = [
        float(host_agent.alpha.detach().cpu().item())
        for host_agent in host_sac_agents.values()
    ]

    host_alpha_mean = (
        float(np.mean(host_alpha_values)) if host_alpha_values else float("nan")
    )

    host_update_step_total = int(
        sum(int(host_agent.update_step) for host_agent in host_sac_agents.values())
    )

    host_training_steps_total = int(
        sum(int(step_count) for step_count in host_training_action_steps.values())
    )

    neighbor_feedback_summary = neighbor_feedback_store.summary()

    row: Dict[str, Any] = {
        "episode": int(stats.episode),
        "episode_seed": int(stats.episode_seed),
        "training_stage": str(stats.training_stage),
        "bcgh2_bayesian_game_enabled": stats.bayesian_game_enabled,
        "bcgh2_heuristic_guidance_enabled": bool(
            stats.reward_shaping_enabled or stats.discount_reduction_enabled
        ),
        "bcgh2_beta": float(stats.heuristic_beta),
        "bcgh2_original_gamma": float(stats.routing_gamma),
        "bcgh2_effective_gamma": float(
            stats.routing_gamma * stats.heuristic_beta
            if stats.discount_reduction_enabled
            else stats.routing_gamma
        ),
        "bcgh2_heuristic_reward_sum": float(stats.heuristic_reward_sum),
        "bcgh2_modified_routing_reward_sum": float(
            stats.routing_layer_reward_sum + stats.heuristic_reward_sum
        ),
        "bcgh2_nonterminal_score_mean": float(
            stats.heuristic_score_sum / max(1, stats.heuristic_nonterminal_count)
        ),
        "bcgh2_nonterminal_score_count": int(stats.heuristic_nonterminal_count),
        "bcgh2_score_coverage": (
            float(stats.heuristic_valid_score_count / stats.heuristic_nonterminal_count)
            if stats.heuristic_nonterminal_count
            else float("nan")
        ),
        "bcgh2_nonterminal_edge_fraction": float(
            stats.heuristic_nonterminal_count / max(1, stats.routing_edge_count)
        ),
        "bcgh2_reward_shaping_enabled": stats.reward_shaping_enabled,
        "bcgh2_discount_reduction_enabled": stats.discount_reduction_enabled,
        "bcgh_zero_diff_mode": bool(
            not conf.BCGH_ENABLE_BAYESIAN_GAME
            and not conf.BCGH_ENABLE_HEURISTIC_GUIDANCE
        ),
        "cloud_enabled": bool(
            getattr(
                env,
                "enable_cloud_action",
                False,
            )
        ),
        "neighbor_feedback_collection_enabled": bool(
            conf.COLLECT_NEIGHBOR_HISTORICAL_FEEDBACK
        ),
        "neighbor_feedback_decision_enabled": bool(
            conf.USE_NEIGHBOR_HISTORICAL_FEEDBACK
        ),
        "neighbor_feedback_episode_terminal_jobs_seen": int(
            neighbor_feedback_summary["episode_terminal_jobs_seen"]
        ),
        "neighbor_feedback_episode_pair_samples": int(
            neighbor_feedback_summary["episode_pair_samples"]
        ),
        "neighbor_feedback_total_terminal_jobs_seen": int(
            neighbor_feedback_summary["terminal_jobs_seen"]
        ),
        "neighbor_feedback_total_pair_samples": int(
            neighbor_feedback_summary["total_pair_samples"]
        ),
        "neighbor_feedback_active_pair_count": int(
            neighbor_feedback_summary["active_pair_count"]
        ),
        "wall_time_seconds": float(wall_time_seconds),
        "simulation_end_time": float(
            getattr(
                env,
                "current_time",
                0.0,
            )
        ),
        "training_episode_reward": float(stats.episode_return),
        "total_jobs": total_jobs,
        "completed_jobs": completed_jobs,
        "dropped_jobs": dropped_jobs,
        "unresolved_jobs": unresolved_jobs,
        "completion_rate": safe_ratio(
            completed_jobs,
            total_jobs,
        ),
        "drop_rate": safe_ratio(
            dropped_jobs,
            total_jobs,
        ),
        "queued_jobs": queued_jobs,
        "queue_admission_rate": safe_ratio(
            queued_jobs,
            total_jobs,
        ),
        "started_from_waiting_jobs": started_from_waiting_jobs,
        "waiting_timeout_drops": waiting_timeout_drops,
        "waiting_timeout_drop_rate": safe_ratio(
            waiting_timeout_drops,
            total_jobs,
        ),
        "max_waiting_queue_length": max_waiting_queue_length,
        "remaining_waiting_jobs": remaining_waiting_jobs,
        "energy_normalization_j": float(conf.ENERGY_NORMALIZATION_J),
        "energy_cost_weight": float(conf.ENERGY_COST_WEIGHT),
        "energy_optimization_enabled": bool(float(conf.ENERGY_COST_WEIGHT) > 0.0),
        "routing_decision_count": int(stats.routing_decision_count),
        "routing_actor_controlled_count": int(
            stats.routing_random_action_count + stats.routing_policy_action_count
        ),
        "routing_forced_action_count": int(stats.routing_forced_action_count),
        "routing_orchestrator_action_count": int(
            stats.routing_orchestrator_action_count
        ),
        "routing_random_action_count": int(stats.routing_random_action_count),
        "routing_policy_action_count": int(stats.routing_policy_action_count),
        "routing_self_count": int(stats.routing_self_count),
        "routing_self_rate": safe_ratio(
            stats.routing_self_count,
            stats.routing_decision_count,
        ),
        "routing_edge_count": int(stats.routing_edge_count),
        "routing_edge_rate": safe_ratio(
            stats.routing_edge_count,
            stats.routing_decision_count,
        ),
        "routing_cloud_count": int(stats.routing_cloud_count),
        "routing_cloud_rate": safe_ratio(
            stats.routing_cloud_count,
            stats.routing_decision_count,
        ),
        "routing_drop_count": int(stats.routing_drop_count),
        "routing_drop_rate": safe_ratio(
            stats.routing_drop_count,
            stats.routing_decision_count,
        ),
        "multi_hop_job_count": int(stats.multi_hop_job_count),
        "multi_hop_job_rate": safe_ratio(
            stats.multi_hop_job_count,
            stats.terminal_trace_flushed_count,
        ),
        "avg_routing_edge_hops_per_job": safe_ratio(
            stats.routing_edge_hop_total,
            stats.terminal_trace_flushed_count,
        ),
        "max_routing_hops": int(stats.max_routing_hops),
        "p95_routing_edge_hops": (
            float(np.percentile(stats.routing_edge_hops_by_job, 95))
            if stats.routing_edge_hops_by_job
            else 0.0
        ),
        "first_edge_absorption_count": int(stats.first_edge_absorbed_count),
        "first_edge_absorption_rate": safe_ratio(
            stats.first_edge_absorbed_count, stats.first_edge_successor_total
        ),
        "edge_successor_total": int(stats.edge_successor_total),
        "edge_target_absorption_count": int(stats.edge_successor_self_count),
        "edge_target_absorption_rate": safe_ratio(
            stats.edge_successor_self_count, stats.edge_successor_total
        ),
        "edge_to_edge_reforward_count": int(stats.edge_successor_edge_count),
        "edge_to_edge_reforward_rate": safe_ratio(
            stats.edge_successor_edge_count, stats.edge_successor_total
        ),
        "edge_to_cloud_transit_count": int(stats.edge_successor_cloud_count),
        "edge_to_cloud_transit_rate": safe_ratio(
            stats.edge_successor_cloud_count, stats.edge_successor_total
        ),
        "edge_target_drop_count": int(stats.edge_successor_drop_count),
        "edge_target_drop_rate": safe_ratio(
            stats.edge_successor_drop_count, stats.edge_successor_total
        ),
        "absorption_pair_counts_json": json.dumps(
            stats.absorption_pair_counts, ensure_ascii=False, sort_keys=True
        ),
        "routing_source_target_matrix_json": json.dumps(
            stats.routing_source_target_counts,
            ensure_ascii=False,
            sort_keys=True,
        ),
        "routing_reward_by_agent_json": json.dumps(
            stats.per_agent_returns,
            ensure_ascii=False,
            sort_keys=True,
        ),
        "host_decision_count": int(stats.host_decision_count),
        "host_random_action_count": int(stats.host_random_action_count),
        "host_policy_action_count": int(stats.host_policy_action_count),
        "host_started_count": int(stats.host_started_count),
        "host_started_rate": safe_ratio(
            stats.host_started_count,
            stats.host_decision_count,
        ),
        "host_queued_count": int(stats.host_queued_count),
        "host_queued_rate": safe_ratio(
            stats.host_queued_count,
            stats.host_decision_count,
        ),
        "host_dropped_count": int(stats.host_dropped_count),
        "host_dropped_rate": safe_ratio(
            stats.host_dropped_count,
            stats.host_decision_count,
        ),
        "routing_episode_updates": int(stats.routing_update_count),
        "routing_replay_size": int(len(routing_replay_buffer)),
        "routing_replay_trainable_size": int(
            routing_replay_buffer.num_trainable_actions
        ),
        "routing_update_step": int(routing_masac.update_step),
        "routing_global_decision_steps": int(global_decision_steps),
        "routing_training_action_steps": int(routing_normal_action_steps),
        "routing_critic_loss": stats.mean_routing_metric("critic_loss"),
        "routing_q1_loss": stats.mean_routing_metric("q1_loss"),
        "routing_q2_loss": stats.mean_routing_metric("q2_loss"),
        "routing_actor_loss": stats.mean_routing_metric("actor_loss"),
        "routing_alpha_loss": stats.mean_routing_metric("alpha_loss"),
        "routing_alpha": float(routing_masac.alpha.detach().cpu().item()),
        "routing_policy_entropy": stats.mean_routing_metric("policy_entropy"),
        "routing_target_entropy": stats.mean_routing_metric("target_entropy"),
        "routing_mean_q1": stats.mean_routing_metric("mean_q1"),
        "routing_mean_q2": stats.mean_routing_metric("mean_q2"),
        "routing_mean_target_q": stats.mean_routing_metric("mean_target_q"),
        "routing_heuristic_reward_mean": stats.mean_routing_metric(
            "heuristic_reward_mean"
        ),
        "routing_modified_reward_mean": stats.mean_routing_metric(
            "modified_reward_mean"
        ),
        "routing_raw_reward_mean": stats.mean_routing_metric("raw_reward_mean"),
        "host_episode_updates": int(stats.host_update_count),
        "host_replay_size_total": int(
            sum(len(buffer) for buffer in host_replay_buffers.values())
        ),
        "host_training_action_steps_total": host_training_steps_total,
        "host_update_step_total": host_update_step_total,
        "host_critic_loss": stats.mean_host_metric("critic_loss"),
        "host_q1_loss": stats.mean_host_metric("q1_loss"),
        "host_q2_loss": stats.mean_host_metric("q2_loss"),
        "host_actor_loss": stats.mean_host_metric("actor_loss"),
        "host_alpha_loss": stats.mean_host_metric("alpha_loss"),
        "host_alpha_mean": host_alpha_mean,
        "host_policy_entropy": stats.mean_host_metric("policy_entropy"),
        "host_target_entropy": stats.mean_host_metric("target_entropy"),
        "host_mean_q1": stats.mean_host_metric("mean_q1"),
        "host_mean_q2": stats.mean_host_metric("mean_q2"),
        "host_mean_target_q": stats.mean_host_metric("mean_target_q"),
        "routing_layer_reward_sum": float(stats.routing_layer_reward_sum),
        "host_layer_reward_sum": float(stats.host_layer_reward_sum),
        "terminal_trace_flushed_count": int(stats.terminal_trace_flushed_count),
        "routing_transition_flushed_count": int(stats.routing_transition_flushed_count),
        "host_transition_flushed_count": int(stats.host_transition_flushed_count),
        "pending_trace_count_end": int(pending_trace_store.pending_trace_count),
        "finalized_trace_count_end": int(pending_trace_store.finalized_trace_count),
        "causal_terminal_job_gap": int(total_jobs - stats.terminal_trace_flushed_count),
    }

    row.update(service_metrics)

    row.update(energy_metrics)

    for metric_name, metric_value in load_metrics.items():

        if metric_name in {
            "dc_load_details",
            "edge_host_load_details",
        }:
            continue

        row[metric_name] = metric_value

    return row


def build_dc_log_rows(
    stats: EpisodeStatistics,
    env: Any,
    host_sac_agents: Dict[str, LocalHostSAC],
    host_replay_buffers: Dict[str, HostReplayBuffer],
    host_training_action_steps: Dict[str, int],
    load_metrics: Dict[str, Any],
    neighbor_feedback_store: NeighborHistoricalFeedbackStore,
) -> list[Dict[str, Any]]:
    """
    构造 BCGH2-MASCA DC-level 日志。

    每个 Episode：
        每个 Edge DC 产生一行。

    用于分别诊断：
        Routing Agent 行为
        Local Host SAC
        当前 DC Resource Load
    """

    raw_dc_load_details = load_metrics.get(
        "dc_load_details",
        "{}",
    )

    raw_host_load_details = load_metrics.get(
        "edge_host_load_details",
        "{}",
    )

    dc_load_details = (
        json.loads(raw_dc_load_details)
        if isinstance(
            raw_dc_load_details,
            str,
        )
        else dict(raw_dc_load_details)
    )

    host_load_details = (
        json.loads(raw_host_load_details)
        if isinstance(
            raw_host_load_details,
            str,
        )
        else dict(raw_host_load_details)
    )

    rows: list[Dict[str, Any]] = []

    for dc_id_raw in env.edge_dc_ids:

        dc_id = str(dc_id_raw)
        neighbor_feedback_source_summary = neighbor_feedback_store.source_summary(dc_id)

        neighbor_feedback_snapshot = neighbor_feedback_store.snapshot_for_source(dc_id)

        dc_stats = stats.dc_counters.get(
            dc_id,
            {},
        )

        dc_load = dc_load_details.get(
            dc_id,
            {},
        )

        host_agent = host_sac_agents[dc_id]

        host_replay = host_replay_buffers[dc_id]

        dc_host_load_details = {
            host_key: host_detail
            for host_key, host_detail in host_load_details.items()
            if str(host_key).startswith(f"{dc_id}/")
        }

        source_target_counts = stats.routing_source_target_counts.get(
            dc_id,
            {},
        )

        row = {
            "episode": int(stats.episode),
            "episode_seed": int(stats.episode_seed),
            "training_stage": str(stats.training_stage),
            "dc_id": dc_id,
            "arrival_mode": str(
                dc_load.get(
                    "arrival_mode",
                    getattr(
                        env,
                        "dc_arrival_mode",
                        "uniform",
                    ),
                )
            ),
            "arrival_role": str(
                dc_load.get(
                    "arrival_role",
                    "uniform",
                )
            ),
            "configured_arrival_weight": float(
                dc_load.get(
                    "configured_arrival_weight",
                    1.0,
                )
            ),
            "configured_arrival_probability": float(
                dc_load.get(
                    "configured_arrival_probability",
                    0.0,
                )
            ),
            "configured_arrival_rate": float(
                dc_load.get(
                    "configured_arrival_rate",
                    0.0,
                )
            ),
            "expected_exogenous_arrival_count": float(
                dc_load.get(
                    "expected_exogenous_arrival_count",
                    0.0,
                )
            ),
            "exogenous_arrival_count": int(
                dc_load.get(
                    "exogenous_arrival_count",
                    0,
                )
            ),
            "exogenous_arrival_share": float(
                dc_load.get(
                    "exogenous_arrival_share",
                    0.0,
                )
            ),
            "observed_exogenous_arrival_rate": float(
                dc_load.get(
                    "observed_exogenous_arrival_rate",
                    0.0,
                )
            ),
            "routing_decisions": int(
                dc_stats.get(
                    "routing_decisions",
                    0,
                )
            ),
            "route_self_count": int(
                dc_stats.get(
                    "route_self_count",
                    0,
                )
            ),
            "route_out_edge_count": int(
                dc_stats.get(
                    "route_out_edge_count",
                    0,
                )
            ),
            "route_in_edge_count": int(
                dc_stats.get(
                    "route_in_edge_count",
                    0,
                )
            ),
            "incoming_edge_successor_count": int(
                dc_stats.get("incoming_edge_successor_count", 0)
            ),
            "incoming_edge_absorbed_count": int(
                dc_stats.get("incoming_edge_self_count", 0)
            ),
            "incoming_edge_absorption_rate": safe_ratio(
                dc_stats.get("incoming_edge_self_count", 0),
                dc_stats.get("incoming_edge_successor_count", 0),
            ),
            "incoming_edge_reforward_count": int(
                dc_stats.get("incoming_edge_edge_count", 0)
            ),
            "incoming_edge_cloud_count": int(
                dc_stats.get("incoming_edge_cloud_count", 0)
            ),
            "incoming_edge_drop_count": int(
                dc_stats.get("incoming_edge_drop_count", 0)
            ),
            "net_edge_migration_count": int(
                dc_stats.get(
                    "route_in_edge_count",
                    0,
                )
                - dc_stats.get(
                    "route_out_edge_count",
                    0,
                )
            ),
            "route_cloud_count": int(
                dc_stats.get(
                    "route_cloud_count",
                    0,
                )
            ),
            "route_drop_count": int(
                dc_stats.get(
                    "route_drop_count",
                    0,
                )
            ),
            "routing_reward": float(
                stats.per_agent_returns.get(
                    dc_id,
                    0.0,
                )
            ),
            "routing_out_targets_json": json.dumps(
                source_target_counts,
                ensure_ascii=False,
                sort_keys=True,
            ),
            "neighbor_feedback_outgoing_pair_count": int(
                neighbor_feedback_source_summary["outgoing_pair_count"]
            ),
            "neighbor_feedback_outgoing_sample_count": int(
                neighbor_feedback_source_summary["outgoing_sample_count"]
            ),
            "neighbor_feedback_snapshot_json": json.dumps(
                neighbor_feedback_snapshot,
                ensure_ascii=False,
                sort_keys=True,
            ),
            "host_decision_count": int(
                dc_stats.get(
                    "host_decisions",
                    0,
                )
            ),
            "host_random_count": int(
                dc_stats.get(
                    "host_random_count",
                    0,
                )
            ),
            "host_policy_count": int(
                dc_stats.get(
                    "host_policy_count",
                    0,
                )
            ),
            "host_started_count": int(
                dc_stats.get(
                    "host_started_count",
                    0,
                )
            ),
            "host_queued_count": int(
                dc_stats.get(
                    "host_queued_count",
                    0,
                )
            ),
            "host_dropped_count": int(
                dc_stats.get(
                    "host_dropped_count",
                    0,
                )
            ),
            "host_episode_updates": int(
                dc_stats.get(
                    "host_updates",
                    0,
                )
            ),
            "host_training_action_steps": int(
                host_training_action_steps.get(
                    dc_id,
                    0,
                )
            ),
            "host_replay_size": int(len(host_replay)),
            "host_update_step": int(host_agent.update_step),
            "host_critic_loss": stats.mean_host_metric(
                "critic_loss",
                dc_id=dc_id,
            ),
            "host_q1_loss": stats.mean_host_metric(
                "q1_loss",
                dc_id=dc_id,
            ),
            "host_q2_loss": stats.mean_host_metric(
                "q2_loss",
                dc_id=dc_id,
            ),
            "host_actor_loss": stats.mean_host_metric(
                "actor_loss",
                dc_id=dc_id,
            ),
            "host_alpha_loss": stats.mean_host_metric(
                "alpha_loss",
                dc_id=dc_id,
            ),
            "host_alpha": float(host_agent.alpha.detach().cpu().item()),
            "host_policy_entropy": stats.mean_host_metric(
                "policy_entropy",
                dc_id=dc_id,
            ),
            "host_target_entropy": stats.mean_host_metric(
                "target_entropy",
                dc_id=dc_id,
            ),
            "host_mean_q1": stats.mean_host_metric(
                "mean_q1",
                dc_id=dc_id,
            ),
            "host_mean_q2": stats.mean_host_metric(
                "mean_q2",
                dc_id=dc_id,
            ),
            "host_mean_target_q": stats.mean_host_metric(
                "mean_target_q",
                dc_id=dc_id,
            ),
            "dc_completed_jobs": int(
                dc_load.get(
                    "completed_jobs",
                    0,
                )
            ),
            "dc_cpu_capacity": float(
                dc_load.get(
                    "cpu_capacity",
                    0.0,
                )
            ),
            "dc_gpu_capacity": float(
                dc_load.get(
                    "gpu_capacity",
                    0.0,
                )
            ),
            "dc_avg_cpu_load": float(
                dc_load.get(
                    "avg_cpu_load",
                    0.0,
                )
            ),
            "dc_peak_cpu_load": float(
                dc_load.get(
                    "peak_cpu_load",
                    0.0,
                )
            ),
            "dc_avg_gpu_load": float(
                dc_load.get(
                    "avg_gpu_load",
                    0.0,
                )
            ),
            "dc_peak_gpu_load": float(
                dc_load.get(
                    "peak_gpu_load",
                    0.0,
                )
            ),
            "dc_avg_running_jobs": float(
                dc_load.get(
                    "avg_running_jobs",
                    0.0,
                )
            ),
            "dc_peak_running_jobs": int(
                dc_load.get(
                    "peak_running_jobs",
                    0,
                )
            ),
            "dc_busy_ratio": float(
                dc_load.get(
                    "busy_ratio",
                    0.0,
                )
            ),
            "host_load_details_json": json.dumps(
                dc_host_load_details,
                ensure_ascii=False,
                sort_keys=True,
            ),
        }

        rows.append(row)

    return rows


def print_episode_summary(
    row: Dict[str, Any],
) -> None:
    """
    打印 双层训练摘要。

    System / Routing / Host / Causal 分行显示，
    不再沿用单 MASAC 混合输出格式。
    """

    def format_float(
        value: Any,
        precision: int = 6,
    ) -> str:

        value = float(value)

        if not np.isfinite(value):
            return "nan"

        return f"{value:.{precision}f}"

    print(
        f"Episode "
        f"{int(row['episode']):5d} | "
        f"stage="
        f"{row['training_stage']} | "
        f"system_R="
        f"{float(row['training_episode_reward']):9.4f} | "
        f"completed="
        f"{int(row['completed_jobs']):5d} | "
        f"dropped="
        f"{int(row['dropped_jobs']):5d} | "
        f"SLA_vio="
        f"{float(row['sla_violation_rate']):6.2%} | "
        f"avg_T="
        f"{float(row['avg_completion_time']):8.2f}s"
    )

    print(
        f"  Routing | "
        f"decisions="
        f"{int(row['routing_decision_count']):5d} | "
        f"Self="
        f"{int(row['routing_self_count']):5d} "
        f"({float(row['routing_self_rate']):6.2%}) | "
        f"Edge="
        f"{int(row['routing_edge_count']):5d} "
        f"({float(row['routing_edge_rate']):6.2%}) | "
        f"Cloud="
        f"{int(row['routing_cloud_count']):5d} "
        f"({float(row['routing_cloud_rate']):6.2%}) | "
        f"multi-hop="
        f"{int(row['multi_hop_job_count']):4d}"
    )

    print(
        f"  R-Train | "
        f"buffer="
        f"{int(row['routing_replay_trainable_size']):7d} | "
        f"updates="
        f"{int(row['routing_episode_updates']):5d} | "
        f"critic="
        f"{format_float(row['routing_critic_loss'])} | "
        f"alpha="
        f"{format_float(row['routing_alpha'], 5)} | "
        f"entropy="
        f"{format_float(row['routing_policy_entropy'], 5)}"
    )

    print(
        f"  Host    | "
        f"decisions="
        f"{int(row['host_decision_count']):5d} | "
        f"started="
        f"{int(row['host_started_count']):5d} | "
        f"queued="
        f"{int(row['host_queued_count']):5d} | "
        f"dropped="
        f"{int(row['host_dropped_count']):5d}"
    )

    print(
        f"  H-Train | "
        f"buffer="
        f"{int(row['host_replay_size_total']):7d} | "
        f"updates="
        f"{int(row['host_episode_updates']):5d} | "
        f"critic="
        f"{format_float(row['host_critic_loss'])} | "
        f"alpha_mean="
        f"{format_float(row['host_alpha_mean'], 5)} | "
        f"entropy="
        f"{format_float(row['host_policy_entropy'], 5)}"
    )

    print(
        f"  Causal  | "
        f"terminal="
        f"{int(row['terminal_trace_flushed_count']):5d} | "
        f"routing_T="
        f"{int(row['routing_transition_flushed_count']):5d} | "
        f"host_T="
        f"{int(row['host_transition_flushed_count']):5d} | "
        f"pending="
        f"{int(row['pending_trace_count_end']):3d} | "
        f"gap="
        f"{int(row['causal_terminal_job_gap']):3d}"
    )

    print(
        f"  Load    | "
        f"EdgeCPU="
        f"{float(row['edge_avg_cpu_load']):6.2%}/"
        f"{float(row['edge_peak_cpu_load']):6.2%} | "
        f"EdgeGPU="
        f"{float(row['edge_avg_gpu_load']):6.2%}/"
        f"{float(row['edge_peak_gpu_load']):6.2%} | "
        f"HostCPUStd="
        f"{float(row['edge_host_avg_cpu_load_std']):6.2%}"
    )

    print(
        f"  Energy  | "
        f"total="
        f"{float(row['total_system_energy_kwh']):10.6f} kWh | "
        f"per_job="
        f"{float(row['system_energy_per_completed_job_j']):10.2f} J | "
        f"avgP="
        f"{float(row['system_total_equivalent_avg_power_w']):10.2f} W"
    )


def evaluate_episode(
    env,
    routing_agent,
    host_agents,
    settings,
    *,
    seed=42,
    deterministic=True,
    diagnostic_scores=False,
):
    """Evaluate learned Actor with online feedback, without logit bias or updates."""
    validate_bcgh_feature_config(settings)
    routing_obs = RoutingObservationBuilder(
        env,
        use_neighbor_historical_feedback=settings.use_neighbor_historical_feedback,
    )
    routing_state = RoutingCentralizedStateBuilder(
        env, routing_observation_builder=routing_obs
    )
    host_obs = HostObservationBuilder(env)
    if (
        routing_agent.local_obs_dim != routing_obs.obs_dim
        or routing_agent.action_dim != env.action_dim
    ):
        raise ValueError("Loaded routing Actor does not match environment")
    if (
        routing_agent.global_state_dim != routing_state.state_dim
        or routing_agent.num_agents != len(env.edge_dc_ids)
    ):
        raise ValueError("Loaded routing Critic does not match environment")
    if set(host_agents) != set(env.edge_dc_ids):
        raise ValueError("Loaded Host agents do not match edge DCs")
    for dc_id, agent in host_agents.items():
        if agent.obs_dim != host_obs.get_obs_dim(
            dc_id
        ) or agent.action_dim != host_obs.get_action_dim(dc_id):
            raise ValueError(f"Host shape mismatch: {dc_id}")
    structure = build_checkpoint_structure_metadata(env, routing_agent, host_agents)
    if getattr(routing_agent, "environment_structure", structure) != structure:
        raise ValueError("Evaluation topology/DC/Host order differs from training")
    runtime = (
        ShortWindowRuntime(build_short_window_game(env, settings))
        if settings.enable_bayesian_game
        else None
    )
    feedback = (
        ShortWindowFeedbackStore(
            env,
            window_s=settings.short_window_s,
            confidence_scale_samples=settings.bayesian_confidence_scale,
        )
        if runtime is not None
        else NeighborHistoricalFeedbackStore(
            env,
            ewma_alpha=settings.neighbor_feedback_ewma_alpha,
            age_scale_samples=settings.neighbor_feedback_age_scale_samples,
            confidence_scale_samples=settings.neighbor_feedback_confidence_scale_samples,
        )
    )
    routing_obs.set_neighbor_feedback_provider(feedback)
    reward_values = {
        f.name: getattr(conf, f.name.upper())
        for f in fields(TrainingRewardConfig)
        if hasattr(conf, f.name.upper())
    }
    reward_values.update(
        max_latency_s=env.max_latency,
        max_job_duration_s=env.max_job_duration,
        sla_deadline_ratio=env.sla_deadline_ratio,
        drop_deadline_ratio=env.drop_deadline_ratio,
    )
    reward = HMasacTrainingRewardModel(TrainingRewardConfig(**reward_values))
    traces = PendingJobTraceStore()
    score_settings = replace(
        settings,
        enable_heuristic_guidance=bool(
            diagnostic_scores and settings.enable_heuristic_guidance
        ),
    )
    scorer = RoutingStateHeuristic(
        runtime, feedback, score_settings, env.routing_action_target_dc_ids
    )
    collector = TransitionCollector(
        env,
        routing_obs,
        routing_state,
        traces,
        reward,
        short_window_runtime=runtime,
        state_heuristic=scorer,
    )
    env.reset(seed=seed)
    routing_count = 0
    decision = None

    def finalize(trace):
        if settings.collect_neighbor_historical_feedback:
            feedback.update_from_finalized_trace(trace)
        if runtime is not None:
            for item in build_bayesian_evidence_from_finalized_trace(trace, env):
                runtime.game.update_belief(item, update_clock=env.current_time)
        traces.pop_finalized_trace(trace.job_id)

    def consume_outcomes():
        for event in env.pop_job_outcome_events():
            traces.record_reward_event(
                job_id=str(event["job_id"]),
                env_time=float(event["env_time"]),
                reward_delta=reward.calculate_outcome_reward(event),
                reason=str(event["reason"]),
                terminal=bool(event.get("terminal", False)),
            )
            if event.get("terminal", False):
                finalize(traces.finalize_terminal_trace(str(event["job_id"])))

    while env.agents:
        if env.has_pending_host_decision():
            ctx = env.get_pending_host_decision()
            dc, job_id, now = (
                str(ctx["dc_id"]),
                str(ctx["job_id"]),
                float(env.current_time),
            )
            obs = host_obs.build(dc_id=dc, job_id=job_id)
            action = host_agents[dc].select_action(obs, deterministic=deterministic)
            result = env.execute_pending_host_action(host_action=action)
            traces.record_host_step(
                job_id=job_id,
                dc_id=dc,
                env_time=now,
                host_obs=obs,
                action=action,
                host_id=str(result["host_id"]),
                action_source="policy",
            )
            traces.record_host_result(
                job_id=job_id,
                result=result["execution_result"],
                env_time=float(result["env_time"]),
            )
            if runtime is not None:
                runtime.record_host_result(
                    job_id=job_id,
                    dc_id=dc,
                    execution_result=str(result["execution_result"]),
                    now=float(result["env_time"]),
                )
            decision = None
        elif collector.drain_one_dead_agent():
            decision = None
            continue
        else:
            if decision is None:
                decision = collector.capture_decision()
            if decision.forced_action is not None:
                action, source = decision.forced_action, "forced"
            else:
                action = routing_agent.select_action(
                    decision.local_obs,
                    decision.agent_index,
                    deterministic=deterministic,
                )
                source = "policy"
            result, decision = collector.execute_and_record(
                decision, action, action_source=source
            )
            routing_count += 1
            if result.job_finalized:
                finalize(traces.get_finalized_trace(result.job_id))
        consume_outcomes()
    consume_outcomes()
    traces.assert_no_open_trace()
    traces.assert_no_unflushed_finalized_trace()
    return {
        **calculate_service_metrics(env),
        "routing_decisions": routing_count,
        "actor_logit_bias": False,
        "data_augmentation": False,
        "absorption": runtime.absorption_summary() if runtime is not None else {},
    }
