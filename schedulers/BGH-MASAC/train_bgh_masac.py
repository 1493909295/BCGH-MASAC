from __future__ import annotations
import csv
import json
import random
import sys
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

# The BGH-MASAC modules use script-style imports (for example,
# ``from h_masac_agent import ...``).  When this entry point is launched with
# ``python -m schedulers.BGH-MASAC.train_bgh_masac``, Python only puts the
# project root on sys.path, not this file's directory.  Add both locations
# before importing any project modules so the command works consistently on
# the server and when the file is executed directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BGH_MASAC_DIR = Path(__file__).resolve().parent

if str(BGH_MASAC_DIR) not in sys.path:
    sys.path.insert(0, str(BGH_MASAC_DIR))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from datetime import datetime
from h_masac_agent import (
    RoutingMASAC,
    RoutingMASACConfig,
    LocalHostSAC,
    HostSACConfig,
    atomic_checkpoint_text_write,
)
from neighbor_feedback import (
    NeighborHistoricalFeedbackStore,
    ShortWindowFeedbackStore,
)
from routing_replay_buffer import (RoutingReplayBuffer,)
from host_replay_buffer import (HostReplayBuffer,)
from pending_job_trace import (PendingJobTraceStore, FinalizedJobTrace,)
from transition_collector import (DecisionSnapshot,TransitionCollector,)
from environment.cloud_edge_env import CloudEdgeEnv
import config as conf
from routing_observation import (RoutingObservationBuilder,)
from routing_centralized_state import (RoutingCentralizedStateBuilder,)
from host_observation import (HostObservationBuilder,)
from bayesian_game import (
    BayesianRoutingGameDefinition,
    BayesianStaticRoutingContext,
    build_bayesian_static_routing_context,
    build_bayesian_routing_game_definition,
)
from bayesian_congestion_game import BayesianCongestionGame
from bayesian_evidence import (
    build_bayesian_evidence_from_finalized_trace,
)
from guided_policy import GuidedRoutingPolicy
from guidance_schedule import GuidanceLambdaSchedule
from short_window_runtime import ShortWindowRuntime, build_short_window_game, GuidanceDecisionLog


from training_reward import (
    TrainingRewardConfig,
    HMasacTrainingRewardModel,
)

CHECKPOINT_SCHEMA_VERSION = 2

CHECKPOINT_ARCHITECTURE = (
     "bgh_masac_two_layer_routing_host_v1"
)

TERMINAL_FAILURE_REASONS = frozenset({
    "waiting_timeout",
    "cloud_arrival_timeout",
    "cloud_resource_failure",
    "local_host_arrival_timeout",
    "local_host_resource_failure",
})

UPDATE_TENSOR_METRIC_NAMES = ("critic_loss",
        "q1_loss",
        "q2_loss",
        "mean_q1",
        "mean_q2",
        "mean_target_q",
        "actor_loss",
        "alpha_loss",
        "alpha",
        "policy_entropy",
        "target_entropy",)

# 找环境代码
ENVIRONMENT_DIR = PROJECT_ROOT / "environment"

@dataclass(frozen=True)
class TrainConfig:

    num_episodes: int = (conf.Episodes)

    routing_replay_capacity: int = (conf.ROUTING_REPLAY_CAPACITY)
    host_replay_capacity: int = (conf.HOST_REPLAY_CAPACITY)
    routing_batch_size: int = (conf.ROUTING_BATCH_SIZE)
    routing_random_warmup_steps: int = (conf.ROUTING_RANDOM_WARMUP_STEPS)
    routing_learning_starts: int = (conf.ROUTING_LEARNING_STARTS)
    routing_train_every: int = (conf.ROUTING_TRAIN_EVERY)
    routing_updates_per_train: int = (conf.ROUTING_UPDATES_PER_TRAIN)

    host_batch_size: int = (conf.HOST_BATCH_SIZE)
    host_random_warmup_steps: int = (conf.HOST_RANDOM_WARMUP_STEPS)
    host_learning_starts: int = (conf.HOST_LEARNING_STARTS)
    host_train_every: int = (conf.HOST_TRAIN_EVERY)
    host_updates_per_train: int = (conf.HOST_UPDATES_PER_TRAIN)

    # Three-stage training
    host_pretrain_episodes: int = (conf.HOST_PRETRAIN_EPISODES)
    routing_train_episodes: int = (conf.ROUTING_TRAIN_EPISODES)
    joint_finetune_episodes: int = (conf.JOINT_FINETUNE_EPISODES)
    log_interval: int = (conf.Log_interval)
    checkpoint_interval: int = (conf.Checkpoint_Interval)  # 兼容旧调用，双层训练不按间隔保存
    seed: int = conf.Seed

    checkpoint_dir: str = (conf.BGH_MASAC_CHECKPOINT_DIR)
    episode_log_csv_path: str = (conf.BGH_MASAC_EPISODE_LOG_CSV_PATH)
    dc_log_csv_path: str = (conf.BGH_MASAC_DC_LOG_CSV_PATH)
    old_env_path: Optional[str] = (conf.Old_Env_Path)
    resume_checkpoint: Optional[str] = (conf.BGH_MASAC_RESUME_CHECKPOINT)

    vary_episode_seed: bool = ( conf.Vary_Episode_Seed)
    collect_neighbor_historical_feedback: bool = (conf.COLLECT_NEIGHBOR_HISTORICAL_FEEDBACK)
    use_neighbor_historical_feedback: bool = (conf.USE_NEIGHBOR_HISTORICAL_FEEDBACK)
    neighbor_feedback_ewma_alpha: float = (conf.NEIGHBOR_FEEDBACK_EWMA_ALPHA)
    neighbor_feedback_age_scale_samples: float = (conf.NEIGHBOR_FEEDBACK_AGE_SCALE_SAMPLES)
    neighbor_feedback_confidence_scale_samples: float = (conf.NEIGHBOR_FEEDBACK_CONFIDENCE_SCALE_SAMPLES)
    enable_bayesian_game: bool = (conf.BGH_ENABLE_BAYESIAN_GAME)
    enable_heuristic_guidance: bool = (conf.BGH_ENABLE_HEURISTIC_GUIDANCE)
    bayesian_prior_alpha: float = (conf.BGH_BAYESIAN_PRIOR_ALPHA)
    bayesian_prior_beta: float = (conf.BGH_BAYESIAN_PRIOR_BETA)
    bayesian_confidence_scale: float = (conf.BGH_BAYESIAN_CONFIDENCE_SCALE)
    short_window_s: float = conf.BGH_SHORT_WINDOW_S
    source_evidence_weight: float = conf.BGH_SOURCE_EVIDENCE_WEIGHT
    source_volume_scale: float = conf.BGH_SOURCE_VOLUME_SCALE
    source_confidence_scale: float = conf.BGH_SOURCE_CONFIDENCE_SCALE
    resource_cpu_weight: float = conf.BGH_RESOURCE_CPU_WEIGHT
    resource_gpu_weight: float = conf.BGH_RESOURCE_GPU_WEIGHT
    resource_cost_scale: float = conf.BGH_RESOURCE_COST_SCALE
    max_logit_bias: float = conf.BGH_MAX_LOGIT_BIAS
    absorption_enabled: bool = conf.BGH_ABSORPTION_ENABLED
    absorption_prior_self: float = conf.BGH_ABSORPTION_PRIOR_SELF
    absorption_prior_edge: float = conf.BGH_ABSORPTION_PRIOR_EDGE
    absorption_prior_cloud: float = conf.BGH_ABSORPTION_PRIOR_CLOUD
    absorption_prior_drop: float = conf.BGH_ABSORPTION_PRIOR_DROP
    absorption_confidence_scale: float = conf.BGH_ABSORPTION_CONFIDENCE_SCALE
    absorption_edge_weight: float = conf.BGH_ABSORPTION_EDGE_WEIGHT
    absorption_cloud_weight: float = conf.BGH_ABSORPTION_CLOUD_WEIGHT
    absorption_drop_weight: float = conf.BGH_ABSORPTION_DROP_WEIGHT
    absorption_demand_thresholds: tuple = conf.BGH_ABSORPTION_DEMAND_THRESHOLDS
    absorption_class_confidence_scale: float = conf.BGH_ABSORPTION_CLASS_CONFIDENCE_SCALE
    absorption_policy_actions_only: bool = conf.BGH_ABSORPTION_POLICY_ACTIONS_ONLY
    pressure_linear_cost_weight: float = (conf.BGH_PRESSURE_LINEAR_WEIGHT)
    pressure_quadratic_cost_weight: float = (conf.BGH_PRESSURE_QUADRATIC_WEIGHT)
    benefit_success_weight: float = (conf.BGH_BENEFIT_SUCCESS_WEIGHT)
    benefit_sla_weight: float = (conf.BGH_BENEFIT_SLA_WEIGHT)
    benefit_delay_weight: float = (conf.BGH_BENEFIT_DELAY_WEIGHT)
    risk_congestion_cost_weight: float = (
        conf.BGH_RISK_CONGESTION_COST_WEIGHT
    )
    guidance_scale: float = (conf.BGH_GUIDANCE_SCALE)
    guidance_lambda_stage2_start: float = (
        conf.BGH_GUIDANCE_LAMBDA_STAGE2_START
    )
    guidance_lambda_stage2_peak: float = (
        conf.BGH_GUIDANCE_LAMBDA_STAGE2_PEAK
    )
    guidance_lambda_stage3_end: float = (
        conf.BGH_GUIDANCE_LAMBDA_STAGE3_END
    )

def bgh_guidance_metadata(train_config):
    """Runtime semantics are part of the checkpoint even when tensor shapes match."""
    names = ("enable_bayesian_game", "enable_heuristic_guidance", "short_window_s",
        "bayesian_prior_alpha", "bayesian_prior_beta", "bayesian_confidence_scale",
        "source_evidence_weight", "source_volume_scale", "source_confidence_scale",
        "resource_cpu_weight", "resource_gpu_weight", "resource_cost_scale",
        "absorption_enabled", "absorption_prior_self", "absorption_prior_edge",
        "absorption_prior_cloud", "absorption_prior_drop", "absorption_confidence_scale",
        "absorption_edge_weight", "absorption_cloud_weight", "absorption_drop_weight",
        "absorption_demand_thresholds", "absorption_class_confidence_scale",
        "absorption_policy_actions_only",
        "pressure_linear_cost_weight", "pressure_quadratic_cost_weight",
        "benefit_success_weight", "benefit_sla_weight", "benefit_delay_weight",
        "risk_congestion_cost_weight", "guidance_scale", "max_logit_bias",
        "guidance_lambda_stage2_start", "guidance_lambda_stage2_peak", "guidance_lambda_stage3_end",
        "collect_neighbor_historical_feedback", "use_neighbor_historical_feedback")
    parameters = {}
    for name in names:
        value = getattr(train_config, name)
        parameters[name] = list(value) if isinstance(value, tuple) else value
    return {"version": 3, "clock": "episode_simulation_seconds",
        "source_semantics": "local_queue_direct_previous_hop_bounded_soft_evidence",
        "pressure_semantics": "executed_edge_transfer_cpu_gpu_marginal_cost",
        "absorption_semantics": "directed_edge_successor_self_edge_cloud_drop",
        "parameters": parameters}


# ==============================================================
# BGH-MASAC Runtime Feature Mode
#
# 第三步的核心约束：
#
#   Bayesian Game = False
#   Heuristic Guidance = False
#
# 时，BGH-MASAC 必须严格执行 H-MASAC-equivalent 路径。
#
# 注意：
#   Zero-Diff Mode 是运行时派生状态，
#   不是第三个人工配置开关。
# ==============================================================


def is_bgh_zero_diff_mode(
        train_config: TrainConfig,
) -> bool:
    """
    判断当前 BGH-MASAC 是否运行在 H-MASAC 等价模式。

    Zero-Diff Mode：
        Bayesian Game        = False
        Heuristic Guidance   = False
    """

    return (
        not bool(
            train_config.enable_bayesian_game
        )
        and not bool(
            train_config.enable_heuristic_guidance
        )
    )


def get_bgh_runtime_mode(
        train_config: TrainConfig,
) -> str:
    """
    返回当前 BGH-MASAC 的实验模式名称。

    这些名称提前固定下来，后续可直接用于消融实验：
        H-MASAC-equivalent
        Bayesian-only
        Heuristic-only
        Bayesian+Heuristic
    """

    bayesian_enabled = bool(
        train_config.enable_bayesian_game
    )

    heuristic_enabled = bool(
        train_config.enable_heuristic_guidance
    )

    if (
        not bayesian_enabled
        and not heuristic_enabled
    ):
        return "H-MASAC-equivalent"

    if (
        bayesian_enabled
        and not heuristic_enabled
    ):
        return "Bayesian-only"

    if (
        not bayesian_enabled
        and heuristic_enabled
    ):
        return "Heuristic-only"

    return "Bayesian+Heuristic"


def validate_bgh_feature_config(
        train_config: TrainConfig,
) -> None:
    """
    验证 BGH-MASAC 运行模式的依赖关系。

    Bayesian Game 负责维护终止任务 Evidence、Pair Posterior 和
    历史拥塞压力；Heuristic Guidance 在此状态之上生成 Actor Bias，
    因而不能脱离 Bayesian Game 单独启用。
    """

    bayesian_enabled = bool(
        train_config.enable_bayesian_game
    )
    heuristic_enabled = bool(
        train_config.enable_heuristic_guidance
    )

    if heuristic_enabled and not bayesian_enabled:
        raise RuntimeError(
            "BGH_ENABLE_HEURISTIC_GUIDANCE=True 要求同时启用 "
            "BGH_ENABLE_BAYESIAN_GAME；启发式偏置依赖贝叶斯-拥塞状态。"
        )

    if is_bgh_zero_diff_mode(
        train_config
    ):

        if bool(
            train_config
                .use_neighbor_historical_feedback
        ):
            raise RuntimeError(
                "BGH-MASAC Zero-Diff Mode 要求 "
                "USE_NEIGHBOR_HISTORICAL_FEEDBACK=False。"
                "Historical Feedback 可以继续收集，"
                "但当前不能进入 Routing Observation。"
            )

        return

    # Bayesian-only 与 Bayesian+Heuristic 均为已接通的正式模式。
    return

# Two-Level Scheduler 三阶段训练状态。
class TrainingStage( str,Enum,):
    HOST_PRETRAIN = ("host_pretrain")
    ROUTING_TRAIN = ("routing_train")
    JOINT_FINETUNE = ("joint_finetune")

#   一个 Episode 内的双层训练统计
@dataclass
class EpisodeStatistics:

    episode: int
    episode_seed: int
    training_stage: str

    # System / Routing Reward Bookkeeping
    per_agent_returns: Dict[str, float]
    episode_return: float = 0.0

    # Routing Behavior
    routing_decision_count: int = 0
    routing_forced_action_count: int = 0
    routing_orchestrator_action_count: int = 0
    routing_random_action_count: int = 0
    routing_policy_action_count: int = 0
    routing_self_count: int = 0
    routing_edge_count: int = 0
    routing_cloud_count: int = 0
    routing_drop_count: int = 0

    # Host Behavior
    host_decision_count: int = 0
    host_random_action_count: int = 0
    host_policy_action_count: int = 0
    host_started_count: int = 0
    host_queued_count: int = 0
    host_dropped_count: int = 0

    # Gradient Update Counts
    routing_update_count: int = 0
    host_update_count: int = 0

    # Routing Update Metrics
    routing_update_metric_sums: Dict[ str, float,] = field(default_factory=dict)
    routing_update_metric_counts: Dict[str,int,] = field(default_factory=dict)

    # Host Update Metrics：全系统聚合
    host_update_metric_sums: Dict[str,float,] = field(default_factory=dict)
    host_update_metric_counts: Dict[str,int,] = field(default_factory=dict)

    # Host Update Metrics：逐 DC
    host_update_metric_sums_by_dc: Dict[str, Dict[str, float],] = field(default_factory=dict)
    host_update_metric_counts_by_dc: Dict[ str,Dict[str, int],] = field(default_factory=dict)

    # Per-DC Counters
    dc_counters: Dict[str,Dict[str, int],] = field(default_factory=dict)

    # Routing Source -> Target Matrix
    routing_source_target_counts: Dict[ str,Dict[str, int],] = field(default_factory=dict )

    # Finalized Causal Trace Statistics
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

    # 当前 Episode 的 Guided Policy λ；仅记录强度，不改变 Replay。
    guidance_lambda: float = 0.0

    def _inc_dc(self,dc_id: str, metric_name: str,delta: int = 1,) -> None:
        dc_id = str(dc_id)
        dc_metrics = (self.dc_counters.setdefault(dc_id,{},))
        dc_metrics[metric_name] = int(dc_metrics.get(metric_name, 0,) + int(delta))

    @staticmethod
    def _accumulate_metric(sums: Dict[str, float], counts: Dict[str, int], metric_name: str,  metric_value: float,) -> None:

        metric_value = float(metric_value)

        if not np.isfinite(metric_value):
            return

        sums[metric_name] = float(sums.get(metric_name, 0.0,) + metric_value)
        counts[metric_name] = int(counts.get(metric_name, 0,) + 1)

    # Routing Decision Statistics
    def record_routing_decision(self,agent_id: str,reward: float,action_type: str,action_source: str,target_dc_id: Optional[str],) -> None:
        agent_id = str(agent_id)
        action_type = str(action_type)
        action_source = str(action_source)
        reward = float(reward)
        target_dc_id = (
            None
            if target_dc_id is None
            else str(target_dc_id)
        )

        # System reward bookkeeping
        self.episode_return += reward
        self.per_agent_returns[agent_id] = float(self.per_agent_returns.get(agent_id, 0.0,) + reward )

        # Routing decision source
        self.routing_decision_count += 1
        self._inc_dc(agent_id,"routing_decisions",)

        if action_source == "forced":
            self.routing_forced_action_count += 1
        elif action_source == "orchestrator":
            self.routing_orchestrator_action_count += 1
        elif action_source == "random":
            self.routing_random_action_count += 1
        elif action_source == "policy":
            self.routing_policy_action_count += 1
        else:
            raise ValueError("未知 Routing action_source："f"{action_source}")

        # Routing action semantics
        if action_type == "self":
            self.routing_self_count += 1
            self._inc_dc(agent_id,"route_self_count",)
        elif action_type == "edge_dc":
            if target_dc_id is None:
                raise RuntimeError("统计 Edge Routing 时缺少 target_dc_id。")
            self.routing_edge_count += 1
            self._inc_dc(agent_id,"route_out_edge_count",)
            self._inc_dc(target_dc_id,"route_in_edge_count",)
        elif action_type == "cloud":
            self.routing_cloud_count += 1
            self._inc_dc(agent_id,"route_cloud_count",)
        elif action_type == "drop":
            self.routing_drop_count += 1
            self._inc_dc(agent_id,"route_drop_count",)
        else:
            raise ValueError("未知 Routing action_type："f"{action_type}")


        # Source -> Target Matrix
        target_key = (
            str(target_dc_id)
            if target_dc_id is not None
            else "__drop__"
        )
        source_targets = (self.routing_source_target_counts.setdefault( agent_id,{},))

        source_targets[
            target_key
        ] = int(
            source_targets.get(
                target_key,
                0,
            )
            + 1
        )

    # ==========================================================
    # Host Decision Statistics
    # ==============================================================

    def record_host_decision(
            self,
            dc_id: str,
            action_source: str,
    ) -> None:

        dc_id = str(
            dc_id
        )

        action_source = str(
            action_source
        )

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
            raise ValueError(
                "未知 Host action_source："
                f"{action_source}"
            )

    def record_host_result(
            self,
            dc_id: str,
            execution_result: str,
    ) -> None:

        dc_id = str(
            dc_id
        )

        execution_result = str(
            execution_result
        )

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
            raise ValueError(
                "未知 Host execution_result："
                f"{execution_result}"
            )

    # ==========================================================
    # Delayed Reward
    # ==============================================================

    def record_delayed_training_reward(
            self,
            agent_id: str,
            reward_delta: float,
    ) -> None:

        agent_id = str(
            agent_id
        )

        reward_delta = float(
            reward_delta
        )

        self.episode_return += (
            reward_delta
        )

        self.per_agent_returns[
            agent_id
        ] = float(
            self.per_agent_returns.get(
                agent_id,
                0.0,
            )
            + reward_delta
        )

    # ==========================================================
    # Routing Update Metrics
    # ==============================================================

    def record_routing_update(
            self,
            update_info: Dict[str, float],
    ) -> None:

        self.routing_update_count += 1

        for metric_name, metric_value in (
                update_info.items()
        ):

            self._accumulate_metric(
                sums=(
                    self.routing_update_metric_sums
                ),

                counts=(
                    self.routing_update_metric_counts
                ),

                metric_name=(
                    metric_name
                ),

                metric_value=float(
                    metric_value
                ),
            )

    # ==========================================================
    # Host Update Metrics
    # ==============================================================

    def record_host_update(
            self,
            dc_id: str,
            update_info: Dict[str, float],
    ) -> None:

        dc_id = str(
            dc_id
        )

        self.host_update_count += 1

        self._inc_dc(
            dc_id,
            "host_updates",
        )

        dc_sums = (
            self.host_update_metric_sums_by_dc
            .setdefault(
                dc_id,
                {},
            )
        )

        dc_counts = (
            self.host_update_metric_counts_by_dc
            .setdefault(
                dc_id,
                {},
            )
        )

        for metric_name, metric_value in (
                update_info.items()
        ):

            metric_value = float(
                metric_value
            )

            # 所有 Host SAC 的整体加权平均。
            self._accumulate_metric(
                sums=(
                    self.host_update_metric_sums
                ),

                counts=(
                    self.host_update_metric_counts
                ),

                metric_name=(
                    metric_name
                ),

                metric_value=(
                    metric_value
                ),
            )

            # 当前 DC 自己的平均。
            self._accumulate_metric(
                sums=dc_sums,
                counts=dc_counts,
                metric_name=(
                    metric_name
                ),
                metric_value=(
                    metric_value
                ),
            )

    # ==========================================================
    # Metric Mean Helpers
    # ==============================================================

    def mean_routing_metric(
            self,
            metric_name: str,
    ) -> float:

        count = (
            self.routing_update_metric_counts
            .get(
                metric_name,
                0,
            )
        )

        if count <= 0:
            return float("nan")

        return float(
            self.routing_update_metric_sums[
                metric_name
            ]
            / count
        )

    def mean_host_metric(
            self,
            metric_name: str,
            dc_id: Optional[str] = None,
    ) -> float:

        if dc_id is None:

            sums = (
                self.host_update_metric_sums
            )

            counts = (
                self.host_update_metric_counts
            )

        else:

            dc_id = str(
                dc_id
            )

            sums = (
                self.host_update_metric_sums_by_dc
                .get(
                    dc_id,
                    {},
                )
            )

            counts = (
                self.host_update_metric_counts_by_dc
                .get(
                    dc_id,
                    {},
                )
            )

        count = counts.get(
            metric_name,
            0,
        )

        if count <= 0:
            return float("nan")

        return float(
            sums[
                metric_name
            ]
            / count
        )

    # ==========================================================
    # Finalized Job Causal Trace
    #
    # 必须在 Replay 写入成功以后调用。
    # ==============================================================

    def record_finalized_trace(
            self,
            finalized_trace: FinalizedJobTrace,
    ) -> None:

        self.terminal_trace_flushed_count += 1

        routing_transition_count = len(
            finalized_trace.routing_transitions
        )

        self.routing_transition_flushed_count += (
            routing_transition_count
        )

        if (
                finalized_trace.host_transition
                is not None
        ):
            self.host_transition_flushed_count += 1

        # ------------------------------------------------------
        # Multi-hop 统计的是 Edge -> Edge forwarding 次数。
        # Self 本身不是 Edge forwarding hop。
        # ------------------------------------------------------

        edge_hops = sum(
            1
            for routing_step
            in finalized_trace.routing_steps
            if (
                routing_step.action_type
                == "edge_dc"
            )
        )

        self.routing_edge_hop_total += int(
            edge_hops
        )
        self.routing_edge_hops_by_job.append(int(edge_hops))

        self.max_routing_hops = max(
            self.max_routing_hops,
            int(edge_hops),
        )

        if edge_hops >= 2:
            self.multi_hop_job_count += 1

        # Classify the immediate successor of every Edge transfer.  SELF is
        # counted as absorption only when the real Host admission succeeded.
        first_edge_seen = False
        routing_steps = tuple(finalized_trace.routing_steps)
        for index, step in enumerate(routing_steps):
            if str(step.action_type) != "edge_dc":
                continue
            if index + 1 >= len(routing_steps):
                raise RuntimeError(
                    f"Edge step has no successor: job={finalized_trace.job_id}, sequence={step.sequence_index}")
            successor = routing_steps[index + 1]
            if str(successor.agent_id) != str(step.target_dc_id):
                raise RuntimeError(
                    f"Edge successor target mismatch: job={finalized_trace.job_id}, "
                    f"expected={step.target_dc_id}, actual={successor.agent_id}")
            successor_type = str(successor.action_type)
            if successor_type == "self":
                host_result = (None if finalized_trace.host_step is None
                               else str(finalized_trace.host_step.execution_result))
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
            setattr(self, f"edge_successor_{outcome}_count",
                    int(getattr(self, f"edge_successor_{outcome}_count")) + 1)
            target_dc_id = str(step.target_dc_id)
            self._inc_dc(target_dc_id, "incoming_edge_successor_count")
            self._inc_dc(target_dc_id, f"incoming_edge_{outcome}_count")
            pair = f"{step.source_dc_id}->{target_dc_id}"
            pair_counts = self.absorption_pair_counts.setdefault(
                pair, {name: 0 for name in ("self", "edge", "cloud", "drop")})
            pair_counts[outcome] += 1
            if not first_edge_seen:
                self.first_edge_successor_total += 1
                self.first_edge_absorbed_count += int(outcome == "self")
                first_edge_seen = True

        # ------------------------------------------------------
        # 两层 Replay reward 分开统计。
        #
        # 注意：
        # Routing reward + Host reward
        # 不能作为 system reward。
        # ------------------------------------------------------

        self.routing_layer_reward_sum += float(
            sum(
                float(
                    transition.reward
                )
                for transition
                in finalized_trace.routing_transitions
            )
        )

        if (
                finalized_trace.host_transition
                is not None
        ):
            self.host_layer_reward_sum += float(
                finalized_trace
                .host_transition
                .reward
            )


def validate_training_stage_config(
        train_config: TrainConfig,
) -> None:

    host_pretrain = int(
        train_config
        .host_pretrain_episodes
    )

    routing_train = int(
        train_config
        .routing_train_episodes
    )

    joint_finetune = int(
        train_config
        .joint_finetune_episodes
    )

    if min(
        host_pretrain,
        routing_train,
        joint_finetune,
    ) < 0:
        raise ValueError(
            "三阶段 Episode 数不能为负数。"
        )

    configured_total = (
        host_pretrain
        + routing_train
        + joint_finetune
    )

    if (
        configured_total
        != int(
            train_config.num_episodes
        )
    ):
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

    episode = int(
        episode
    )

    host_end = int(
        train_config
        .host_pretrain_episodes
    )

    routing_end = (
        host_end
        + int(
            train_config
            .routing_train_episodes
        )
    )

    if episode <= host_end:
        return (
            TrainingStage
            .HOST_PRETRAIN
        )

    if episode <= routing_end:
        return (
            TrainingStage
            .ROUTING_TRAIN
        )

    return (
        TrainingStage
        .JOINT_FINETUNE
    )

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

    host_end = int(
        train_config.host_pretrain_episodes
    )

    routing_end = (
        host_end
        + int(
            train_config.routing_train_episodes
        )
    )

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

    host_end = int(
        train_config.host_pretrain_episodes
    )

    routing_end = (
        host_end
        + int(
            train_config.routing_train_episodes
        )
    )

    if stage == TrainingStage.HOST_PRETRAIN:
        return host_end

    if stage == TrainingStage.ROUTING_TRAIN:
        return routing_end

    return int(
        train_config.num_episodes
    )


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
        host_sac_agents:
        Dict[str, LocalHostSAC],
) -> None:
    """
    根据 Training Stage 明确设置两层网络模式。

    真正是否更新参数仍由 Orchestrator
    是否调用 update() 决定。
    """

    if (
        stage
        == TrainingStage.HOST_PRETRAIN
    ):

        routing_masac.eval_mode()

        for host_agent in (
            host_sac_agents.values()
        ):
            host_agent.train_mode()

        return

    if (
        stage
        == TrainingStage.ROUTING_TRAIN
    ):

        routing_masac.train_mode()

        for host_agent in (
            host_sac_agents.values()
        ):
            host_agent.eval_mode()

        return

    # Joint fine-tune
    routing_masac.train_mode()

    for host_agent in (
        host_sac_agents.values()
    ):
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

    agent_id = str(
        agent_id
    )

    action = (
        env.routing_dc_id_to_action
        .get(
            agent_id
        )
    )

    if action is None:
        raise RuntimeError(
            "找不到当前 DC 对应的 Self "
            "Routing action："
            f"dc={agent_id}"
        )

    decoded = env._decode_action(
        agent_id=agent_id,
        action=int(action),
    )

    if (
        str(
            decoded["action_type"]
        )
        != "self"
    ):
        raise RuntimeError(
            "Stage 1 Self bypass "
            "动作编码错误："
            f"dc={agent_id}, "
            f"action={action}"
        )

    return int(
        action
    )

def choose_random_host_action(
        action_dim: int,
        rng: np.random.Generator,
) -> int:

    action_dim = int(
        action_dim
    )

    if action_dim <= 0:
        raise ValueError(
            "Host action_dim 必须 > 0"
        )

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

        reference_value = (
            update_info[
                "critic_loss"
            ]
        )

        reference_device = (
            reference_value.device
            if torch.is_tensor(
                reference_value
            )
            else torch.device("cpu")
        )

        metric_tensors = []

        for metric_name in (
                UPDATE_TENSOR_METRIC_NAMES
        ):

            metric_value = (
                update_info[
                    metric_name
                ]
            )

            if torch.is_tensor(
                    metric_value
            ):
                metric_tensor = (
                    metric_value
                    .detach()
                    .to(
                        device=reference_device,
                        dtype=torch.float32,
                    )
                    .reshape(())
                )

            else:
                metric_tensor = torch.tensor(
                    float(
                        metric_value
                    ),
                    dtype=torch.float32,
                    device=reference_device,
                )

            metric_tensors.append(
                metric_tensor
            )

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

    update_matrix_cpu = (
        update_matrix
        .detach()
        .cpu()
        .numpy()
    )

    return [
        {
            metric_name: float(
                row[
                    metric_index
                ]
            )
            for metric_index, metric_name
            in enumerate(
                UPDATE_TENSOR_METRIC_NAMES
            )
        }
        for row
        in update_matrix_cpu
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

    for update_info in (
        _convert_update_block_to_cpu(
            update_infos
        )
    ):
        stats.record_routing_update(
            update_info
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

    for update_info in (
        _convert_update_block_to_cpu(
            update_infos
        )
    ):
        stats.record_host_update(
            dc_id=dc_id,
            update_info=(
                update_info
            ),
        )


def flush_finalized_trace_to_replay(
        finalized_trace: FinalizedJobTrace,

        routing_replay_buffer:
        RoutingReplayBuffer,

        host_replay_buffers:
        Dict[str, HostReplayBuffer],

        stats: EpisodeStatistics,

        neighbor_feedback_store:
        NeighborHistoricalFeedbackStore,

        collect_neighbor_historical_feedback:
        bool,

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

    # ==========================================================
    # 先检查 Host Buffer 是否存在。
    #
    # 在真正写 Routing Replay 前先检查，
    # 避免 Host buffer 配置错误造成半写入状态。
    # ==========================================================

    host_transition = (
        finalized_trace
        .host_transition
    )

    host_replay_buffer = None

    if host_transition is not None:

        host_dc_id = str(
            host_transition.dc_id
        )

        host_replay_buffer = (
            host_replay_buffers.get(
                host_dc_id
            )
        )

        if host_replay_buffer is None:
            raise RuntimeError(
                "找不到 HostTransition 对应的 "
                "HostReplayBuffer："
                f"job={finalized_trace.job_id}, "
                f"dc={host_dc_id}"
            )

    # Evidence 先完成转换和边界校验，再写 Replay，避免转换失败时出现
    # Replay 已写入但 Bayesian 链未完成的半成功状态。
    bayesian_evidences = tuple()
    if bayesian_game is not None:
        if env is None:
            raise RuntimeError(
                "启用 Bayesian Evidence 转换时必须提供 Environment。"
            )
        bayesian_evidences = (
            build_bayesian_evidence_from_finalized_trace(
                finalized_trace,
                env,
            )
        )

    # ==========================================================
    # Routing Replay
    # ==========================================================

    for routing_transition in (
        finalized_trace
        .routing_transitions
    ):

        routing_replay_buffer.add(
            routing_transition
        )

    # ==========================================================
    # Host Replay
    #
    # 一个 Self Job 最多写一条。
    # ==========================================================

    if (
            host_transition is not None
            and host_replay_buffer is not None
    ):
        host_replay_buffer.add(
            host_transition
        )

    # ==============================================================
    # Routing / Host 两个 ReplayBuffer 全部写入成功以后，
    # 才把这条完整 Job 因果链计入 Episode 日志。
    #
    # 这样日志中的：
    #
    #   terminal_trace_flushed_count
    #   routing_transition_flushed_count
    #   host_transition_flushed_count
    #
    # 与真正成功写入 Replay 的经验严格一致。
    # ==============================================================

    stats.record_finalized_trace(
        finalized_trace
    )
    if collect_neighbor_historical_feedback:
        neighbor_feedback_store.update_from_finalized_trace(
            finalized_trace
        )

    # 结果在终止后才可见；按仿真时间更新短窗口后验。
    # 竞争需求已在转发执行时记录，不能在这里重复累计。
    if bayesian_game is not None:
        for evidence in bayesian_evidences:
            bayesian_game.update_belief(
                evidence,
                update_clock=float(env.current_time),
            )

def consume_environment_outcome_events(
        env: CloudEdgeEnv,

        pending_trace_store:
        PendingJobTraceStore,

        routing_replay_buffer:
        RoutingReplayBuffer,

        host_replay_buffers:
        Dict[str, HostReplayBuffer],

        stats: EpisodeStatistics,

        neighbor_feedback_store:
        NeighborHistoricalFeedbackStore,

        collect_neighbor_historical_feedback:
        bool,

        training_reward_model:
        HMasacTrainingRewardModel,

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

    # ==============================================================
    # Environment 只返回 Job Outcome Facts。
    # ==============================================================

    outcome_events = (
        env.pop_job_outcome_events()
    )

    if not outcome_events:
        return

    for outcome_event in (
            outcome_events
    ):

        job_id = str(
            outcome_event[
                "job_id"
            ]
        )

        reason = str(
            outcome_event[
                "reason"
            ]
        )

        env_time = float(
            outcome_event[
                "env_time"
            ]
        )

        is_terminal = bool(
            outcome_event.get(
                "terminal",
                False,
            )
        )

        # ==========================================================
        # 真正 Training Reward 在 Trainer 侧生成。
        # ==============================================================

        reward_delta = float(
            training_reward_model
                .calculate_outcome_reward(
                outcome_event
            )
        )



        # ======================================================
        # 1. Reward Event 永远先进入因果链。
        # ======================================================

        pending_trace_store.record_reward_event(
            job_id=job_id,
            env_time=env_time,
            reward_delta=reward_delta,
            reason=reason,
            terminal=is_terminal,
        )

        # ======================================================
        # 2. Episode statistics：
        #
        # 这里只做统计归因。
        # 不再修改任何 ReplayBuffer。
        # ======================================================

        if is_terminal:

            finalized_trace = (
                pending_trace_store
                .finalize_terminal_trace(
                    job_id=job_id
                )
            )

            correction_agent_id = None

            if (
                finalized_trace
                .routing_transitions
            ):
                correction_agent_id = str(
                    finalized_trace
                    .routing_transitions[-1]
                    .agent_id
                )

            # ==================================================
            # 3. Job terminal 后，
            #    一次性写入两个正式经验池。
            # ==================================================

            flush_finalized_trace_to_replay(
                finalized_trace=(
                    finalized_trace
                ),

                routing_replay_buffer=(
                    routing_replay_buffer
                ),

                host_replay_buffers=(
                    host_replay_buffers
                ),

                stats=stats,

                neighbor_feedback_store=(
                    neighbor_feedback_store
                ),

                collect_neighbor_historical_feedback=(
                    collect_neighbor_historical_feedback
                ),

                bayesian_game=bayesian_game,
                env=env,
            )

            # ==================================================
            # 4. Replay 写入成功以后，
            #    才允许从 Finalized Trace Store 移除。
            # ==================================================

            pending_trace_store.pop_finalized_trace(
                job_id
            )

            if correction_agent_id is not None:

                stats.record_delayed_training_reward(
                    agent_id=(
                        correction_agent_id
                    ),

                    reward_delta=(
                        reward_delta
                    ),
                )

            continue

        # ======================================================
        # Non-terminal delayed reward：
        #
        # 只记录在 Causal Trace。
        # 不修改 Replay。
        #
        # Episode statistics 仍然正常记录。
        # ======================================================

        pending_trace = (
            pending_trace_store
            .get_trace(
                job_id
            )
        )

        correction_agent_id = None

        for routing_step in reversed(
            pending_trace.routing_steps
        ):

            if (
                routing_step.action_source
                != "forced"
            ):
                correction_agent_id = str(
                    routing_step.agent_id
                )
                break

        if correction_agent_id is not None:

            stats.record_delayed_training_reward(
                agent_id=(
                    correction_agent_id
                ),

                reward_delta=(
                    reward_delta
                ),
            )

def set_global_random_seeds(seed: int) -> None:
    """统一设置 Python、NumPy 和 PyTorch 随机种子。"""

    # 转换成标准 Python int。
    seed = int(seed)

    # 设置 Python random 随机种子。
    random.seed(seed)

    # 设置 NumPy 旧式全局随机接口的种子。
    # ReplayBuffer 和预热动作仍会使用各自独立的 default_rng。
    np.random.seed(seed)

    # 设置 CPU 上的 PyTorch 随机种子。
    torch.manual_seed(seed)

    # CUDA 可用时设置所有 GPU 的随机种子。
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_environment(seed: int, old_env_path: Optional[str] = None,) -> Any:
    if old_env_path is None:
        return CloudEdgeEnv(seed=int(seed))
    return CloudEdgeEnv(
        old_env_path=str(old_env_path),
        seed=int(seed),
    )

# 从 DecisionSnapshot 的 action_mask 中随机选择一个合法动作'
def choose_random_routing_action(
    action_dim: int,
    rng: np.random.Generator,
) -> int:

    action_dim = int(
        action_dim
    )

    if action_dim <= 0:
        raise ValueError(
            "Routing action_dim 必须大于 0。"
        )

    return int(
        rng.integers(
            low=0,
            high=action_dim,
        )
    )



# 遍历所有数据中心和 host，统计完成队列中的任务数量
def count_completed_jobs(env: Any) -> int:
    completed_count = 0
    for datacenter in getattr(env, "datacenters", []):
        # 遍历当前数据中心的全部 host。
        for host in getattr(datacenter, "host_list", []):
            # 读取 host 完成队列。
            completed_queue = getattr(host, "completed_queue", None)

            # 没有完成队列时跳过。
            if completed_queue is None:
                continue

            # 将该 host 的完成任务数量加入总数。
            completed_count += len(completed_queue)
    return int(completed_count)

# 统计当前 episode 的 SLA 与任务完成时间指标
def calculate_service_metrics(env: Any) -> Dict[str, float]:
    completed_jobs = []
    for job in getattr(env, "jobs", []):
        if getattr(job, "finish_time", None) is not None:
            completed_jobs.append(job)

    total_jobs = int(len(getattr(env, "jobs", [])))

    dropped_jobs = int(len(getattr(env, "dropped_jobs_info", [],)))

    total_completion_time = 0.0
    sla_satisfied_jobs = 0
    sla_violated_completed_jobs = 0
    total_violation_degree = 0.0

    sla_ratio = float(env.sla_deadline_ratio)
    drop_ratio = float(env.drop_deadline_ratio)

    for job in completed_jobs:
        turnaround_time = (job.get_turnaround_time())
        if turnaround_time is None:
            continue

        turnaround_time = float(turnaround_time)
        duration = max(float(job.duration),1e-8,)
        sla_limit = (sla_ratio * duration)
        drop_limit = (drop_ratio * duration)
        total_completion_time += (turnaround_time)

        if turnaround_time <= sla_limit:
            sla_satisfied_jobs += 1

        else:
            sla_violated_completed_jobs += 1
            violation_degree = (
                                       turnaround_time - sla_limit
                               ) / max(
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

    completed_count = len(
        completed_jobs
    )

    avg_completion_time = (
        total_completion_time
        / completed_count
        if completed_count > 0
        else 0.0
    )

    # Drop 直接视为最严重 SLA violation，degree = 1。
    total_violation_degree += float(
        dropped_jobs
    )

    sla_satisfaction_rate = (
        sla_satisfied_jobs / total_jobs
        if total_jobs > 0
        else 0.0
    )

    sla_violation_rate = (
        (
                sla_violated_completed_jobs
                + dropped_jobs
        )
        / total_jobs
        if total_jobs > 0
        else 0.0
    )

    mean_sla_violation_degree = (
        total_violation_degree
        / total_jobs
        if total_jobs > 0
        else 0.0
    )

    return {
        "sla_satisfied_jobs": int(
            sla_satisfied_jobs
        ),
        "sla_violated_completed_jobs": int(
            sla_violated_completed_jobs
        ),
        "sla_satisfaction_rate": float(
            sla_satisfaction_rate
        ),
        "sla_violation_rate": float(
            sla_violation_rate
        ),
        "mean_sla_violation_degree": float(
            mean_sla_violation_degree
        ),
        "total_completion_time": float(
            total_completion_time
        ),
        "avg_completion_time": float(
            avg_completion_time
        ),
    }

# 统计当前环境所有 Host 中剩余的 waiting job 数量
def count_waiting_jobs(env: Any) -> int:
    waiting_count = 0
    for datacenter in getattr(env, "datacenters", [],):
        for host in getattr(datacenter, "host_list", [],):
            waiting_queue = getattr(host, "waiting_queue", None,)
            if waiting_queue is None:
                continue
            waiting_count += len(waiting_queue)
    return int(waiting_count)

# 功能函数计算安全比例，避免分母为0日志出问题
def safe_ratio(numerator: float, denominator: float,) -> float:
    denominator = float(denominator)
    if denominator <= 0.0:
        return 0.0
    return float(float(numerator) / denominator)

# 日志用，计算资源占用情况
def calculate_resource_timeline_metrics(jobs: list, episode_duration_s: float,) -> Dict[str, float]:
    episode_duration_s = max(float(episode_duration_s), 0.0,)
    cpu_resource_seconds = 0.0
    gpu_resource_seconds = 0.0
    running_job_seconds = 0.0
    events: Dict[float, list] = {}
    valid_running_jobs = 0

    for job in jobs:
        start_time = getattr(job, "start_time", None,)
        finish_time = getattr(job, "finish_time", None,)

        if (start_time is None or finish_time is None):
            continue

        start_time = float(start_time)
        finish_time = float(finish_time)
        run_time_s = max(finish_time - start_time, 0.0,)
        cpu_request = float(getattr(job, "cpu_request", 0.0,))
        gpu_request = float(getattr( job, "gpu_request", 0.0,))

        cpu_resource_seconds += (cpu_request * run_time_s)
        gpu_resource_seconds += (gpu_request * run_time_s)
        running_job_seconds += (run_time_s)
        valid_running_jobs += 1

        if start_time not in events:
            events[start_time] = [0.0, 0.0, 0,]
        if finish_time not in events:
            events[finish_time] = [0.0, 0.0, 0,]

        # 任务开始：CPU/GPU/Running Count 增加。
        events[start_time][0] += (cpu_request)
        events[start_time][1] += (gpu_request)
        events[start_time][2] += 1

        # 任务结束：CPU/GPU/Running Count 释放。
        events[finish_time][0] -= (cpu_request)
        events[finish_time][1] -= (gpu_request)
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
        interval_s = max(event_time - previous_event_time, 0.0,)

        # 在前一个事件到当前事件之间，如果至少有一个 Job 在运行，则 Host/System 是 busy 状态。
        if current_running_jobs > 0:
            busy_time_s += (interval_s)

        cpu_delta, gpu_delta, running_delta = (events[event_time])

        current_used_cpu += float(cpu_delta)
        current_used_gpu += float(gpu_delta)
        current_running_jobs += int(running_delta)
        # 防止浮点残差产生 -1e-15 一类值。
        current_used_cpu = max(current_used_cpu, 0.0,)
        current_used_gpu = max(current_used_gpu, 0.0,)
        current_running_jobs = max(current_running_jobs, 0,)

        peak_used_cpu = max(peak_used_cpu, current_used_cpu,)
        peak_used_gpu = max(peak_used_gpu, current_used_gpu,)
        peak_running_jobs = max(peak_running_jobs, current_running_jobs,)
        previous_event_time = (event_time)

    return {
        "valid_running_jobs":
            int(valid_running_jobs),
        "cpu_resource_seconds":
            float(cpu_resource_seconds),
        "gpu_resource_seconds":
            float(gpu_resource_seconds),
        "avg_used_cpu":
            safe_ratio(
                cpu_resource_seconds,
                episode_duration_s,
            ),
        "avg_used_gpu":
            safe_ratio(
                gpu_resource_seconds,
                episode_duration_s,
            ),
        "peak_used_cpu":
            float(peak_used_cpu),
        "peak_used_gpu":
            float(peak_used_gpu),
        "avg_running_jobs":
            safe_ratio(
                running_job_seconds,
                episode_duration_s,
            ),
        "peak_running_jobs":
            int(peak_running_jobs),
        "busy_time_s":
            float(busy_time_s),
        "busy_ratio":
            safe_ratio(
                busy_time_s,
                episode_duration_s,
            ),
    }

# Episode 级 Workload / Edge / Cloud Load 统计
def calculate_episode_load_metrics(env: Any,) -> Dict[str, Any]:

    episode_duration_s = max(float(getattr(env, "current_time", 0.0,)), 0.0,)
    all_jobs = list(getattr(env, "jobs", [],))
    cloud_id = str(getattr(env, "cloud_id", "cloud", ))

    cpu_requests = np.asarray([float(getattr(job, "cpu_request", 0.0,)) for job in all_jobs],dtype=np.float64,)
    gpu_requests = np.asarray([float(getattr(job, "gpu_request", 0.0,)) for job in all_jobs],dtype=np.float64,)
    durations = np.asarray([float( getattr( job, "duration", 0.0,)) for job in all_jobs], dtype=np.float64,)
    arrival_times = np.asarray([float(getattr(job, "arrive_time", 0.0,)) for job in all_jobs],dtype=np.float64,)

    edge_dc_ids = [
        str(dc_id)
        for dc_id in getattr(env, "edge_dc_ids", [])
    ]
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
    dc_exogenous_arrival_times = {
        dc_id: []
        for dc_id in edge_dc_ids
    }
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
            float(getattr(job, "arrive_time", 0.0,))
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
                np.max(dc_arrival_times)
                - np.min(dc_arrival_times)
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
                float(getattr(env, "lambda_rate", 0.0))
                * configured_probability,
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
            "configured_arrival_probability": (
                configured_probability
            ),
            "configured_arrival_rate": (
                configured_arrival_rate
            ),
            "expected_exogenous_arrival_count": float(
                len(all_jobs) * configured_probability
            ),
            "exogenous_arrival_count": dc_arrival_count,
            "exogenous_arrival_share": safe_ratio(
                dc_arrival_count,
                len(all_jobs),
            ),
            "exogenous_arrival_span_s": dc_arrival_span_s,
            "observed_exogenous_arrival_rate": (
                dc_observed_arrival_rate
            ),
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

    for dc in getattr(env, "datacenters", [],):
        dc_id = str(dc.dc_id)
        is_cloud = (dc_id == cloud_id)
        dc_jobs = []
        dc_cpu_capacity = 0.0
        dc_gpu_capacity = 0.0

        for host_idx, host in enumerate(getattr(dc, "host_list", [],)):
            completed_queue = getattr(host, "completed_queue", None,)
            host_jobs = list(getattr(completed_queue, "_queue", [],))
            dc_jobs.extend(host_jobs)
            host_timeline = (
                calculate_resource_timeline_metrics(
                    jobs=host_jobs,
                    episode_duration_s=episode_duration_s,
                )
            )

            cpu_capacity = float(getattr(host, "cpu_num", 0.0,))
            gpu_capacity = float(getattr(host, "gpu_capacity_num", 0.0,))

            if not is_cloud:
                dc_cpu_capacity += (cpu_capacity)
                dc_gpu_capacity += (gpu_capacity)
                edge_total_cpu_capacity += (cpu_capacity)
                edge_total_gpu_capacity += (gpu_capacity)
                avg_cpu_load = safe_ratio(host_timeline["avg_used_cpu"], cpu_capacity,)
                peak_cpu_load = safe_ratio(host_timeline["peak_used_cpu"], cpu_capacity,)
                avg_gpu_load = safe_ratio(host_timeline["avg_used_gpu"], gpu_capacity,)
                peak_gpu_load = safe_ratio(host_timeline["peak_used_gpu"], gpu_capacity,)
                edge_host_cpu_avg_loads.append(avg_cpu_load)

                if gpu_capacity > 0.0:
                    edge_host_gpu_avg_loads.append(avg_gpu_load)

                edge_host_busy_ratios.append(float(host_timeline["busy_ratio"]))
                edge_host_details[
                    f"{dc_id}/{host.host_id}"
                ] = {
                    "cpu_capacity":cpu_capacity,
                    "gpu_capacity":gpu_capacity,
                    "completed_jobs":len(host_jobs),
                    "avg_cpu_load":avg_cpu_load,
                    "peak_cpu_load":peak_cpu_load,
                    "avg_gpu_load":avg_gpu_load,
                    "peak_gpu_load":peak_gpu_load,
                    "avg_running_jobs":host_timeline["avg_running_jobs"],
                    "peak_running_jobs":host_timeline["peak_running_jobs"],
                    "busy_ratio":host_timeline["busy_ratio"],
                }

        dc_timeline = (calculate_resource_timeline_metrics(jobs=dc_jobs, episode_duration_s=episode_duration_s,))

        if is_cloud:
            cloud_jobs.extend(dc_jobs)
            dc_load_details[dc_id] = {
                "completed_jobs":len(dc_jobs),
                "avg_used_cpu":dc_timeline["avg_used_cpu"],
                "peak_used_cpu":dc_timeline["peak_used_cpu"],
                "avg_used_gpu":dc_timeline["avg_used_gpu"],
                "peak_used_gpu":dc_timeline[ "peak_used_gpu"],
                "avg_running_jobs":dc_timeline["avg_running_jobs"],
                "peak_running_jobs":dc_timeline["peak_running_jobs"],
                "busy_ratio":dc_timeline[ "busy_ratio"],
            }

        else:
            edge_jobs.extend(dc_jobs)
            dc_load_details[dc_id] = {
                "cpu_capacity":dc_cpu_capacity,
                "gpu_capacity":dc_gpu_capacity,
                "completed_jobs":len(dc_jobs),
                "avg_cpu_load":safe_ratio(dc_timeline["avg_used_cpu"],dc_cpu_capacity,),
                "peak_cpu_load":safe_ratio(dc_timeline["peak_used_cpu"],dc_cpu_capacity,),
                "avg_gpu_load":safe_ratio(dc_timeline["avg_used_gpu"], dc_gpu_capacity,),
                "peak_gpu_load":safe_ratio(dc_timeline["peak_used_gpu"],dc_gpu_capacity,),
                "avg_running_jobs":dc_timeline["avg_running_jobs"],
                "peak_running_jobs": dc_timeline["peak_running_jobs"],
                "busy_ratio":dc_timeline["busy_ratio"],
            }
            dc_load_details[dc_id].update(
                dc_exogenous_arrival_details.get(
                    dc_id,
                    {},
                )
            )

    edge_timeline = (calculate_resource_timeline_metrics(jobs=edge_jobs, episode_duration_s=episode_duration_s,))
    cloud_timeline = (calculate_resource_timeline_metrics(jobs=cloud_jobs,episode_duration_s=episode_duration_s,))
    edge_cpu_host_mean = (float(np.mean(edge_host_cpu_avg_loads))
        if edge_host_cpu_avg_loads
        else 0.0
    )
    edge_cpu_host_std = (float(np.std(edge_host_cpu_avg_loads))
        if edge_host_cpu_avg_loads
        else 0.0
    )
    edge_cpu_host_p95 = (float(np.percentile(edge_host_cpu_avg_loads,95,))
        if edge_host_cpu_avg_loads
        else 0.0
    )
    edge_gpu_host_mean = (float(np.mean(edge_host_gpu_avg_loads))
        if edge_host_gpu_avg_loads
        else 0.0
    )
    edge_gpu_host_std = (float(np.std(edge_host_gpu_avg_loads))
        if edge_host_gpu_avg_loads
        else 0.0
    )

    if len(arrival_times) >= 2:
        arrival_span_s = float(np.max(arrival_times) - np.min(arrival_times))
        observed_arrival_rate = safe_ratio(len(arrival_times) - 1, arrival_span_s,)

    else:
        arrival_span_s = 0.0
        observed_arrival_rate = 0.0

    return {

        "dc_arrival_mode":
            dc_arrival_mode,

        "configured_total_arrival_rate":
            float(
                getattr(
                    env,
                    "lambda_rate",
                    0.0,
                )
            ),

        "workload_missing_origin_count":
            int(missing_origin_count),

        "workload_unknown_origin_count":
            int(unknown_origin_count),

        "dc_exogenous_arrival_details":
            json.dumps(
                dc_exogenous_arrival_details,
                ensure_ascii=False,
                sort_keys=True,
            ),

        "workload_total_cpu_request":
            float(np.sum(cpu_requests))
            if cpu_requests.size
            else 0.0,

        "workload_mean_cpu_request":
            float(
                np.mean(cpu_requests)
            )
            if cpu_requests.size
            else 0.0,

        "workload_max_cpu_request":
            float(
                np.max(cpu_requests)
            )
            if cpu_requests.size
            else 0.0,

        "workload_total_gpu_request":
            float(
                np.sum(gpu_requests)
            )
            if gpu_requests.size
            else 0.0,

        "workload_mean_gpu_request":
            float(
                np.mean(gpu_requests)
            )
            if gpu_requests.size
            else 0.0,

        "workload_max_gpu_request":
            float(
                np.max(gpu_requests)
            )
            if gpu_requests.size
            else 0.0,

        "workload_gpu_job_ratio":
            float(
                np.mean(
                    gpu_requests > 0.0
                )
            )
            if gpu_requests.size
            else 0.0,

        "workload_total_duration_s":
            float(
                np.sum(durations)
            )
            if durations.size
            else 0.0,

        "workload_mean_duration_s":
            float(
                np.mean(durations)
            )
            if durations.size
            else 0.0,

        "workload_p95_duration_s":
            float(
                np.percentile(
                    durations,
                    95,
                )
            )
            if durations.size
            else 0.0,

        "workload_max_duration_s":
            float(
                np.max(durations)
            )
            if durations.size
            else 0.0,

        "workload_arrival_span_s":
            arrival_span_s,

        "workload_observed_arrival_rate":
            observed_arrival_rate,

        "edge_completed_jobs":
            int(
                len(edge_jobs)
            ),

        "edge_total_cpu_capacity":
            float(
                edge_total_cpu_capacity
            ),

        "edge_total_gpu_capacity":
            float(
                edge_total_gpu_capacity
            ),

        "edge_cpu_resource_seconds":
            float(
                edge_timeline[
                    "cpu_resource_seconds"
                ]
            ),

        "edge_gpu_resource_seconds":
            float(
                edge_timeline[
                    "gpu_resource_seconds"
                ]
            ),

        # 整个 Edge 系统容量加权、时间加权平均负载。
        "edge_avg_cpu_load":
            safe_ratio(
                edge_timeline[
                    "avg_used_cpu"
                ],
                edge_total_cpu_capacity,
            ),

        "edge_peak_cpu_load":
            safe_ratio(
                edge_timeline[
                    "peak_used_cpu"
                ],
                edge_total_cpu_capacity,
            ),

        "edge_avg_gpu_load":
            safe_ratio(
                edge_timeline[
                    "avg_used_gpu"
                ],
                edge_total_gpu_capacity,
            ),

        "edge_peak_gpu_load":
            safe_ratio(
                edge_timeline[
                    "peak_used_gpu"
                ],
                edge_total_gpu_capacity,
            ),

        "edge_avg_running_jobs":
            float(
                edge_timeline[
                    "avg_running_jobs"
                ]
            ),

        "edge_peak_running_jobs":
            int(
                edge_timeline[
                    "peak_running_jobs"
                ]
            ),

        # 每台 Edge Host 平均 CPU Load 的均值、标准差、P95。
        #
        # std 越大说明长期负载越不均衡。
        "edge_host_avg_cpu_load_mean":
            edge_cpu_host_mean,

        "edge_host_avg_cpu_load_std":
            edge_cpu_host_std,

        "edge_host_avg_cpu_load_p95":
            edge_cpu_host_p95,

        "edge_host_avg_gpu_load_mean":
            edge_gpu_host_mean,

        "edge_host_avg_gpu_load_std":
            edge_gpu_host_std,

        "edge_host_busy_ratio_mean":
            float(
                np.mean(
                    edge_host_busy_ratios
                )
            )
            if edge_host_busy_ratios
            else 0.0,

        "edge_host_busy_ratio_max":
            float(
                np.max(
                    edge_host_busy_ratios
                )
            )
            if edge_host_busy_ratios
            else 0.0,

        "cloud_completed_jobs":
            int(
                len(cloud_jobs)
            ),

        "cloud_cpu_resource_seconds":
            float(
                cloud_timeline[
                    "cpu_resource_seconds"
                ]
            ),

        "cloud_gpu_resource_seconds":
            float(
                cloud_timeline[
                    "gpu_resource_seconds"
                ]
            ),

        "cloud_avg_used_cpu":
            float(
                cloud_timeline[
                    "avg_used_cpu"
                ]
            ),

        "cloud_peak_used_cpu":
            float(
                cloud_timeline[
                    "peak_used_cpu"
                ]
            ),

        "cloud_avg_used_gpu":
            float(
                cloud_timeline[
                    "avg_used_gpu"
                ]
            ),

        "cloud_peak_used_gpu":
            float(
                cloud_timeline[
                    "peak_used_gpu"
                ]
            ),

        "cloud_avg_running_jobs":
            float(
                cloud_timeline[
                    "avg_running_jobs"
                ]
            ),

        "cloud_peak_running_jobs":
            int(
                cloud_timeline[
                    "peak_running_jobs"
                ]
            ),

        "cloud_busy_ratio":
            float(
                cloud_timeline[
                    "busy_ratio"
                ]
            ),

        "dc_load_details":
            json.dumps(
                dc_load_details,
                ensure_ascii=False,
                sort_keys=True,
            ),

        "edge_host_load_details":
            json.dumps(
                edge_host_details,
                ensure_ascii=False,
                sort_keys=True,
            ),
    }

# Episode Energy / Power 统计
def calculate_episode_energy_metrics(env: Any,) -> Dict[str, float]:

    simulation_end_time_s = float(getattr(env,"current_time", 0.0,))

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
        edge_idle_energy_j
        + edge_cpu_dynamic_energy_j
        + edge_gpu_dynamic_energy_j
    )

    system_compute_energy_j = (
        edge_total_energy_j
        + cloud_compute_energy_j
    )

    system_dynamic_compute_energy_j = (
        edge_cpu_dynamic_energy_j
        + edge_gpu_dynamic_energy_j
        + cloud_compute_energy_j
    )

    total_system_energy_j = (
        system_compute_energy_j
        + transfer_energy_j
    )


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

    task_attributable_energy_j = (
        task_compute_energy_j
        + task_transfer_energy_j
    )


    total_jobs = len(
        all_jobs
    )

    completed_jobs = count_completed_jobs(
        env
    )



    # 系统 Transmission 总量应该与 EE + EC 严格一致。
    transfer_split_gap_j = (
        transfer_energy_j
        - edge_edge_transfer_energy_j
        - edge_cloud_transfer_energy_j
    )

    # 系统账本与全部 Job 的 Transmission attribution
    # 正常情况下也应该一致。
    transfer_job_accounting_gap_j = (
        transfer_energy_j
        - task_transfer_energy_j
    )

    # Energy clock 与 Simulation clock 应在 Episode 结束时一致。
    energy_time_gap_s = (
        simulation_end_time_s
        - energy_end_time_s
    )


    return {

        "episode_energy_time_s":
            energy_time_s,

        "energy_time_gap_s":
            float(
                energy_time_gap_s
            ),

        "edge_idle_energy_j":
            edge_idle_energy_j,

        "edge_cpu_dynamic_energy_j":
            edge_cpu_dynamic_energy_j,

        "edge_gpu_dynamic_energy_j":
            edge_gpu_dynamic_energy_j,

        "edge_total_energy_j":
            edge_total_energy_j,

        "cloud_compute_energy_j":
            cloud_compute_energy_j,

        "system_compute_energy_j":
            system_compute_energy_j,

        "system_dynamic_compute_energy_j":
            system_dynamic_compute_energy_j,

        "transfer_energy_j":
            transfer_energy_j,

        "edge_edge_transfer_energy_j":
            edge_edge_transfer_energy_j,

        "edge_cloud_transfer_energy_j":
            edge_cloud_transfer_energy_j,

        "total_system_energy_j":
            total_system_energy_j,

        "total_system_energy_kwh":
            total_system_energy_j
            / 3_600_000.0,


        "edge_idle_avg_power_w":
            safe_ratio(
                edge_idle_energy_j,
                energy_time_s,
            ),

        "edge_cpu_dynamic_avg_power_w":
            safe_ratio(
                edge_cpu_dynamic_energy_j,
                energy_time_s,
            ),

        "edge_gpu_dynamic_avg_power_w":
            safe_ratio(
                edge_gpu_dynamic_energy_j,
                energy_time_s,
            ),

        "edge_total_avg_power_w":
            safe_ratio(
                edge_total_energy_j,
                energy_time_s,
            ),

        "cloud_compute_avg_power_w":
            safe_ratio(
                cloud_compute_energy_j,
                energy_time_s,
            ),

        "system_compute_avg_power_w":
            safe_ratio(
                system_compute_energy_j,
                energy_time_s,
            ),

        # Transmission 模型是离散一次性 Energy，
        # 这里写的是 Episode horizon 上的等效平均能量率，
        # 不是单条链路的瞬时物理功率。
        "transfer_equivalent_avg_power_w":
            safe_ratio(
                transfer_energy_j,
                energy_time_s,
            ),

        "system_total_equivalent_avg_power_w":
            safe_ratio(
                total_system_energy_j,
                energy_time_s,
            ),


        "edge_idle_energy_share":
            safe_ratio(
                edge_idle_energy_j,
                total_system_energy_j,
            ),

        "edge_cpu_dynamic_energy_share":
            safe_ratio(
                edge_cpu_dynamic_energy_j,
                total_system_energy_j,
            ),

        "edge_gpu_dynamic_energy_share":
            safe_ratio(
                edge_gpu_dynamic_energy_j,
                total_system_energy_j,
            ),

        "cloud_compute_energy_share":
            safe_ratio(
                cloud_compute_energy_j,
                total_system_energy_j,
            ),

        "transfer_energy_share":
            safe_ratio(
                transfer_energy_j,
                total_system_energy_j,
            ),


        "system_energy_per_completed_job_j":
            safe_ratio(
                total_system_energy_j,
                completed_jobs,
            ),

        "system_energy_per_total_job_j":
            safe_ratio(
                total_system_energy_j,
                total_jobs,
            ),

        "task_compute_energy_j":
            float(
                task_compute_energy_j
            ),

        "task_transfer_energy_j":
            float(
                task_transfer_energy_j
            ),

        "task_attributable_energy_j":
            float(
                task_attributable_energy_j
            ),

        "task_compute_energy_per_completed_job_j":
            safe_ratio(
                task_compute_energy_j,
                completed_jobs,
            ),

        "task_attributable_energy_per_total_job_j":
            safe_ratio(
                task_attributable_energy_j,
                total_jobs,
            ),


        "transfer_split_gap_j":
            float(
                transfer_split_gap_j
            ),

        "transfer_job_accounting_gap_j":
            float(
                transfer_job_accounting_gap_j
            ),

        "task_edge_edge_transfer_energy_j":
            float(
                task_edge_edge_transfer_energy_j
            ),

        "task_edge_cloud_transfer_energy_j":
            float(
                task_edge_cloud_transfer_energy_j
            ),
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

    edge_dc_ids = [
        str(dc_id)
        for dc_id
        in env.edge_dc_ids
    ]

    base_dc_map = {
        str(dc.dc_id): dc
        for dc
        in env.base_datacenters
    }

    host_ids_per_dc: Dict[
        str,
        list,
    ] = {}

    host_count_per_dc: Dict[
        str,
        int,
    ] = {}

    for dc_id in edge_dc_ids:
        dc = base_dc_map.get(
            dc_id
        )

        if dc is None:
            raise RuntimeError(
                "构造 checkpoint metadata 时 "
                "找不到 Edge DC："
                f"{dc_id}"
            )

        host_ids = [
            str(host.host_id)
            for host
            in dc.host_list
        ]

        host_ids_per_dc[
            dc_id
        ] = host_ids

        host_count_per_dc[
            dc_id
        ] = len(
            host_ids
        )

    host_model_metadata = {
        str(dc_id): {
            "obs_dim": int(
                host_agent.obs_dim
            ),

            "action_dim": int(
                host_agent.action_dim
            ),
        }
        for dc_id, host_agent
        in host_sac_agents.items()
    }

    return {
        "cloud_enabled": bool(
            env.enable_cloud_action
        ),

        # 顺序必须保存。
        # Routing Actor 中 agent one-hot 和 action index
        # 都依赖这些顺序。
        "edge_dc_ids": edge_dc_ids,

        "routing_action_target_dc_ids": [
            str(dc_id)
            for dc_id
            in env.routing_action_target_dc_ids
        ],

        # Host action index 同样依赖 host_list 顺序，
        # 不能只比较 Host 数量。
        "host_count_per_dc":
            host_count_per_dc,

        "host_ids_per_dc":
            host_ids_per_dc,

        "routing": {
            "local_obs_dim": int(
                routing_masac.local_obs_dim
            ),

            "global_state_dim": int(
                routing_masac.global_state_dim
            ),

            "action_dim": int(
                routing_masac.action_dim
            ),

            "num_agents": int(
                routing_masac.num_agents
            ),
        },

        "hosts":
            host_model_metadata,
    }

def validate_checkpoint_structure_metadata(
        checkpoint_metadata:
        Dict[str, Any],

        env: CloudEdgeEnv,

        routing_masac: RoutingMASAC,

        host_sac_agents:
        Dict[str, LocalHostSAC],
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

    if (
            schema_version
            != CHECKPOINT_SCHEMA_VERSION
    ):
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

    if (
            architecture
            != CHECKPOINT_ARCHITECTURE
    ):
        raise RuntimeError(
            "Checkpoint architecture 不兼容："
            f"saved={architecture!r}, "
            f"current="
            f"{CHECKPOINT_ARCHITECTURE!r}"
        )

    saved_structure = (
        checkpoint_metadata.get(
            "structure",
            {}
        )
    )

    current_structure = (
        build_checkpoint_structure_metadata(
            env=env,
            routing_masac=(
                routing_masac
            ),
            host_sac_agents=(
                host_sac_agents
            ),
        )
    )

    # ==========================================================
    # Cloud Action
    #
    # Cloud ON/OFF 会直接改变 Routing action space，
    # 因此不允许跨配置恢复。
    # ==========================================================

    saved_cloud_enabled = bool(
        saved_structure.get(
            "cloud_enabled",
            False,
        )
    )

    current_cloud_enabled = bool(
        current_structure[
            "cloud_enabled"
        ]
    )

    if (
            saved_cloud_enabled
            != current_cloud_enabled
    ):
        raise RuntimeError(
            "Checkpoint Cloud 配置不兼容："
            f"saved={saved_cloud_enabled}, "
            f"current={current_cloud_enabled}"
        )

    # ==========================================================
    # Edge DC identity / order
    # ==========================================================

    saved_edge_dc_ids = list(
        saved_structure.get(
            "edge_dc_ids",
            [],
        )
    )

    current_edge_dc_ids = list(
        current_structure[
            "edge_dc_ids"
        ]
    )

    if (
            saved_edge_dc_ids
            != current_edge_dc_ids
    ):
        raise RuntimeError(
            "Checkpoint Edge DC 列表或顺序不兼容："
            f"saved={saved_edge_dc_ids}, "
            f"current={current_edge_dc_ids}"
        )

    # ==========================================================
    # Routing action semantic mapping
    # ==========================================================

    saved_targets = list(
        saved_structure.get(
            "routing_action_target_dc_ids",
            [],
        )
    )

    current_targets = list(
        current_structure[
            "routing_action_target_dc_ids"
        ]
    )

    if (
            saved_targets
            != current_targets
    ):
        raise RuntimeError(
            "Checkpoint Routing action mapping 不兼容："
            f"saved={saved_targets}, "
            f"current={current_targets}"
        )

    # ==========================================================
    # Routing dimensions
    # ==========================================================

    saved_routing = (
        saved_structure.get(
            "routing",
            {}
        )
    )

    current_routing = (
        current_structure[
            "routing"
        ]
    )

    for field_name in (
        "local_obs_dim",
        "global_state_dim",
        "action_dim",
        "num_agents",
    ):
        if (
                int(
                    saved_routing.get(
                        field_name,
                        -1,
                    )
                )
                != int(
                    current_routing[
                        field_name
                    ]
                )
        ):
            raise RuntimeError(
                "Checkpoint Routing 结构不兼容："
                f"field={field_name}, "
                f"saved="
                f"{saved_routing.get(field_name)}, "
                f"current="
                f"{current_routing[field_name]}"
            )

    # ==========================================================
    # Host physical mapping
    # ==========================================================

    saved_host_ids = (
        saved_structure.get(
            "host_ids_per_dc",
            {}
        )
    )

    current_host_ids = (
        current_structure[
            "host_ids_per_dc"
        ]
    )

    if (
            saved_host_ids
            != current_host_ids
    ):
        raise RuntimeError(
            "Checkpoint Host ID / action mapping "
            "与当前环境不兼容。"
        )

    # ==========================================================
    # Host SAC dimensions
    # ==========================================================

    saved_hosts = (
        saved_structure.get(
            "hosts",
            {}
        )
    )

    current_hosts = (
        current_structure[
            "hosts"
        ]
    )

    if (
            set(saved_hosts.keys())
            != set(current_hosts.keys())
    ):
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
            if (
                    int(
                        saved_hosts[
                            dc_id
                        ].get(
                            field_name,
                            -1,
                        )
                    )
                    != int(
                        current_hosts[
                            dc_id
                        ][
                            field_name
                        ]
                    )
            ):
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

# 根据模型文件路径生成配套的训练器状态 JSON 路径
def checkpoint_state_path(model_path: Path) -> Path:
    return model_path.with_suffix(".trainer.json")

def host_checkpoint_dir(
        routing_model_path: Path,
) -> Path:

    return routing_model_path.with_name(
        f"{routing_model_path.stem}"
        "_hosts"
    )

# 同时保存 MASAC 模型和训练主循环状态
def save_two_layer_checkpoint(
        env: CloudEdgeEnv,

        routing_masac: RoutingMASAC,

        host_sac_agents:
        Dict[str, LocalHostSAC],

        model_path: Path,

        training_stage: TrainingStage,

        train_config: TrainConfig,

        next_episode: int,

        global_decision_steps: int,

        routing_normal_action_steps: int,

        host_training_action_steps:
        Dict[str, int],

        best_episode_return: float,
) -> None:


    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    routing_masac.save(
        model_path
    )

    host_dir = host_checkpoint_dir(model_path)

    host_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for dc_id, host_agent in host_sac_agents.items():
        host_agent.save(
            host_dir / f"{dc_id}.pt"
        )

    checkpoint_metadata = {
        "schema_version": int(
            CHECKPOINT_SCHEMA_VERSION
        ),

        "architecture":
            CHECKPOINT_ARCHITECTURE,

        "bgh_guidance": bgh_guidance_metadata(train_config),

        "saved_training_stage":
            str(
                training_stage.value
            ),

        "structure":
            build_checkpoint_structure_metadata(
                env=env,

                routing_masac=(
                    routing_masac
                ),

                host_sac_agents=(
                    host_sac_agents
                ),
            ),

        # 训练阶段长度保存下来主要用于实验追溯。
        # 不把它作为神经网络结构兼容性的硬约束，
        # 因为后续可能人为延长 Joint Fine-tune。
        "training_schedule": {
            "num_episodes": int(
                train_config.num_episodes
            ),

            "host_pretrain_episodes": int(
                train_config
                    .host_pretrain_episodes
            ),

            "routing_train_episodes": int(
                train_config
                    .routing_train_episodes
            ),

            "joint_finetune_episodes": int(
                train_config
                    .joint_finetune_episodes
            ),
        },

        "trainer_state": {
            "next_episode": int(
                next_episode
            ),

            "global_decision_steps": int(
                global_decision_steps
            ),

            "routing_normal_action_steps":
                int(
                    routing_normal_action_steps
                ),

            # 作为全局诊断量保存。
            "routing_global_steps": int(
                routing_normal_action_steps
            ),

            "host_training_action_steps": {
                str(dc_id): int(
                    step_count
                )
                for dc_id, step_count
                in host_training_action_steps.items()
            },

            "host_global_steps": int(
                sum(
                    int(step_count)
                    for step_count
                    in host_training_action_steps.values()
                )
            ),

            "best_episode_return": float(
                best_episode_return
            ),
        },
    }

    state_path = (
        checkpoint_state_path(
            model_path
        )
    )

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

    # ==========================================================
    # 默认 Host Training Step
    #
    # 只有在“完全没有指定 Resume checkpoint”的情况下，
    # 才允许使用这些默认值开始全新训练。
    #
    # 一旦用户明确指定 checkpoint，
    # 就不允许因为 metadata 缺失而偷偷回到 0。
    # ==========================================================

    default_host_steps = {
        str(dc_id): 0
        for dc_id
        in host_sac_agents.keys()
    }

    # ==========================================================
    # 0. 没有指定 Resume checkpoint
    #
    # 这是唯一允许从 Episode 1 / Step 0 开始的情况。
    # ==========================================================

    if resume_checkpoint is None:
        return (
            1,                  # start_episode
            0,                  # global_decision_steps
            0,                  # routing_normal_action_steps
            default_host_steps,
            float("-inf"),      # best_episode_return
        )

    # ==========================================================
    # 1. Routing checkpoint 路径
    # ==========================================================

    model_path = Path(
        resume_checkpoint
    )

    if not model_path.exists():
        raise FileNotFoundError(
            "找不到 Routing MASAC checkpoint："
            f"{model_path}"
        )

    if not model_path.is_file():
        raise FileNotFoundError(
            "Routing MASAC checkpoint 不是有效文件："
            f"{model_path}"
        )

    # ==========================================================
    # 2. Host checkpoint 目录
    #
    # 例如：
    #
    #   latest.pt
    #
    # 对应：
    #
    #   latest_hosts/
    #       DC1.pt
    #       DC2.pt
    #       ...
    # ==========================================================

    host_dir = host_checkpoint_dir(
        model_path
    )

    if not host_dir.exists():
        raise FileNotFoundError(
            "恢复 Two-Level checkpoint 时找不到 "
            "Local Host SAC checkpoint 目录："
            f"{host_dir}"
        )

    if not host_dir.is_dir():
        raise FileNotFoundError(
            "Local Host SAC checkpoint 路径不是目录："
            f"{host_dir}"
        )

    # ==========================================================
    # 3. 在加载任何模型参数以前，
    #    先检查所有 DC 的 Host checkpoint 是否完整。
    #
    # 不能出现：
    #
    #   Routing 已经 load
    #       ↓
    #   DC3 Host checkpoint 不存在
    #       ↓
    #   当前进程模型进入“半恢复”状态
    #
    # 因此这里首先只检查文件，不修改模型。
    # ==========================================================

    expected_host_paths: Dict[
        str,
        Path,
    ] = {}

    for dc_id in host_sac_agents.keys():

        dc_id = str(
            dc_id
        )

        host_path = (
            host_dir
            / f"{dc_id}.pt"
        )

        expected_host_paths[
            dc_id
        ] = host_path

        if not host_path.exists():
            raise FileNotFoundError(
                "缺少 Local Host SAC checkpoint："
                f"dc={dc_id}, "
                f"path={host_path}"
            )

        if not host_path.is_file():
            raise FileNotFoundError(
                "Local Host SAC checkpoint "
                "不是有效文件："
                f"dc={dc_id}, "
                f"path={host_path}"
            )

    # ==========================================================
    # 4. Trainer Metadata
    #
    # 新 checkpoint 中：
    #
    #   *.trainer.json
    #
    # 不再是“可有可无”的辅助文件，
    # 而是整个 Two-Level checkpoint 的结构声明与
    # 完整写入标志。
    #
    # 因此缺少它时必须 fail-fast。
    # ==========================================================

    state_path = (
        checkpoint_state_path(
            model_path
        )
    )

    if not state_path.exists():
        raise FileNotFoundError(
            "Two-Level checkpoint 缺少必要的 "
            "trainer metadata："
            f"{state_path}"
        )

    if not state_path.is_file():
        raise FileNotFoundError(
            "Trainer metadata 不是有效文件："
            f"{state_path}"
        )

    # ==========================================================
    # 5. 读取完整 checkpoint metadata
    # ==========================================================

    try:
        checkpoint_metadata = json.loads(
            state_path.read_text(
                encoding="utf-8"
            )
        )
    except (
            json.JSONDecodeError,
            OSError,
    ) as exc:
        raise RuntimeError(
            "无法读取 Two-Level checkpoint "
            "trainer metadata："
            f"{state_path}"
        ) from exc

    if not isinstance(
            checkpoint_metadata,
            dict,
    ):
        raise RuntimeError(
            "Two-Level checkpoint metadata "
            "根节点必须是 dict："
            f"{state_path}"
        )

    if train_config.enable_bayesian_game:
        if checkpoint_metadata.get("bgh_guidance") != bgh_guidance_metadata(train_config):
            raise RuntimeError(
                "Checkpoint BGH guidance semantics/configuration differ from this run. "
                "Use matching short-window parameters or start a new training run; "
                "legacy guidance checkpoints cannot silently resume as v3.")

    # ==========================================================
    # 6. 验证 checkpoint 的结构身份
    #
    # validate_checkpoint_structure_metadata() 应检查：
    #
    #   schema_version
    #   architecture
    #   cloud_enabled
    #   edge_dc_ids
    #   routing_action_target_dc_ids
    #   Routing observation/state/action dimensions
    #   num_agents
    #   host_ids_per_dc
    #   每个 Host SAC obs_dim/action_dim
    #
    # 注意：
    # 这里仍然没有真正 load_state_dict()。
    # ==========================================================

    validate_checkpoint_structure_metadata(
        checkpoint_metadata=(
            checkpoint_metadata
        ),

        env=env,

        routing_masac=(
            routing_masac
        ),

        host_sac_agents=(
            host_sac_agents
        ),
    )

    # ==========================================================
    # 7. 检查 saved_training_stage
    #
    # 它主要用于：
    #   - checkpoint provenance；
    #   - Resume 日志；
    #   - 判断 checkpoint 是在哪个训练阶段产生的。
    #
    # Stage 本身不在这里强制要求与当前 TrainConfig
    # 完全相同，因为后续可能人为延长 Joint Fine-tune。
    # ==========================================================

    saved_training_stage_raw = (
        checkpoint_metadata.get(
            "saved_training_stage"
        )
    )

    if saved_training_stage_raw is None:
        raise RuntimeError(
            "Checkpoint metadata 缺少 "
            "saved_training_stage。"
        )

    try:
        saved_training_stage = (
            TrainingStage(
                str(
                    saved_training_stage_raw
                )
            )
        )
    except ValueError as exc:
        raise RuntimeError(
            "Checkpoint 中存在未知 Training Stage："
            f"{saved_training_stage_raw!r}"
        ) from exc

    # ==========================================================
    # 8. 检查 Training Schedule metadata
    #
    # Schedule 主要用于实验追溯。
    #
    # 不作为网络结构的硬兼容条件：
    # 例如可以在已有 checkpoint 基础上延长
    # JOINT_FINETUNE_EPISODES。
    #
    # 如果与当前配置不同，只给出明确警告。
    # ==========================================================

    saved_training_schedule = (
        checkpoint_metadata.get(
            "training_schedule"
        )
    )

    if not isinstance(
            saved_training_schedule,
            dict,
    ):
        raise RuntimeError(
            "Checkpoint metadata 缺少有效的 "
            "training_schedule。"
        )

    current_training_schedule = {
        "num_episodes": int(
            train_config.num_episodes
        ),

        "host_pretrain_episodes": int(
            train_config
                .host_pretrain_episodes
        ),

        "routing_train_episodes": int(
            train_config
                .routing_train_episodes
        ),

        "joint_finetune_episodes": int(
            train_config
                .joint_finetune_episodes
        ),
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

    if (
            saved_schedule_normalized
            != current_training_schedule
    ):
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

    # ==========================================================
    # 9. Trainer State 必须完整存在
    # ==========================================================

    trainer_state = (
        checkpoint_metadata.get(
            "trainer_state"
        )
    )

    if not isinstance(
            trainer_state,
            dict,
    ):
        raise RuntimeError(
            "Checkpoint metadata 缺少有效的 "
            "trainer_state。"
        )

    required_trainer_fields = {
        "next_episode",
        "global_decision_steps",
        "routing_normal_action_steps",
        "host_training_action_steps",
        "best_episode_return",
    }

    missing_trainer_fields = (
        required_trainer_fields
        - set(
            trainer_state.keys()
        )
    )

    if missing_trainer_fields:
        raise RuntimeError(
            "Checkpoint trainer_state 缺少必要字段："
            f"{sorted(missing_trainer_fields)}"
        )

    # ==========================================================
    # 10. 先解析 Trainer counters。
    #
    # 仍然没有加载模型。
    #
    # 目的是保证 metadata 有问题时，
    # 当前 Routing / Host 网络保持原始初始化状态。
    # ==========================================================

    start_episode = int(
        trainer_state[
            "next_episode"
        ]
    )

    global_decision_steps = int(
        trainer_state[
            "global_decision_steps"
        ]
    )

    routing_normal_action_steps = int(
        trainer_state[
            "routing_normal_action_steps"
        ]
    )

    best_episode_return = float(
        trainer_state[
            "best_episode_return"
        ]
    )

    # ==========================================================
    # next_episode 合法性检查
    #
    # num_episodes + 1 是允许的：
    #
    #   例如加载已经完成 Episode 1000 的 final checkpoint，
    #   next_episode 可以为 1001。
    # ==========================================================

    if start_episode < 1:
        raise RuntimeError(
            "Checkpoint next_episode 非法："
            f"{start_episode}"
        )

    if (
            start_episode
            > int(
                train_config.num_episodes
            ) + 1
    ):
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

    # ==========================================================
    # 11. Host Training Steps
    #
    # 每个 DC 的 counter 都必须存在。
    #
    # 不能像旧实现一样：
    #
    #   missing -> 0
    #
    # 因为这样 Resume 后某个 Host SAC 会错误地
    # 重新进入 random warmup。
    # ==========================================================

    saved_host_steps = (
        trainer_state[
            "host_training_action_steps"
        ]
    )

    if not isinstance(
            saved_host_steps,
            dict,
    ):
        raise RuntimeError(
            "Checkpoint "
            "host_training_action_steps "
            "必须是 dict。"
        )

    expected_host_dc_ids = {
        str(dc_id)
        for dc_id
        in host_sac_agents.keys()
    }

    saved_host_dc_ids = {
        str(dc_id)
        for dc_id
        in saved_host_steps.keys()
    }

    if (
            saved_host_dc_ids
            != expected_host_dc_ids
    ):
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

    for dc_id in sorted(
            expected_host_dc_ids
    ):

        step_count = int(
            saved_host_steps[
                dc_id
            ]
        )

        if step_count < 0:
            raise RuntimeError(
                "Checkpoint Host training step "
                "不能为负数："
                f"dc={dc_id}, "
                f"steps={step_count}"
            )

        host_training_action_steps[
            dc_id
        ] = step_count

    # ==========================================================
    # 到这里为止：
    #
    #   文件完整性
    #   Metadata
    #   Schema
    #   Environment structure
    #   Routing structure
    #   Host structure
    #   Training stage
    #   Trainer counters
    #
    # 已经全部验证成功。
    #
    # 从下面开始才允许真正修改当前模型参数。
    # ==========================================================

    # ==========================================================
    # 12. 加载 Routing MASAC
    # ==========================================================

    routing_masac.load(
        file_path=(
            model_path
        ),

        load_optimizers=True,
    )

    # ==========================================================
    # 13. 加载全部 Local Host SAC
    # ==========================================================

    for dc_id, host_agent in (
        host_sac_agents.items()
    ):

        dc_id = str(
            dc_id
        )

        host_path = (
            expected_host_paths[
                dc_id
            ]
        )

        host_agent.load(
            host_path,

            load_optimizers=True,
        )

    # ==========================================================
    # 14. Resume 成功信息
    # ==========================================================

    print(
        "\n"
        "============================================================\n"
        "✅ Two-Level BGH-MASAC checkpoint 恢复成功\n"
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

    # ==========================================================
    # 15. 返回 Trainer Resume State
    # ==========================================================

    return (
        start_episode,
        global_decision_steps,
        routing_normal_action_steps,
        host_training_action_steps,
        best_episode_return,
    )

# 写训练日志用的
def build_timestamped_log_path(
        base_log_path: str,
        run_start_time: str,
) -> Path:
    """
    根据统一 run timestamp 生成一份日志路径。
    """

    base_path = Path(
        base_log_path
    )

    if not base_path.is_absolute():

        base_path = (
            PROJECT_ROOT
            / base_path
        )

    base_path = (
        base_path.resolve()
    )

    log_file_name = (
        f"{base_path.stem}_"
        f"{run_start_time}"
        f"{base_path.suffix}"
    )

    return (
        base_path.parent
        / log_file_name
    )


def build_run_log_paths(
        episode_base_log_path: str,
        dc_base_log_path: str,
) -> Tuple[Path, Path]:
    """
    为同一次训练生成 Episode Log 和 DC Log。

    两个 CSV 必须共用完全相同的 timestamp，
    才能保证离线分析时可以一一匹配。
    """

    run_start_time = (
        datetime.now()
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    episode_log_path = (
        build_timestamped_log_path(
            base_log_path=(
                episode_base_log_path
            ),

            run_start_time=(
                run_start_time
            ),
        )
    )

    dc_log_path = (
        build_timestamped_log_path(
            base_log_path=(
                dc_base_log_path
            ),

            run_start_time=(
                run_start_time
            ),
        )
    )

    return (
        episode_log_path,
        dc_log_path,
    )

# 把一个 episode 的统计信息追加到 CSV 文件
def append_csv_log(csv_path: Path, row: Dict[str, Any],) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # 判断文件是否已经存在且不是空文件。
    write_header = (
            not csv_path.exists()
            or csv_path.stat().st_size == 0
    )

    # 以追加模式打开文件。
    with csv_path.open(
            mode="a",
            newline="",
            encoding="utf-8-sig",
    ) as csv_file:
        # 使用当前 row 的键作为固定列名。
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(row.keys()),
        )

        # 新文件先写入表头。
        if write_header:
            writer.writeheader()

        # 写入当前 episode 数据。
        writer.writerow(row)
        csv_file.flush()
        os.fsync(csv_file.fileno())

# 把 episode 统计整理成固定结构的 CSV 行
def build_episode_log_row(
        stats: EpisodeStatistics,
        env: Any,

        routing_replay_buffer:
        RoutingReplayBuffer,

        host_replay_buffers:
        Dict[str, HostReplayBuffer],

        routing_masac:
        RoutingMASAC,

        host_sac_agents:
        Dict[str, LocalHostSAC],

        host_training_action_steps:
        Dict[str, int],

        pending_trace_store:
        PendingJobTraceStore,
        neighbor_feedback_store:NeighborHistoricalFeedbackStore,

        global_decision_steps: int,

        routing_normal_action_steps: int,

        wall_time_seconds: float,

        service_metrics:
        Dict[str, Any],

        energy_metrics:
        Dict[str, Any],

        load_metrics:
        Dict[str, Any],
) -> Dict[str, Any]:
    """
    构造 BGH-MASAC Episode-level CSV。

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

    completed_jobs = int(
        count_completed_jobs(
            env
        )
    )

    dropped_jobs = int(
        len(
            getattr(
                env,
                "dropped_jobs_info",
                [],
            )
        )
    )

    unresolved_jobs = int(
        total_jobs
        - completed_jobs
        - dropped_jobs
    )

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

    remaining_waiting_jobs = int(
        count_waiting_jobs(
            env
        )
    )

    host_alpha_values = [
        float(
            host_agent.alpha
            .detach()
            .cpu()
            .item()
        )
        for host_agent
        in host_sac_agents.values()
    ]

    host_alpha_mean = (
        float(
            np.mean(
                host_alpha_values
            )
        )
        if host_alpha_values
        else float("nan")
    )

    host_update_step_total = int(
        sum(
            int(
                host_agent.update_step
            )
            for host_agent
            in host_sac_agents.values()
        )
    )

    host_training_steps_total = int(
        sum(
            int(step_count)
            for step_count
            in host_training_action_steps.values()
        )
    )

    neighbor_feedback_summary = (
        neighbor_feedback_store.summary()
    )

    # ==========================================================
    # Explicit Two-Level Log Columns
    # ==============================================================

    row: Dict[str, Any] = {
        # ------------------------------------------------------
        # Run Identity
        # ------------------------------------------------------

        "episode":
            int(
                stats.episode
            ),

        "episode_seed":
            int(
                stats.episode_seed
            ),

        "training_stage":
            str(
                stats.training_stage
            ),
        "bgh_bayesian_game_enabled":
            bool(
                conf.BGH_ENABLE_BAYESIAN_GAME
            ),

        "bgh_heuristic_guidance_enabled":
            bool(
                conf.BGH_ENABLE_HEURISTIC_GUIDANCE
            ),

        # 第 8 步：记录当前 Episode 实际使用的 Guidance λ，
        # 便于核对三阶段边界和复现实验。
        "bgh_guidance_lambda":
            float(
                stats.guidance_lambda
            ),

        "bgh_zero_diff_mode":
            bool(
                not conf.BGH_ENABLE_BAYESIAN_GAME
                and not conf.BGH_ENABLE_HEURISTIC_GUIDANCE
            ),

        "cloud_enabled":
            bool(
                getattr(
                    env,
                    "enable_cloud_action",
                    False,
                )
            ),
        "neighbor_feedback_collection_enabled":
            bool(
                conf
                    .COLLECT_NEIGHBOR_HISTORICAL_FEEDBACK
            ),

        "neighbor_feedback_decision_enabled":
            bool(
                conf
                    .USE_NEIGHBOR_HISTORICAL_FEEDBACK
            ),

        # ------------------------------------------------------
        # 当前 Episode 内，
        # 有多少 terminal Job 被 Feedback Store 消费。
        #
        # 正常情况下最终应该接近本 Episode 的 total_jobs。
        # ------------------------------------------------------

        "neighbor_feedback_episode_terminal_jobs_seen":
            int(
                neighbor_feedback_summary[
                    "episode_terminal_jobs_seen"
                ]
            ),

        # ------------------------------------------------------
        # 当前 Episode 内真正产生多少条：
        #
        #   source Edge DC -> target Edge DC
        #
        # Historical Feedback Sample。
        #
        # 一个多跳 Job 可以产生多条 pair sample。
        # ------------------------------------------------------

        "neighbor_feedback_episode_pair_samples":
            int(
                neighbor_feedback_summary[
                    "episode_pair_samples"
                ]
            ),

        # ------------------------------------------------------
        # 从训练开始到当前 Episode，
        # Feedback Store 一共消费了多少 terminal Job。
        #
        # 这是跨 Episode 累积值。
        # ------------------------------------------------------

        "neighbor_feedback_total_terminal_jobs_seen":
            int(
                neighbor_feedback_summary[
                    "terminal_jobs_seen"
                ]
            ),

        # ------------------------------------------------------
        # 从训练开始到当前 Episode，
        # 一共形成了多少 source->target 历史样本。
        #
        # 同样是跨 Episode 累积值。
        # ------------------------------------------------------

        "neighbor_feedback_total_pair_samples":
            int(
                neighbor_feedback_summary[
                    "total_pair_samples"
                ]
            ),

        # ------------------------------------------------------
        # 当前已有历史数据的有向 pair 数量。
        #
        # 例如：
        #
        #   DC1 -> DC2
        #   DC1 -> DC3
        #   DC2 -> DC5
        #
        # 分别算 3 个 active pair。
        # ------------------------------------------------------

        "neighbor_feedback_active_pair_count":
            int(
                neighbor_feedback_summary[
                    "active_pair_count"
                ]
            ),

        "wall_time_seconds":
            float(
                wall_time_seconds
            ),

        "simulation_end_time":
            float(
                getattr(
                    env,
                    "current_time",
                    0.0,
                )
            ),

        # ------------------------------------------------------
        # System Objective
        #
        # 不把 Host layer reward 再加一次。
        # ------------------------------------------------------

        "training_episode_reward":
            float(
                stats.episode_return
            ),

        "total_jobs":
            total_jobs,

        "completed_jobs":
            completed_jobs,

        "dropped_jobs":
            dropped_jobs,

        "unresolved_jobs":
            unresolved_jobs,

        "completion_rate":
            safe_ratio(
                completed_jobs,
                total_jobs,
            ),

        "drop_rate":
            safe_ratio(
                dropped_jobs,
                total_jobs,
            ),

        "queued_jobs":
            queued_jobs,

        "queue_admission_rate":
            safe_ratio(
                queued_jobs,
                total_jobs,
            ),

        "started_from_waiting_jobs":
            started_from_waiting_jobs,

        "waiting_timeout_drops":
            waiting_timeout_drops,

        "waiting_timeout_drop_rate":
            safe_ratio(
                waiting_timeout_drops,
                total_jobs,
            ),

        "max_waiting_queue_length":
            max_waiting_queue_length,

        "remaining_waiting_jobs":
            remaining_waiting_jobs,

        # ------------------------------------------------------
        # Reward / Energy Configuration
        # ------------------------------------------------------

        "energy_normalization_j":
            float(
                conf
                .ENERGY_NORMALIZATION_J
            ),

        "energy_cost_weight":
            float(
                conf
                .ENERGY_COST_WEIGHT
            ),

        "energy_optimization_enabled":
            bool(
                float(
                    conf
                    .ENERGY_COST_WEIGHT
                )
                > 0.0
            ),

        # ------------------------------------------------------
        # Routing Behavior
        # ------------------------------------------------------

        "routing_decision_count":
            int(
                stats
                .routing_decision_count
            ),

        "routing_actor_controlled_count":
            int(
                stats.routing_random_action_count
                + stats.routing_policy_action_count
            ),

        "routing_forced_action_count":
            int(
                stats
                .routing_forced_action_count
            ),

        "routing_orchestrator_action_count":
            int(
                stats
                .routing_orchestrator_action_count
            ),

        "routing_random_action_count":
            int(
                stats
                .routing_random_action_count
            ),

        "routing_policy_action_count":
            int(
                stats
                .routing_policy_action_count
            ),

        "routing_self_count":
            int(
                stats.routing_self_count
            ),

        "routing_self_rate":
            safe_ratio(
                stats.routing_self_count,
                stats.routing_decision_count,
            ),

        "routing_edge_count":
            int(
                stats.routing_edge_count
            ),

        "routing_edge_rate":
            safe_ratio(
                stats.routing_edge_count,
                stats.routing_decision_count,
            ),

        "routing_cloud_count":
            int(
                stats.routing_cloud_count
            ),

        "routing_cloud_rate":
            safe_ratio(
                stats.routing_cloud_count,
                stats.routing_decision_count,
            ),

        "routing_drop_count":
            int(
                stats.routing_drop_count
            ),

        "routing_drop_rate":
            safe_ratio(
                stats.routing_drop_count,
                stats.routing_decision_count,
            ),

        "multi_hop_job_count":
            int(
                stats.multi_hop_job_count
            ),

        "multi_hop_job_rate":
            safe_ratio(
                stats.multi_hop_job_count,
                stats.terminal_trace_flushed_count,
            ),

        "avg_routing_edge_hops_per_job":
            safe_ratio(
                stats.routing_edge_hop_total,
                stats.terminal_trace_flushed_count,
            ),

        "max_routing_hops":
            int(
                stats.max_routing_hops
            ),

        "p95_routing_edge_hops":
            float(np.percentile(stats.routing_edge_hops_by_job, 95))
            if stats.routing_edge_hops_by_job else 0.0,

        "first_edge_absorption_count":
            int(stats.first_edge_absorbed_count),

        "first_edge_absorption_rate":
            safe_ratio(stats.first_edge_absorbed_count, stats.first_edge_successor_total),

        "edge_successor_total":
            int(stats.edge_successor_total),

        "edge_target_absorption_count":
            int(stats.edge_successor_self_count),

        "edge_target_absorption_rate":
            safe_ratio(stats.edge_successor_self_count, stats.edge_successor_total),

        "edge_to_edge_reforward_count":
            int(stats.edge_successor_edge_count),

        "edge_to_edge_reforward_rate":
            safe_ratio(stats.edge_successor_edge_count, stats.edge_successor_total),

        "edge_to_cloud_transit_count":
            int(stats.edge_successor_cloud_count),

        "edge_to_cloud_transit_rate":
            safe_ratio(stats.edge_successor_cloud_count, stats.edge_successor_total),

        "edge_target_drop_count":
            int(stats.edge_successor_drop_count),

        "edge_target_drop_rate":
            safe_ratio(stats.edge_successor_drop_count, stats.edge_successor_total),

        "absorption_pair_counts_json":
            json.dumps(stats.absorption_pair_counts, ensure_ascii=False, sort_keys=True),

        "routing_source_target_matrix_json":
            json.dumps(
                stats
                .routing_source_target_counts,
                ensure_ascii=False,
                sort_keys=True,
            ),

        "routing_reward_by_agent_json":
            json.dumps(
                stats.per_agent_returns,
                ensure_ascii=False,
                sort_keys=True,
            ),

        # ------------------------------------------------------
        # Host Behavior
        # ------------------------------------------------------

        "host_decision_count":
            int(
                stats.host_decision_count
            ),

        "host_random_action_count":
            int(
                stats
                .host_random_action_count
            ),

        "host_policy_action_count":
            int(
                stats
                .host_policy_action_count
            ),

        "host_started_count":
            int(
                stats.host_started_count
            ),

        "host_started_rate":
            safe_ratio(
                stats.host_started_count,
                stats.host_decision_count,
            ),

        "host_queued_count":
            int(
                stats.host_queued_count
            ),

        "host_queued_rate":
            safe_ratio(
                stats.host_queued_count,
                stats.host_decision_count,
            ),

        "host_dropped_count":
            int(
                stats.host_dropped_count
            ),

        "host_dropped_rate":
            safe_ratio(
                stats.host_dropped_count,
                stats.host_decision_count,
            ),

        # ------------------------------------------------------
        # Routing MASAC Learning
        # ------------------------------------------------------

        "routing_episode_updates":
            int(
                stats.routing_update_count
            ),

        "routing_replay_size":
            int(
                len(
                    routing_replay_buffer
                )
            ),

        "routing_replay_trainable_size":
            int(
                routing_replay_buffer
                .num_trainable_actions
            ),

        "routing_update_step":
            int(
                routing_masac.update_step
            ),

        "routing_global_decision_steps":
            int(
                global_decision_steps
            ),

        "routing_training_action_steps":
            int(
                routing_normal_action_steps
            ),

        "routing_critic_loss":
            stats.mean_routing_metric(
                "critic_loss"
            ),

        "routing_q1_loss":
            stats.mean_routing_metric(
                "q1_loss"
            ),

        "routing_q2_loss":
            stats.mean_routing_metric(
                "q2_loss"
            ),

        "routing_actor_loss":
            stats.mean_routing_metric(
                "actor_loss"
            ),

        "routing_alpha_loss":
            stats.mean_routing_metric(
                "alpha_loss"
            ),

        "routing_alpha":
            float(
                routing_masac.alpha
                .detach()
                .cpu()
                .item()
            ),

        "routing_policy_entropy":
            stats.mean_routing_metric(
                "policy_entropy"
            ),

        "routing_target_entropy":
            stats.mean_routing_metric(
                "target_entropy"
            ),

        "routing_mean_q1":
            stats.mean_routing_metric(
                "mean_q1"
            ),

        "routing_mean_q2":
            stats.mean_routing_metric(
                "mean_q2"
            ),

        "routing_mean_target_q":
            stats.mean_routing_metric(
                "mean_target_q"
            ),

        # ------------------------------------------------------
        # Host SAC Aggregate Learning
        # ------------------------------------------------------

        "host_episode_updates":
            int(
                stats.host_update_count
            ),

        "host_replay_size_total":
            int(
                sum(
                    len(buffer)
                    for buffer
                    in host_replay_buffers.values()
                )
            ),

        "host_training_action_steps_total":
            host_training_steps_total,

        "host_update_step_total":
            host_update_step_total,

        "host_critic_loss":
            stats.mean_host_metric(
                "critic_loss"
            ),

        "host_q1_loss":
            stats.mean_host_metric(
                "q1_loss"
            ),

        "host_q2_loss":
            stats.mean_host_metric(
                "q2_loss"
            ),

        "host_actor_loss":
            stats.mean_host_metric(
                "actor_loss"
            ),

        "host_alpha_loss":
            stats.mean_host_metric(
                "alpha_loss"
            ),

        "host_alpha_mean":
            host_alpha_mean,

        "host_policy_entropy":
            stats.mean_host_metric(
                "policy_entropy"
            ),

        "host_target_entropy":
            stats.mean_host_metric(
                "target_entropy"
            ),

        "host_mean_q1":
            stats.mean_host_metric(
                "mean_q1"
            ),

        "host_mean_q2":
            stats.mean_host_metric(
                "mean_q2"
            ),

        "host_mean_target_q":
            stats.mean_host_metric(
                "mean_target_q"
            ),

        # ------------------------------------------------------
        # Layer Reward Diagnostics
        #
        # 禁止做：
        # system_episode_reward =
        # routing_layer_reward_sum + host_layer_reward_sum
        # ------------------------------------------------------

        "routing_layer_reward_sum":
            float(
                stats.routing_layer_reward_sum
            ),

        "host_layer_reward_sum":
            float(
                stats.host_layer_reward_sum
            ),

        # ------------------------------------------------------
        # Causal Trace Health
        # ------------------------------------------------------

        "terminal_trace_flushed_count":
            int(
                stats
                .terminal_trace_flushed_count
            ),

        "routing_transition_flushed_count":
            int(
                stats
                .routing_transition_flushed_count
            ),

        "host_transition_flushed_count":
            int(
                stats
                .host_transition_flushed_count
            ),

        "pending_trace_count_end":
            int(
                pending_trace_store
                .pending_trace_count
            ),

        "finalized_trace_count_end":
            int(
                pending_trace_store
                .finalized_trace_count
            ),

        "causal_terminal_job_gap":
            int(
                total_jobs
                - stats
                .terminal_trace_flushed_count
            ),
    }

    # ==========================================================
    # Existing System Metrics
    #
    # 这些属于真实 workload/service/load/energy，
    # 与 Routing / Host 学习器无关，因此继续保留。
    # ==============================================================

    row.update(
        service_metrics
    )

    row.update(
        energy_metrics
    )

    # DC/Host 详细 JSON 移到 dc_log.csv。
    # Episode Log 只保留 aggregate load metrics。
    for metric_name, metric_value in (
            load_metrics.items()
    ):

        if metric_name in {
            "dc_load_details",
            "edge_host_load_details",
        }:
            continue

        row[
            metric_name
        ] = metric_value

    return row


def build_dc_log_rows(
        stats: EpisodeStatistics,
        env: Any,

        host_sac_agents:
        Dict[str, LocalHostSAC],

        host_replay_buffers:
        Dict[str, HostReplayBuffer],

        host_training_action_steps:
        Dict[str, int],

        load_metrics:
        Dict[str, Any],

        neighbor_feedback_store:
        NeighborHistoricalFeedbackStore,
) -> list[Dict[str, Any]]:
    """
    构造 BGH-MASAC DC-level 日志。

    每个 Episode：
        每个 Edge DC 产生一行。

    用于分别诊断：
        Routing Agent 行为
        Local Host SAC
        当前 DC Resource Load
    """

    raw_dc_load_details = (
        load_metrics.get(
            "dc_load_details",
            "{}",
        )
    )

    raw_host_load_details = (
        load_metrics.get(
            "edge_host_load_details",
            "{}",
        )
    )

    dc_load_details = (
        json.loads(
            raw_dc_load_details
        )
        if isinstance(
            raw_dc_load_details,
            str,
        )
        else dict(
            raw_dc_load_details
        )
    )

    host_load_details = (
        json.loads(
            raw_host_load_details
        )
        if isinstance(
            raw_host_load_details,
            str,
        )
        else dict(
            raw_host_load_details
        )
    )

    rows: list[
        Dict[str, Any]
    ] = []

    for dc_id_raw in (
        env.edge_dc_ids
    ):

        dc_id = str(
            dc_id_raw
        )
        neighbor_feedback_source_summary = (
            neighbor_feedback_store
                .source_summary(
                dc_id
            )
        )

        neighbor_feedback_snapshot = (
            neighbor_feedback_store
                .snapshot_for_source(
                dc_id
            )
        )

        dc_stats = (
            stats.dc_counters.get(
                dc_id,
                {},
            )
        )

        dc_load = (
            dc_load_details.get(
                dc_id,
                {},
            )
        )

        host_agent = (
            host_sac_agents[
                dc_id
            ]
        )

        host_replay = (
            host_replay_buffers[
                dc_id
            ]
        )

        dc_host_load_details = {
            host_key:
                host_detail

            for host_key, host_detail
            in host_load_details.items()

            if str(
                host_key
            ).startswith(
                f"{dc_id}/"
            )
        }

        source_target_counts = (
            stats
            .routing_source_target_counts
            .get(
                dc_id,
                {},
            )
        )

        row = {
            # --------------------------------------------------
            # Identity
            # --------------------------------------------------

            "episode":
                int(
                    stats.episode
                ),

            "episode_seed":
                int(
                    stats.episode_seed
                ),

            "training_stage":
                str(
                    stats.training_stage
                ),

            "dc_id":
                dc_id,

            # --------------------------------------------------
            # Exogenous Workload Arrival
            # --------------------------------------------------

            "arrival_mode":
                str(
                    dc_load.get(
                        "arrival_mode",
                        getattr(
                            env,
                            "dc_arrival_mode",
                            "uniform",
                        ),
                    )
                ),

            "arrival_role":
                str(
                    dc_load.get(
                        "arrival_role",
                        "uniform",
                    )
                ),

            "configured_arrival_weight":
                float(
                    dc_load.get(
                        "configured_arrival_weight",
                        1.0,
                    )
                ),

            "configured_arrival_probability":
                float(
                    dc_load.get(
                        "configured_arrival_probability",
                        0.0,
                    )
                ),

            "configured_arrival_rate":
                float(
                    dc_load.get(
                        "configured_arrival_rate",
                        0.0,
                    )
                ),

            "expected_exogenous_arrival_count":
                float(
                    dc_load.get(
                        "expected_exogenous_arrival_count",
                        0.0,
                    )
                ),

            "exogenous_arrival_count":
                int(
                    dc_load.get(
                        "exogenous_arrival_count",
                        0,
                    )
                ),

            "exogenous_arrival_share":
                float(
                    dc_load.get(
                        "exogenous_arrival_share",
                        0.0,
                    )
                ),

            "observed_exogenous_arrival_rate":
                float(
                    dc_load.get(
                        "observed_exogenous_arrival_rate",
                        0.0,
                    )
                ),

            # --------------------------------------------------
            # Routing Behavior
            # --------------------------------------------------

            "routing_decisions":
                int(
                    dc_stats.get(
                        "routing_decisions",
                        0,
                    )
                ),

            "route_self_count":
                int(
                    dc_stats.get(
                        "route_self_count",
                        0,
                    )
                ),

            "route_out_edge_count":
                int(
                    dc_stats.get(
                        "route_out_edge_count",
                        0,
                    )
                ),

            "route_in_edge_count":
                int(
                    dc_stats.get(
                        "route_in_edge_count",
                        0,
                    )
                ),

            "incoming_edge_successor_count":
                int(dc_stats.get("incoming_edge_successor_count", 0)),

            "incoming_edge_absorbed_count":
                int(dc_stats.get("incoming_edge_self_count", 0)),

            "incoming_edge_absorption_rate":
                safe_ratio(dc_stats.get("incoming_edge_self_count", 0),
                           dc_stats.get("incoming_edge_successor_count", 0)),

            "incoming_edge_reforward_count":
                int(dc_stats.get("incoming_edge_edge_count", 0)),

            "incoming_edge_cloud_count":
                int(dc_stats.get("incoming_edge_cloud_count", 0)),

            "incoming_edge_drop_count":
                int(dc_stats.get("incoming_edge_drop_count", 0)),

            "net_edge_migration_count":
                int(
                    dc_stats.get(
                        "route_in_edge_count",
                        0,
                    )
                    - dc_stats.get(
                        "route_out_edge_count",
                        0,
                    )
                ),

            "route_cloud_count":
                int(
                    dc_stats.get(
                        "route_cloud_count",
                        0,
                    )
                ),

            "route_drop_count":
                int(
                    dc_stats.get(
                        "route_drop_count",
                        0,
                    )
                ),

            "routing_reward":
                float(
                    stats
                    .per_agent_returns
                    .get(
                        dc_id,
                        0.0,
                    )
                ),

            "routing_out_targets_json":
                json.dumps(
                    source_target_counts,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            "neighbor_feedback_outgoing_pair_count":
                int(
                    neighbor_feedback_source_summary[
                        "outgoing_pair_count"
                    ]
                ),

            "neighbor_feedback_outgoing_sample_count":
                int(
                    neighbor_feedback_source_summary[
                        "outgoing_sample_count"
                    ]
                ),

            "neighbor_feedback_snapshot_json":
                json.dumps(
                    neighbor_feedback_snapshot,
                    ensure_ascii=False,
                    sort_keys=True,
                ),

            # --------------------------------------------------
            # Host Behavior
            # --------------------------------------------------

            "host_decision_count":
                int(
                    dc_stats.get(
                        "host_decisions",
                        0,
                    )
                ),

            "host_random_count":
                int(
                    dc_stats.get(
                        "host_random_count",
                        0,
                    )
                ),

            "host_policy_count":
                int(
                    dc_stats.get(
                        "host_policy_count",
                        0,
                    )
                ),

            "host_started_count":
                int(
                    dc_stats.get(
                        "host_started_count",
                        0,
                    )
                ),

            "host_queued_count":
                int(
                    dc_stats.get(
                        "host_queued_count",
                        0,
                    )
                ),

            "host_dropped_count":
                int(
                    dc_stats.get(
                        "host_dropped_count",
                        0,
                    )
                ),

            # --------------------------------------------------
            # Local Host SAC
            # --------------------------------------------------

            "host_episode_updates":
                int(
                    dc_stats.get(
                        "host_updates",
                        0,
                    )
                ),

            "host_training_action_steps":
                int(
                    host_training_action_steps.get(
                        dc_id,
                        0,
                    )
                ),

            "host_replay_size":
                int(
                    len(
                        host_replay
                    )
                ),

            "host_update_step":
                int(
                    host_agent.update_step
                ),

            "host_critic_loss":
                stats.mean_host_metric(
                    "critic_loss",
                    dc_id=dc_id,
                ),

            "host_q1_loss":
                stats.mean_host_metric(
                    "q1_loss",
                    dc_id=dc_id,
                ),

            "host_q2_loss":
                stats.mean_host_metric(
                    "q2_loss",
                    dc_id=dc_id,
                ),

            "host_actor_loss":
                stats.mean_host_metric(
                    "actor_loss",
                    dc_id=dc_id,
                ),

            "host_alpha_loss":
                stats.mean_host_metric(
                    "alpha_loss",
                    dc_id=dc_id,
                ),

            "host_alpha":
                float(
                    host_agent.alpha
                    .detach()
                    .cpu()
                    .item()
                ),

            "host_policy_entropy":
                stats.mean_host_metric(
                    "policy_entropy",
                    dc_id=dc_id,
                ),

            "host_target_entropy":
                stats.mean_host_metric(
                    "target_entropy",
                    dc_id=dc_id,
                ),

            "host_mean_q1":
                stats.mean_host_metric(
                    "mean_q1",
                    dc_id=dc_id,
                ),

            "host_mean_q2":
                stats.mean_host_metric(
                    "mean_q2",
                    dc_id=dc_id,
                ),

            "host_mean_target_q":
                stats.mean_host_metric(
                    "mean_target_q",
                    dc_id=dc_id,
                ),

            # --------------------------------------------------
            # DC Resource Load
            # --------------------------------------------------

            "dc_completed_jobs":
                int(
                    dc_load.get(
                        "completed_jobs",
                        0,
                    )
                ),

            "dc_cpu_capacity":
                float(
                    dc_load.get(
                        "cpu_capacity",
                        0.0,
                    )
                ),

            "dc_gpu_capacity":
                float(
                    dc_load.get(
                        "gpu_capacity",
                        0.0,
                    )
                ),

            "dc_avg_cpu_load":
                float(
                    dc_load.get(
                        "avg_cpu_load",
                        0.0,
                    )
                ),

            "dc_peak_cpu_load":
                float(
                    dc_load.get(
                        "peak_cpu_load",
                        0.0,
                    )
                ),

            "dc_avg_gpu_load":
                float(
                    dc_load.get(
                        "avg_gpu_load",
                        0.0,
                    )
                ),

            "dc_peak_gpu_load":
                float(
                    dc_load.get(
                        "peak_gpu_load",
                        0.0,
                    )
                ),

            "dc_avg_running_jobs":
                float(
                    dc_load.get(
                        "avg_running_jobs",
                        0.0,
                    )
                ),

            "dc_peak_running_jobs":
                int(
                    dc_load.get(
                        "peak_running_jobs",
                        0,
                    )
                ),

            "dc_busy_ratio":
                float(
                    dc_load.get(
                        "busy_ratio",
                        0.0,
                    )
                ),

            # 当前 DC 每台 Host 的详细负载。
            "host_load_details_json":
                json.dumps(
                    dc_host_load_details,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
        }

        rows.append(
            row
        )

    return rows

# 打印episode摘要
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

        value = float(
            value
        )

        if not np.isfinite(
                value
        ):
            return "nan"

        return (
            f"{value:.{precision}f}"
        )

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



def train(
    train_config: TrainConfig,

    routing_masac_config:
    Optional[
        RoutingMASACConfig
    ] = None,

    host_sac_config:
    Optional[
        HostSACConfig
    ] = None,

) -> Tuple[
    RoutingMASAC,
    Dict[str, LocalHostSAC],
]:
    validate_training_stage_config(
        train_config
    )

    validate_bgh_feature_config(
        train_config
    )

    bgh_runtime_mode = (
        get_bgh_runtime_mode(
            train_config
        )
    )

    print(
        "\n"
        "============================================================\n"
        "🧭 BGH-MASAC Runtime Configuration\n"
        f"Runtime mode              : {bgh_runtime_mode}\n"
        f"Bayesian Game enabled     : "
        f"{train_config.enable_bayesian_game}\n"
        f"Congestion Game enabled   : "
        f"{train_config.enable_bayesian_game}\n"
        f"Heuristic Guidance enabled: "
        f"{train_config.enable_heuristic_guidance}\n"
        f"Cloud Action enabled      : "
        f"{bool(conf.ENABLE_CLOUD_ACTION)}\n"
        f"Zero-Diff Mode            : "
        f"{is_bgh_zero_diff_mode(train_config)}\n"
        "============================================================\n",
        flush=True,
    )

    set_global_random_seeds(train_config.seed)

    # 创建初始环境
    env = build_environment(
        seed=train_config.seed,
        old_env_path=train_config.old_env_path,
    )

    print(
        "📥 DC Arrival Configuration | "
        f"mode={getattr(env, 'dc_arrival_mode', 'uniform')} | "
        f"global_lambda={float(getattr(env, 'lambda_rate', 0.0)):.6f} | "
        "profile="
        f"{json.dumps(getattr(env, 'dc_arrival_profile', {}), ensure_ascii=False, sort_keys=True)}",
        flush=True,
    )

    bayesian_static_context: (
        BayesianStaticRoutingContext
    ) = (
        build_bayesian_static_routing_context(
            env
        )
    )

    bayesian_game_definition: (
        BayesianRoutingGameDefinition
    ) = (
        build_bayesian_routing_game_definition(
            static_context=(
                bayesian_static_context
            )
        )
    )

    # Bayesian 模式开启时创建正式 Evidence / Belief / Pressure 状态；
    # 关闭时保持 H-MASAC-equivalent Zero-Diff 路径。
    bayesian_game: Optional[BayesianCongestionGame] = None
    if train_config.enable_bayesian_game:
        bayesian_game = build_short_window_game(env, train_config, bayesian_game_definition)


    short_window_runtime = ShortWindowRuntime(bayesian_game) if bayesian_game is not None else None

    bayesian_game_metadata_json = json.dumps(
        bayesian_game_definition.to_metadata(),
        ensure_ascii=False,
        sort_keys=True,
    )

    print(
        "\n"
        "============================================================\n"
        "🎲 BGH-MASAC Bayesian Routing Game Definition\n"
        f"{bayesian_game_metadata_json}\n"
        "============================================================\n",
        flush=True,
    )

    host_observation_builder = (
        HostObservationBuilder(
            env=env,
        )
    )

    # ==========================================================
    # 每个 Edge DC 建立独立 Local Host SAC。
    #
    # Host 层：
    #   - 不使用 PettingZoo
    #   - 不共享 Actor
    #   - 不共享 Critic
    #   - 不使用 Mask
    # ==========================================================

    host_sac_agents: Dict[str, LocalHostSAC] = {}

    host_replay_buffers: Dict[
        str,
        HostReplayBuffer,
    ] = {}

    for host_dc_index, dc_id in enumerate(
            env.edge_dc_ids
    ):
        dc_id = str(dc_id)

        if host_sac_config is None:
            # ==========================================================
            # Local Host SAC fallback configuration
            #
            # 即使 train() 被其他入口直接调用，
            # 没有显式传入 HostSACConfig，
            # 也必须使用 HOST_* 专属参数。
            #
            # 禁止退回 Flat-MASAC / Routing 公共超参数。
            # ==========================================================

            host_sac_config = (
                HostSACConfig(
                    gamma=(
                        conf.HOST_GAMMA
                    ),

                    tau=(
                        conf.HOST_TAU
                    ),

                    actor_lr=(
                        conf.HOST_ACTOR_LR
                    ),

                    critic_lr=(
                        conf.HOST_CRITIC_LR
                    ),

                    alpha_lr=(
                        conf.HOST_ALPHA_LR
                    ),

                    actor_hidden_dim=(
                        conf.HOST_ACTOR_HIDDEN_DIM
                    ),

                    critic_hidden_dim=(
                        conf.HOST_CRITIC_HIDDEN_DIM
                    ),

                    initial_alpha=(
                        conf.HOST_INITIAL_ALPHA
                    ),

                    target_entropy_ratio=(
                        conf.HOST_TARGET_ENTROPY_RATIO
                    ),

                    max_grad_norm=(
                        conf.HOST_MAX_GRAD_NORM
                    ),

                    policy_update_interval=(
                        conf.HOST_POLICY_UPDATE_INTERVAL
                    ),

                    target_update_interval=(
                        conf.HOST_TARGET_UPDATE_INTERVAL
                    ),

                    device=(
                        conf.DEVICE
                    ),

                    seed=int(
                        train_config.seed
                    ),
                )
            )

        host_obs_dim = int(
            host_observation_builder
                .get_obs_dim(
                dc_id
            )
        )

        host_action_dim = int(
            host_observation_builder
                .get_action_dim(
                dc_id
            )
        )

        # ==========================================================
        # 每个 DC 独立 Local Host SAC。
        # ==========================================================

        host_sac_agents[
            dc_id
        ] = LocalHostSAC(
            obs_dim=(
                host_obs_dim
            ),

            action_dim=(
                host_action_dim
            ),

            config=(
                host_sac_config
            ),
        )

        # ==========================================================
        # 第十九步：
        # 每个 Local Host SAC 对应自己的 Host ReplayBuffer。
        #
        # 不共享：
        #   Actor
        #   Critic
        #   ReplayBuffer
        # ==========================================================

        host_replay_buffers[
            dc_id
        ] = HostReplayBuffer(
            dc_id=dc_id,

            capacity=int(
                train_config
                    .host_replay_capacity
            ),

            obs_dim=(
                host_obs_dim
            ),

            action_dim=(
                host_action_dim
            ),

            seed=(
                    int(train_config.seed)
                    + 20_000
                    + host_dc_index
            ),
        )
    host_training_action_steps: Dict[
        str,
        int,
    ] = {
        str(dc_id): 0
        for dc_id
        in env.edge_dc_ids
    }

    # ==============================================================
    # Local Host SAC Independent Action RNG
    #
    # 每个 DC 使用自己独立的 NumPy RNG，
    # 用于 Host random warmup 阶段随机选择 Host。
    #
    # 这样不同 DC 的随机 Host action stream 不会共用
    # 同一个随机生成器。
    # ==============================================================

    host_action_rngs: Dict[
        str,
        np.random.Generator,
    ] = {
        str(dc_id):
            np.random.default_rng(
                int(
                    train_config.seed
                )
                + 30_000
                + dc_index
            )

        for dc_index, dc_id
        in enumerate(
            env.edge_dc_ids
        )
    }

    # ==============================================================
    # Neighbor Historical Feedback Store
    #
    # 第二十九步采用 Collect-Only 模式：
    #
    #   1. Job terminal 后真实收集 source -> target 历史结果；
    #   2. BGH 启用时使用回合内短窗口，关闭时保留原等价路径；
    #   3. USE 开关决定是否将七维邻居反馈送进 Routing Actor。
    #
    # 为避免误操作，本步骤显式要求：
    #
    #   COLLECT = True
    #   USE     = False
    # ==============================================================



    neighbor_feedback_store = (
        NeighborHistoricalFeedbackStore(
            env=env,

            ewma_alpha=(
                train_config
                    .neighbor_feedback_ewma_alpha
            ),

            age_scale_samples=(
                train_config
                    .neighbor_feedback_age_scale_samples
            ),

            confidence_scale_samples=(
                train_config
                    .neighbor_feedback_confidence_scale_samples
            ),
        )
    )

    if bayesian_game is not None:
        neighbor_feedback_store = ShortWindowFeedbackStore(env,
            window_s=train_config.short_window_s,
            confidence_scale_samples=train_config.bayesian_confidence_scale)

    routing_observation_builder = (
        RoutingObservationBuilder(
            env=env,

            # ======================================================
            # 第二十九步仍然 False。
            #
            # RoutingObservationBuilder 在 False 时会直接返回
            # 全 0 Feedback block，并且不会查询 Provider。
            #
            # 因而虽然真实 Store 已经安装，
            # 当前 Routing Actor 仍然完全看不到这些历史信息。
            # ======================================================

            use_neighbor_historical_feedback=(
                train_config
                    .use_neighbor_historical_feedback
            ),

            # Store 现在已经真实存在，
            # 但当前仅作为未来 Provider 接口和日志数据源。
            neighbor_feedback_provider=(
                neighbor_feedback_store
            ),
        )
    )
    routing_obs_dim = int(routing_observation_builder.obs_dim)

    routing_state_builder = (
        RoutingCentralizedStateBuilder(
            env=env,
            routing_observation_builder=(
                routing_observation_builder
            ),
        )
    )

    routing_global_state_dim = int(
        routing_state_builder.state_dim
    )

    pending_trace_store = PendingJobTraceStore()

    # ==============================================================
    # Training Reward Model
    #
    # Reward Model 属于 Trainer，
    # Environment 只提供物理事实。
    #
    # 当前参数数值与第三十一步之前完全一致。
    # ==============================================================

    training_reward_model = (
        HMasacTrainingRewardModel(
            TrainingRewardConfig(
                task_completion_reward=(
                    conf.TASK_COMPLETION_REWARD
                ),

                completion_time_cost_weight=(
                    conf
                        .COMPLETION_TIME_COST_WEIGHT
                ),

                sla_violation_cost_weight=(
                    conf
                        .SLA_VIOLATION_COST_WEIGHT
                ),

                remote_offload_base_penalty=(
                    conf
                        .REMOTE_OFFLOAD_BASE_PENALTY
                ),

                remote_latency_cost_weight=(
                    conf
                        .REMOTE_LATENCY_COST_WEIGHT
                ),

                sla_risk_cost_weight=(
                    conf
                        .SLA_RISK_COST_WEIGHT
                ),

                timeout_drop_penalty=(
                    conf.TIMEOUT_DROP_PENALTY
                ),

                resource_drop_penalty=(
                    conf.RESOURCE_DROP_PENALTY
                ),

                energy_normalization_j=(
                    conf.ENERGY_NORMALIZATION_J
                ),

                energy_cost_weight=(
                    conf.ENERGY_COST_WEIGHT
                ),

                max_latency_s=float(
                    env.max_latency
                ),

                max_job_duration_s=float(
                    env.max_job_duration
                ),

                sla_deadline_ratio=float(
                    env.sla_deadline_ratio
                ),

                drop_deadline_ratio=float(
                    env.drop_deadline_ratio
                ),

                norm_eps=float(
                    env.norm_eps
                ),
            )
        )
    )

    # 创建 Transition 采集器
    collector = TransitionCollector(
        env=env,
        short_window_runtime=short_window_runtime,

        routing_observation_builder=(
            routing_observation_builder
        ),

        routing_state_builder=(
            routing_state_builder
        ),

        pending_trace_store=(
            pending_trace_store
        ),

        # Training Reward 由 Trainer-side Reward Model 负责。
        training_reward_model=(
            training_reward_model
        ),
    )

    # ==============================================================
    # Routing MASAC 专用 ReplayBuffer
    #
    # 与 Host Replay 完全独立。
    #
    # 这里只接收 Job terminal 后生成的
    # Finalized RoutingTransition。
    # ==============================================================

    routing_replay_buffer = RoutingReplayBuffer(
        capacity=int(
            train_config
                .routing_replay_capacity
        ),

        local_obs_dim=(
            routing_obs_dim
        ),

        global_state_dim=(
            routing_global_state_dim
        ),

        seed=int(
            train_config.seed
        ),
    )


    # 没有传入算法配置时，使用 MASACConfig 默认值，但让算法随机种子与训练配置保持一致
    # ==============================================================
    # Routing MASAC + CTDE
    #
    # 这里创建的是整个系统唯一的一套 Routing MASAC。
    #
    # Host SAC 不使用本对象。
    # 每个 Edge DC 的 LocalHostSAC 已在前面独立创建。
    # ==============================================================

    if routing_masac_config is None:
        # ==========================================================
        # Routing MASAC fallback configuration
        #
        # config.py 是  实验的统一参数入口。
        # 即使 train() 被直接调用，
        # Routing 也必须继续使用 ROUTING_* 配置。
        # ==========================================================

        routing_masac_config = (
            RoutingMASACConfig(
                gamma=(
                    conf.ROUTING_GAMMA
                ),

                tau=(
                    conf.ROUTING_TAU
                ),

                actor_lr=(
                    conf.ROUTING_ACTOR_LR
                ),

                critic_lr=(
                    conf.ROUTING_CRITIC_LR
                ),

                alpha_lr=(
                    conf.ROUTING_ALPHA_LR
                ),

                actor_hidden_dim=(
                    conf.ROUTING_ACTOR_HIDDEN_DIM
                ),

                critic_hidden_dim=(
                    conf.ROUTING_CRITIC_HIDDEN_DIM
                ),

                initial_alpha=(
                    conf.ROUTING_INITIAL_ALPHA
                ),

                target_entropy_ratio=(
                    conf.ROUTING_TARGET_ENTROPY_RATIO
                ),

                max_grad_norm=(
                    conf.ROUTING_MAX_GRAD_NORM
                ),

                policy_update_interval=(
                    conf.ROUTING_POLICY_UPDATE_INTERVAL
                ),

                target_update_interval=(
                    conf.ROUTING_TARGET_UPDATE_INTERVAL
                ),

                device=(
                    conf.DEVICE
                ),

                seed=int(
                    train_config.seed
                ),
            )
        )


    routing_masac = RoutingMASAC(
        # Routing Actor local input
        local_obs_dim=(
            routing_obs_dim
        ),

        # Routing CTDE centralized Critic input
        global_state_dim=(
            routing_global_state_dim
        ),

        action_dim=int(
            env.action_dim
        ),

        num_agents=int(
            len(
                env.possible_agents
            )
        ),

        config=(
            routing_masac_config
        ),
    )

    routing_masac.train_mode()

    # 第 6 步：统一 Guided Policy 入口。
    # enabled=False 时内部严格委托 RoutingMASAC.select_action()，
    # 因而当前 H-MASAC-equivalent 路径的 logits、采样和 Replay 语义不变。
    guided_policy = GuidedRoutingPolicy(
        base_policy=routing_masac,
        action_target_dc_ids=env.routing_action_target_dc_ids,
        enabled=train_config.enable_heuristic_guidance,
        bayesian_game=bayesian_game,
    )

    # 第 8 步：Guidance λ 调度器与训练生命周期同范围创建，
    # 但不会修改网络参数；每个 Episode 开始时只读取一次 λ。
    guidance_lambda_schedule = GuidanceLambdaSchedule(
        stage2_start=train_config.guidance_lambda_stage2_start,
        stage2_peak=train_config.guidance_lambda_stage2_peak,
        stage3_end=train_config.guidance_lambda_stage3_end,
    )

    action_rng = np.random.default_rng(int(train_config.seed))

    (
        start_episode,
        global_decision_steps,
        routing_normal_action_steps,
        host_training_action_steps,
        best_episode_return,
    ) = load_two_layer_checkpoint_if_needed(
        # ==========================================================
        # Resume 前需要当前 Environment / TrainConfig，
        # 用于验证 Cloud、DC、Host、Observation、
        # Action 以及三阶段训练 metadata。
        # ==========================================================

        env=env,

        train_config=(
            train_config
        ),

        routing_masac=(
            routing_masac
        ),

        host_sac_agents=(
            host_sac_agents
        ),

        resume_checkpoint=(
            train_config.resume_checkpoint
        ),
    )

    checkpoint_dir = Path(
        train_config.checkpoint_dir
    )

    # ==============================================================
    # Two-Level Training Logs
    #
    # 同一次训练共用相同 timestamp：
    #
    #   episode_log_YYYYMMDD_HHMMSS.csv
    #   dc_log_YYYYMMDD_HHMMSS.csv
    # ==============================================================

    (
        episode_log_csv_path,
        dc_log_csv_path,
    ) = build_run_log_paths(
        episode_base_log_path=(
            train_config
                .episode_log_csv_path
        ),

        dc_base_log_path=(
            train_config
                .dc_log_csv_path
        ),
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    episode_log_csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dc_log_csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ==============================================================
    # Compatibility Pointer
    #
    # current_train_log.txt 继续指向主 Episode Log，
    # 防止已有服务器查看脚本失效。
    # ==============================================================

    current_log_pointer_path = (
            episode_log_csv_path.parent
            / "current_train_log.txt"
    )

    atomic_checkpoint_text_write(
        str(
            episode_log_csv_path.resolve()
        ),
        current_log_pointer_path,
    )

    # 新增 DC 日志 pointer。
    current_dc_log_pointer_path = (
            dc_log_csv_path.parent
            / "current_dc_log.txt"
    )

    atomic_checkpoint_text_write(
        str(
            dc_log_csv_path.resolve()
        ),
        current_dc_log_pointer_path,
    )

    print(
        "\n"
        "============================================================\n"
        "📊 BGH-MASAC 双层训练日志\n"
        f"Episode CSV : "
        f"{episode_log_csv_path.resolve()}\n"
        f"DC CSV      : "
        f"{dc_log_csv_path.resolve()}\n"
        f"Episode Ptr : "
        f"{current_log_pointer_path.resolve()}\n"
        f"DC Ptr      : "
        f"{current_dc_log_pointer_path.resolve()}\n"
        "\n"
        "每个 Episode 结束后立即 flush + fsync。\n"
        "============================================================\n",
        flush=True,
    )


    guidance_log = (GuidanceDecisionLog(episode_log_csv_path.with_suffix(".guidance.jsonl"))
                    if guided_policy.enabled else None)
    try:
        # 从 start_episode 训练到 num_episodes，包含最后一个 episode。
        for episode in range(int(start_episode), int(train_config.num_episodes) + 1):
            if guidance_log is not None:
                guidance_log.flush()

            ##################### 一轮 episode 开始前的准备 ####################
            # 根据配置决定当前 episode 使用哪个环境 seed
            if train_config.vary_episode_seed:
                episode_seed = int(train_config.seed + episode - 1)
            else:
                episode_seed = int(train_config.seed)

            # 重启环境
            pending_trace_store.reset_episode()
            neighbor_feedback_store.reset_episode_counters()
            env.reset(seed=episode_seed)
            if short_window_runtime is not None:
                short_window_runtime.reset()

            collector.reset_episode()

            training_stage = (
                resolve_training_stage(
                    episode=episode,
                    train_config=(
                        train_config
                    ),
                )
            )

            # 第 8 步：按现有三阶段边界计算当前 Episode 的 λ。
            # Stage 1 恒为 0；Stage 2 线性 warm-up；Stage 3 线性 decay。
            scheduled_guidance_lambda = (
                guidance_lambda_schedule.lambda_for_episode(
                    stage=training_stage.value,
                    episode=episode,
                    stage_start_episode=training_stage_start_episode(
                        stage=training_stage,
                        train_config=train_config,
                    ),
                    stage_end_episode=training_stage_end_episode(
                        stage=training_stage,
                        train_config=train_config,
                    ),
                )
            )
            guidance_lambda = float(
                scheduled_guidance_lambda
                if guided_policy.enabled
                else 0.0
            )

            if (
                    episode
                    == training_stage_start_episode(
                stage=training_stage,
                train_config=train_config,
            )
            ):
                print(
                    "\n"
                    "============================================================\n"
                    f"🚦 Training Stage Start: {training_stage.value}\n"
                    f"Episode Range: "
                    f"{training_stage_start_episode(training_stage, train_config)}"
                    f" -> "
                    f"{training_stage_end_episode(training_stage, train_config)}\n"
                    "============================================================\n",
                    flush=True,
                )

                if training_stage == TrainingStage.JOINT_FINETUNE:
                    # Save the state before the first joint optimization episode.
                    save_two_layer_checkpoint(
                        env=env,
                        training_stage=training_stage,
                        train_config=train_config,
                        routing_masac=routing_masac,
                        host_sac_agents=host_sac_agents,
                        model_path=checkpoint_dir / "joint_finetune_start.pt",
                        next_episode=episode,
                        global_decision_steps=global_decision_steps,
                        routing_normal_action_steps=routing_normal_action_steps,
                        host_training_action_steps=host_training_action_steps,
                        best_episode_return=best_episode_return,
                    )

            apply_training_stage_modes(
                stage=(
                    training_stage
                ),

                routing_masac=(
                    routing_masac
                ),

                host_sac_agents=(
                    host_sac_agents
                ),
            )

            # 为每个智能体创建奖励累计字典
            per_agent_returns = {}
            for agent_id in env.possible_agents:
                per_agent_returns[str(agent_id)] = 0.0

            # 创建当前 episode 的统计对象
            stats = EpisodeStatistics(
                episode=int(
                    episode
                ),

                episode_seed=(
                    episode_seed
                ),

                per_agent_returns=(
                    per_agent_returns
                ),

                training_stage=(
                    training_stage.value
                ),

                guidance_lambda=float(guidance_lambda),
            )

            # 记录episode开始的时间
            episode_wall_start = time.perf_counter()
            ###################################################################
            decision: Optional[DecisionSnapshot] = None

            # 只要 PettingZoo 的活跃智能体列表不为空，就继续循环
            while env.agents:

                # ==========================================================
                # Phase 1：Host decision
                #
                # Host 必须优先于任何 PettingZoo 操作处理。
                # Self Routing 后：
                #
                #   agent_selection == None
                #   pending_host_job_id != None
                #
                # 因此此处绝对不能调用：
                #   collector.capture_decision()
                #   env.step()
                # ==========================================================
                if env.has_pending_host_decision():
                    # ==========================================================
                    # Host Phase
                    #
                    # Host 层完全脱离 PettingZoo。
                    # 这里使用与 Routing Collector 相同的
                    # pending_trace_store。
                    # ==========================================================

                    host_context = (
                        env.get_pending_host_decision()
                    )

                    host_job_id = str(
                        host_context["job_id"]
                    )

                    host_dc_id = str(
                        host_context["dc_id"]
                    )

                    # ==========================================================
                    # 保存 Host SAC 真正做决策的时间。
                    #
                    # execute_pending_host_action() 后环境可能已经向前推进，
                    # 因此不能在执行之后再读取 decision time。
                    # ==========================================================

                    host_decision_time = float(
                        env.current_time
                    )

                    # Host Observation 完全独立于 PettingZoo。
                    host_obs = (
                        host_observation_builder.build(
                            dc_id=host_dc_id,
                            job_id=host_job_id,
                        )
                    )

                    # 当前 DC 自己的 Local Host SAC 决策。
                    host_agent = (
                        host_sac_agents[
                            host_dc_id
                        ]
                    )

                    host_replay = (
                        host_replay_buffers[
                            host_dc_id
                        ]
                    )

                    # ==========================================================
                    # Training Stage 决定 Host action 行为。
                    # ==========================================================

                    if (
                            training_stage
                            == TrainingStage.ROUTING_TRAIN
                    ):
                        # ======================================================
                        # Stage 2：
                        #
                        # Host 网络完全冻结。
                        #
                        # 使用 deterministic policy，
                        # 降低 Routing MASAC 所面对环境的非平稳性。
                        # ======================================================

                        host_action = (
                            host_agent.select_action(
                                host_obs=host_obs,
                                deterministic=True,
                            )
                        )

                        host_action_source = (
                            "policy"
                        )

                    else:

                        # ======================================================
                        # Stage 1 / Stage 3：
                        #
                        # Host SAC 参与训练。
                        # 每个 DC 独立进行 random warmup。
                        # ======================================================

                        host_training_steps = int(
                            host_training_action_steps[
                                host_dc_id
                            ]
                        )

                        if (
                                host_training_steps
                                < int(
                            train_config
                                    .host_random_warmup_steps
                        )
                        ):

                            host_action = (
                                choose_random_host_action(
                                    action_dim=(
                                        host_agent.action_dim
                                    ),

                                    rng=(
                                        host_action_rngs[
                                            host_dc_id
                                        ]
                                    ),
                                )
                            )

                            host_action_source = (
                                "random"
                            )

                        else:

                            host_action = (
                                host_agent.select_action(
                                    host_obs=host_obs,
                                    deterministic=False,
                                )
                            )

                            host_action_source = (
                                "policy"
                            )

                        host_training_action_steps[
                            host_dc_id
                        ] += 1

                    # ==========================================================
                    # 非 PettingZoo Host execution。
                    #
                    # Environment 会返回：
                    #
                    #   job_id
                    #   dc_id
                    #   host_action
                    #   host_id
                    #   execution_result
                    #   env_time
                    # ==========================================================

                    host_result = (
                        env.execute_pending_host_action(
                            host_action=host_action,
                        )
                    )

                    # ==========================================================
                    # 防御性检查：
                    # Trace 中记录的 Job/DC/Action 必须与 Environment
                    # 真正执行的对象完全一致。
                    # ==========================================================

                    result_job_id = str(
                        host_result["job_id"]
                    )

                    result_dc_id = str(
                        host_result["dc_id"]
                    )

                    result_host_action = int(
                        host_result["host_action"]
                    )

                    if result_job_id != host_job_id:
                        raise RuntimeError(
                            "Host execution 返回 Job 不一致："
                            f"expected={host_job_id}, "
                            f"actual={result_job_id}"
                        )

                    if result_dc_id != host_dc_id:
                        raise RuntimeError(
                            "Host execution 返回 DC 不一致："
                            f"job={host_job_id}, "
                            f"expected={host_dc_id}, "
                            f"actual={result_dc_id}"
                        )

                    if (
                            result_host_action
                            != int(host_action)
                    ):
                        raise RuntimeError(
                            "Host execution 返回 action 不一致："
                            f"job={host_job_id}, "
                            f"expected={host_action}, "
                            f"actual={result_host_action}"
                        )

                    actual_host_id = str(
                        host_result["host_id"]
                    )

                    # ==========================================================
                    # 将 Host Decision 正式加入 Job Causal Trace。
                    #
                    # 此时仅记录事实。
                    # 仍然不写 Host ReplayBuffer。
                    # ==========================================================

                    pending_trace_store.record_host_step(
                        job_id=host_job_id,
                        dc_id=host_dc_id,
                        env_time=host_decision_time,
                        host_obs=host_obs,
                        action=int(
                            host_action
                        ),
                        host_id=actual_host_id,

                        # Stage 1 warmup 可以是 random；
                        # Stage 1/3 后期以及 Stage 2 为 policy。
                        action_source=(
                            host_action_source
                        ),
                    )

                    # ==========================================================
                    # 回填 Host placement 的即时执行结果：
                    #
                    #   started
                    #   queued
                    #   dropped
                    #
                    # 它不是最终 SLA/completion reward。
                    # ==========================================================

                    pending_trace_store.record_host_result(
                        job_id=host_job_id,

                        result=str(
                            host_result[
                                "execution_result"
                            ]
                        ),

                        env_time=float(
                            host_result[
                                "env_time"
                            ]
                        ),
                    )
                    if short_window_runtime is not None:
                        short_window_runtime.record_host_result(
                            job_id=host_job_id,
                            dc_id=host_dc_id,
                            execution_result=str(host_result["execution_result"]),
                            now=float(host_result["env_time"]),
                        )
                    stats.record_host_decision(
                        dc_id=(
                            host_dc_id
                        ),

                        action_source=(
                            host_action_source
                        ),
                    )

                    stats.record_host_result(
                        dc_id=(
                            host_dc_id
                        ),

                        execution_result=str(
                            host_result[
                                "execution_result"
                            ]
                        ),
                    )

                    # ==========================================================
                    # execute_pending_host_action() 内部会继续推进事件，
                    # 因而期间可能已经发生：
                    #
                    #   completed
                    #   waiting_timeout
                    #   resource failure
                    #
                    # 必须在 Host Step 已经写入以后，
                    # 再消费这些 delayed outcome。
                    # ==========================================================

                    consume_environment_outcome_events(
                        env=env,
                        training_reward_model=(
                            training_reward_model
                        ),

                        pending_trace_store=(
                            pending_trace_store
                        ),

                        routing_replay_buffer=(
                            routing_replay_buffer
                        ),

                        host_replay_buffers=(
                            host_replay_buffers
                        ),

                        stats=stats,
                        neighbor_feedback_store=(
                            neighbor_feedback_store
                        ),

                        collect_neighbor_historical_feedback=(
                            train_config
                                .collect_neighbor_historical_feedback
                        ),

                        bayesian_game=bayesian_game,


                    )
                    # ==========================================================
                    # Local Host SAC Update
                    #
                    # 只在：
                    #
                    #   Stage 1 Host Pretrain
                    #   Stage 3 Joint Fine-tune
                    #
                    # 更新。
                    #
                    # Stage 2 Host 完全冻结。
                    # ==========================================================

                    if stage_trains_host(
                            training_stage
                    ):

                        host_steps = int(
                            host_training_action_steps[
                                host_dc_id
                            ]
                        )

                        ready_to_update_host = (
                                host_steps
                                >= int(
                            train_config
                                .host_learning_starts
                        )

                                and host_steps
                                % int(
                            train_config
                                .host_train_every
                        )
                                == 0

                                and host_replay.can_sample(
                            batch_size=int(
                                train_config
                                    .host_batch_size
                            )
                        )
                        )

                        if ready_to_update_host:

                            host_update_infos = []

                            for _ in range(
                                    int(
                                        train_config
                                                .host_updates_per_train
                                    )
                            ):
                                host_update_infos.append(
                                    host_agent.update(
                                        replay_buffer=(
                                            host_replay
                                        ),

                                        batch_size=int(
                                            train_config
                                                .host_batch_size
                                        ),
                                    )
                                )

                            record_host_update_block(
                                stats=stats,

                                dc_id=(
                                    host_dc_id
                                ),

                                update_infos=(
                                    host_update_infos
                                ),
                            )

                    decision = None


                    continue

                # ==========================================================
                # Phase 2：PettingZoo Routing decision
                # ==========================================================

                if collector.drain_one_dead_agent():
                    decision = None
                    continue

                if decision is None:
                    decision = collector.capture_decision()

                # ----------------------------------------------------------
                # 以下继续保留现有 Routing：
                #
                # forced
                # random warmup
                # Routing MASAC policy
                # ----------------------------------------------------------

                if decision.forced_action is not None:

                    # Environment lifecycle forced drop
                    action = int(
                        decision.forced_action
                    )

                    action_source = (
                        "forced"
                    )


                elif (
                        training_stage
                        == TrainingStage.HOST_PRETRAIN
                ):

                    # ======================================================
                    # Stage 1:
                    #
                    # Routing 不参与学习。
                    # 所有正常 Job 都进入当前 DC 的 Host 层。
                    #
                    # 该 Self action：
                    #   - 写入 Causal Trace；
                    #   - 不进入 Routing Replay；
                    #   - 不增加 Routing warmup step。
                    # ======================================================

                    action = (
                        get_self_routing_action(
                            env=env,
                            agent_id=(
                                decision.agent_id
                            ),
                        )
                    )

                    action_source = (
                        "orchestrator"
                    )


                elif (
                        routing_normal_action_steps
                        < int(
                    train_config
                            .routing_random_warmup_steps
                )
                ):

                    action = (
                        choose_random_routing_action(
                            action_dim=(
                                env.action_dim
                            ),

                            rng=(
                                action_rng
                            ),
                        )
                    )

                    action_source = (
                        "random"
                    )


                else:

                    # 第 7 步：动作执行前生成当前 source DC 的候选动作上下文。
                    # Guidance 关闭时不查询启发式上下文。
                    action_context = None
                    if guided_policy.enabled:
                        action_context = short_window_runtime.build_action_context(
                            guided_policy, env, decision, neighbor_feedback_store)

                    action = (
                        guided_policy
                            .select_action(
                            local_obs=(
                                decision.local_obs
                            ),

                            agent_index=(
                                decision.agent_index
                            ),

                            source_dc_id=(
                                decision.agent_id
                            ),

                            # Guided Policy 只接收归一化历史特征，
                            # 不把实时远端资源写入 Observation。
                            action_context=action_context,

                            guidance_lambda=guidance_lambda,

                            deterministic=False,
                        )
                    )

                    action_source = (
                        "policy"
                    )

                # ==========================================================
                # Routing action 的真实语义统一由 Collector
                # 调用 Environment._decode_action() 决定。
                #
                # Trainer 不再复制 action_type 推断逻辑。
                # ==========================================================

                if guidance_log is not None and action_source == "policy":
                    guidance_log.record(episode=episode, decision=decision, policy=guided_policy,
                        action=action, guidance_lambda=guidance_lambda)

                routing_result, next_decision = (
                    collector.execute_and_record(
                        decision=decision,

                        action=action,

                        action_source=(
                            action_source
                        ),
                    )
                )

                decision = next_decision

                # 记录这条经验的奖励和动作类型
                # ==============================================================
                # Routing Layer Episode Statistics
                #
                # RoutingActionResult 已经由 Collector 根据 Environment
                # 的真实动作语义生成。
                #
                # 这里同时记录：
                #   - source DC
                #   - target DC
                #   - action type
                #   - action source
                #   - immediate reward
                #
                # 从而可以生成完整 source -> target Routing Matrix。
                # ==============================================================

                stats.record_routing_decision(
                    agent_id=(
                        routing_result.agent_id
                    ),

                    reward=(
                        routing_result
                            .immediate_reward
                    ),

                    action_type=(
                        routing_result
                            .action_type
                    ),

                    action_source=(
                        routing_result
                            .action_source
                    ),

                    target_dc_id=(
                        routing_result
                            .target_dc_id
                    ),
                )

                # ==========================================================
                # Forced Drop / Actor Drop 会在 execute_and_collect()
                # 内部直接完成 Job Finalize。
                #
                # 它不会产生后续 Environment reward correction，
                # 所以这里必须检查并立即 Flush。
                # ==========================================================

                # ==========================================================
                # 当前 action 如果让 Job 在 Collector 中直接 Finalize，
                # 典型情况就是 forced drop。
                #
                # 此类 Job 不会再产生 Environment terminal correction，
                # 因此这里立即 Flush。
                # ==========================================================

                if routing_result.job_finalized:
                    finalized_trace = (
                        pending_trace_store
                            .get_finalized_trace(
                            routing_result.job_id
                        )
                    )

                    flush_finalized_trace_to_replay(
                        finalized_trace=(
                            finalized_trace
                        ),

                        routing_replay_buffer=(
                            routing_replay_buffer
                        ),

                        host_replay_buffers=(
                            host_replay_buffers
                        ),

                        stats=stats,
                        neighbor_feedback_store=(
                            neighbor_feedback_store
                        ),

                        collect_neighbor_historical_feedback=(
                            train_config
                                .collect_neighbor_historical_feedback
                        ),

                        bayesian_game=bayesian_game,
                        env=env,
                    )

                    pending_trace_store.pop_finalized_trace(
                        routing_result.job_id
                    )



                consume_environment_outcome_events(
                    env=env,
                    training_reward_model=(
                        training_reward_model
                    ),

                    pending_trace_store=(
                        pending_trace_store
                    ),

                    routing_replay_buffer=(
                        routing_replay_buffer
                    ),

                    host_replay_buffers=(
                        host_replay_buffers
                    ),

                    stats=stats,
                    neighbor_feedback_store=(
                        neighbor_feedback_store
                    ),

                    collect_neighbor_historical_feedback=(
                        train_config
                            .collect_neighbor_historical_feedback
                    ),


                    bayesian_game=bayesian_game,

                )

                # 增加计数
                global_decision_steps += 1
                if action_source in {
                    "random",
                    "policy",
                }:

                    routing_normal_action_steps += 1
                    if (
                            routing_normal_action_steps
                            == int(
                        train_config
                                .routing_random_warmup_steps
                    )
                    ):
                        print(
                            "\n"
                            "============================================================\n"
                            "✅ Routing 随机动作预热结束\n"
                            "============================================================\n"
                        )

                # 同时满足以下条件才允许更新网络：
                # 1. 普通动作总数已经达到 learning_starts；
                # 2. ReplayBuffer 中普通经验足够采样一个 batch。
                ready_to_update_routing = (
                        stage_trains_routing(
                            training_stage
                        )

                        and action_source
                        in {
                            "random",
                            "policy",
                        }

                        and routing_normal_action_steps
                        >= int(
                    train_config
                        .routing_learning_starts
                )

                        and routing_normal_action_steps
                        % int(
                    train_config
                        .routing_train_every
                )
                        == 0

                        and routing_replay_buffer
                        .can_sample(
                    batch_size=int(
                        train_config
                            .routing_batch_size
                    ),

                    include_forced_actions=False,
                )
                )

                # 网络更新
                if ready_to_update_routing:

                    update_info_block = []

                    for _ in range(
                            int(
                                train_config
                                        .routing_updates_per_train
                            )
                    ):
                        update_info = (
                            routing_masac.update(
                                replay_buffer=(
                                    routing_replay_buffer
                                ),

                                batch_size=int(
                                    train_config
                                        .routing_batch_size
                                ),
                            )
                        )

                        update_info_block.append(
                            update_info
                        )

                    record_routing_update_block(
                        stats=stats,

                        update_infos=(
                            update_info_block
                        ),
                    )

            # ==============================================================
            # Episode 结束后的最后一次 delayed outcome flush。
            #
            # 正常情况下 Routing/Host Branch 已经实时消费；
            # 这里作为 Episode tail 的防御性收尾，
            # 防止最后一批 terminal correction 留在 Environment 中。
            # ==============================================================

            consume_environment_outcome_events(
                env=env,
                training_reward_model=(
                    training_reward_model
                ),
                pending_trace_store=(
                    pending_trace_store
                ),
                routing_replay_buffer=(
                    routing_replay_buffer
                ),
                host_replay_buffers=(
                    host_replay_buffers
                ),
                stats=stats,
                neighbor_feedback_store=(
                    neighbor_feedback_store
                ),

                collect_neighbor_historical_feedback=(
                    train_config
                        .collect_neighbor_historical_feedback
                ),
                bayesian_game=bayesian_game,
            )
            pending_trace_store.assert_no_open_trace()
            pending_trace_store.assert_no_unflushed_finalized_trace()

            # 计算当前 episode 的真实运行秒数。
            wall_time_seconds = (time.perf_counter() - episode_wall_start)

            # ==============================================================
            # Episode-level Physical Metrics
            #
            # 三套真实系统指标只计算一次，
            # Episode Log / DC Log 共用同一份 snapshot。
            # ==============================================================

            service_metrics = (
                calculate_service_metrics(
                    env
                )
            )

            energy_metrics = (
                calculate_episode_energy_metrics(
                    env
                )
            )

            load_metrics = (
                calculate_episode_load_metrics(
                    env
                )
            )

            # ==============================================================
            # Episode Log
            # ==============================================================

            episode_log_row = (
                build_episode_log_row(
                    stats=stats,

                    env=env,

                    routing_replay_buffer=(
                        routing_replay_buffer
                    ),

                    host_replay_buffers=(
                        host_replay_buffers
                    ),

                    routing_masac=(
                        routing_masac
                    ),

                    host_sac_agents=(
                        host_sac_agents
                    ),

                    host_training_action_steps=(
                        host_training_action_steps
                    ),

                    pending_trace_store=(
                        pending_trace_store
                    ),
                    neighbor_feedback_store=(
                        neighbor_feedback_store
                    ),

                    global_decision_steps=(
                        global_decision_steps
                    ),

                    routing_normal_action_steps=(
                        routing_normal_action_steps
                    ),

                    wall_time_seconds=(
                        wall_time_seconds
                    ),

                    service_metrics=(
                        service_metrics
                    ),

                    energy_metrics=(
                        energy_metrics
                    ),

                    load_metrics=(
                        load_metrics
                    ),
                )
            )

            # ==============================================================
            # Per-DC Log
            # ==============================================================

            dc_log_rows = (
                build_dc_log_rows(
                    stats=stats,

                    env=env,

                    host_sac_agents=(
                        host_sac_agents
                    ),

                    host_replay_buffers=(
                        host_replay_buffers
                    ),

                    host_training_action_steps=(
                        host_training_action_steps
                    ),

                    load_metrics=(
                        load_metrics
                    ),

                    neighbor_feedback_store=(
                        neighbor_feedback_store
                    ),
                )
            )

            # ==============================================================
            # Immediate Disk Persistence
            # ==============================================================

            append_csv_log(
                csv_path=(
                    episode_log_csv_path
                ),

                row=(
                    episode_log_row
                ),
            )

            for dc_log_row in (
                    dc_log_rows
            ):
                append_csv_log(
                    csv_path=(
                        dc_log_csv_path
                    ),

                    row=(
                        dc_log_row
                    ),
                )

            # ==============================================================
            # Console Summary
            # ==============================================================

            if (
                    episode
                    % int(
                train_config.log_interval
            )
                    == 0
            ):
                print_episode_summary(
                    episode_log_row
                )

            # Keep the best return in the resumable trainer state.
            if stats.episode_return > best_episode_return:
                best_episode_return = float(stats.episode_return)

    finally:
        if guidance_log is not None:
            guidance_log.close()
        close_method = getattr(env, "close", None)
        if callable(close_method):
            close_method()

    # 全部 episode 完成后保存 final checkpoint
    save_two_layer_checkpoint(
        env=env,

        training_stage=(
            resolve_training_stage(
                episode=int(
                    train_config.num_episodes
                ),

                train_config=(
                    train_config
                ),
            )
        ),

        train_config=(
            train_config
        ),

        routing_masac=(
            routing_masac
        ),

        host_sac_agents=(
            host_sac_agents
        ),

        model_path=(
                checkpoint_dir
                / "final.pt"
        ),

        next_episode=(
                int(
                    train_config.num_episodes
                )
                + 1
        ),

        global_decision_steps=(
            global_decision_steps
        ),

        routing_normal_action_steps=(
            routing_normal_action_steps
        ),

        host_training_action_steps=(
            host_training_action_steps
        ),

        best_episode_return=(
            best_episode_return
        ),
    )

    return (
        routing_masac,
        host_sac_agents,
    )

def main() -> None:
    train_config = TrainConfig(
        num_episodes=conf.Episodes,
        routing_replay_capacity=(conf.ROUTING_REPLAY_CAPACITY),
        host_replay_capacity=(conf.HOST_REPLAY_CAPACITY),
        # ==========================================================
        # Routing MASAC Training Schedule
        # ==========================================================

        routing_batch_size=(
            conf.ROUTING_BATCH_SIZE
        ),

        routing_random_warmup_steps=(
            conf.ROUTING_RANDOM_WARMUP_STEPS
        ),

        routing_learning_starts=(
            conf.ROUTING_LEARNING_STARTS
        ),

        routing_train_every=(
            conf.ROUTING_TRAIN_EVERY
        ),

        routing_updates_per_train=(
            conf.ROUTING_UPDATES_PER_TRAIN
        ),

        # ==========================================================
        # Local Host SAC Training Schedule
        # ==========================================================

        host_batch_size=(
            conf.HOST_BATCH_SIZE
        ),

        host_random_warmup_steps=(
            conf.HOST_RANDOM_WARMUP_STEPS
        ),

        host_learning_starts=(
            conf.HOST_LEARNING_STARTS
        ),

        host_train_every=(
            conf.HOST_TRAIN_EVERY
        ),

        host_updates_per_train=(
            conf.HOST_UPDATES_PER_TRAIN
        ),

        # ==========================================================
        # Three-Stage Training
        # ==========================================================

        host_pretrain_episodes=(
            conf.HOST_PRETRAIN_EPISODES
        ),

        routing_train_episodes=(
            conf.ROUTING_TRAIN_EPISODES
        ),

        joint_finetune_episodes=(
            conf.JOINT_FINETUNE_EPISODES
        ),
        log_interval=conf.Log_interval,
        checkpoint_interval=conf.Checkpoint_Interval,
        seed=conf.Seed,
        checkpoint_dir=conf.BGH_MASAC_CHECKPOINT_DIR,

        episode_log_csv_path=(
            conf.BGH_MASAC_EPISODE_LOG_CSV_PATH
        ),

        dc_log_csv_path=(
            conf.BGH_MASAC_DC_LOG_CSV_PATH
        ),
        old_env_path=conf.Old_Env_Path,
        resume_checkpoint=conf.BGH_MASAC_RESUME_CHECKPOINT,
        vary_episode_seed=conf.Vary_Episode_Seed,
        # ==========================================================
        # Neighbor Historical Feedback
        # ==========================================================

        collect_neighbor_historical_feedback=(
            conf.COLLECT_NEIGHBOR_HISTORICAL_FEEDBACK
        ),

        use_neighbor_historical_feedback=(
            conf.USE_NEIGHBOR_HISTORICAL_FEEDBACK
        ),

        neighbor_feedback_ewma_alpha=(
            conf.NEIGHBOR_FEEDBACK_EWMA_ALPHA
        ),

        neighbor_feedback_age_scale_samples=(
            conf.NEIGHBOR_FEEDBACK_AGE_SCALE_SAMPLES
        ),

        neighbor_feedback_confidence_scale_samples=(
            conf.NEIGHBOR_FEEDBACK_CONFIDENCE_SCALE_SAMPLES
        ),
        enable_bayesian_game=(
            conf.BGH_ENABLE_BAYESIAN_GAME
        ),

        enable_heuristic_guidance=(
            conf.BGH_ENABLE_HEURISTIC_GUIDANCE
        ),

    )

    host_sac_config = (
        HostSACConfig(
            # ======================================================
            # Host SAC 使用完全独立的算法超参数。
            #
            # 正常 main() 入口与 train() fallback 必须保持一致，
            # 防止 Host 又退回旧 Flat-MASAC 公共参数。
            # ======================================================

            gamma=(
                conf.HOST_GAMMA
            ),

            tau=(
                conf.HOST_TAU
            ),

            actor_lr=(
                conf.HOST_ACTOR_LR
            ),

            critic_lr=(
                conf.HOST_CRITIC_LR
            ),

            alpha_lr=(
                conf.HOST_ALPHA_LR
            ),

            actor_hidden_dim=(
                conf.HOST_ACTOR_HIDDEN_DIM
            ),

            critic_hidden_dim=(
                conf.HOST_CRITIC_HIDDEN_DIM
            ),

            initial_alpha=(
                conf.HOST_INITIAL_ALPHA
            ),

            target_entropy_ratio=(
                conf.HOST_TARGET_ENTROPY_RATIO
            ),

            max_grad_norm=(
                conf.HOST_MAX_GRAD_NORM
            ),

            policy_update_interval=(
                conf.HOST_POLICY_UPDATE_INTERVAL
            ),

            target_update_interval=(
                conf.HOST_TARGET_UPDATE_INTERVAL
            ),

            device=(
                conf.DEVICE
            ),

            seed=(
                conf.Seed
            ),
        )
    )

    routing_masac_config = (
        RoutingMASACConfig(
            gamma=(
                conf.ROUTING_GAMMA
            ),

            tau=(
                conf.ROUTING_TAU
            ),

            actor_lr=(
                conf.ROUTING_ACTOR_LR
            ),

            critic_lr=(
                conf.ROUTING_CRITIC_LR
            ),

            alpha_lr=(
                conf.ROUTING_ALPHA_LR
            ),

            actor_hidden_dim=(
                conf.ROUTING_ACTOR_HIDDEN_DIM
            ),

            critic_hidden_dim=(
                conf.ROUTING_CRITIC_HIDDEN_DIM
            ),

            initial_alpha=(
                conf.ROUTING_INITIAL_ALPHA
            ),

            target_entropy_ratio=(
                conf.ROUTING_TARGET_ENTROPY_RATIO
            ),

            max_grad_norm=(
                conf.ROUTING_MAX_GRAD_NORM
            ),

            policy_update_interval=(
                conf.ROUTING_POLICY_UPDATE_INTERVAL
            ),

            target_update_interval=(
                conf.ROUTING_TARGET_UPDATE_INTERVAL
            ),

            device=(
                conf.DEVICE
            ),

            seed=(
                conf.Seed
            ),
        )
    )

    train(
        train_config=(
            train_config
        ),

        routing_masac_config=(
            routing_masac_config
        ),

        host_sac_config=(
            host_sac_config
        ),
    )

if __name__ == "__main__":
    main()













