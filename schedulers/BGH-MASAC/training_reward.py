
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


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

    # 当前 Reward 归一化需要的环境固定尺度。
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

        if (
                float(
                    config.energy_normalization_j
                )
                <= 0.0
        ):
            raise ValueError(
                "energy_normalization_j 必须 > 0。"
            )

        if (
                float(
                    config.max_latency_s
                )
                <= 0.0
        ):
            raise ValueError(
                "max_latency_s 必须 > 0。"
            )

        if (
                float(
                    config.max_job_duration_s
                )
                <= 0.0
        ):
            raise ValueError(
                "max_job_duration_s 必须 > 0。"
            )

    # ==========================================================
    # Helpers
    # ==========================================================

    def _normalize(
            self,
            value: float,
            scale: float,
    ) -> float:

        scale = max(
            float(scale),
            float(
                self.config.norm_eps
            ),
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
            float(
                self.config.norm_eps
            ),
        )

        completion_time_s = max(
            float(completion_time_s),
            0.0,
        )

        sla_limit_s = (
            float(
                self.config
                .sla_deadline_ratio
            )
            * job_duration_s
        )

        drop_limit_s = (
            float(
                self.config
                .drop_deadline_ratio
            )
            * job_duration_s
        )

        violation_window_s = max(
            drop_limit_s
            - sla_limit_s,
            float(
                self.config.norm_eps
            ),
        )

        violation_degree = (
            completion_time_s
            - sla_limit_s
        ) / violation_window_s

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

        energy_cost = (
            max(
                float(
                    attributable_energy_j
                ),
                0.0,
            )
            / float(
                self.config
                .energy_normalization_j
            )
        )

        return float(
            float(
                self.config
                .energy_cost_weight
            )
            * energy_cost
        )

    # ==========================================================
    # Routing Immediate Reward
    # ==========================================================

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

        action_type = str(
            facts[
                "action_type"
            ]
        )

        # ------------------------------------------------------
        # Self
        # ------------------------------------------------------

        if action_type == "self":
            return 0.0

        # ------------------------------------------------------
        # Forced / Environment Drop
        # ------------------------------------------------------

        if action_type == "drop":

            failure_reason = str(
                facts.get(
                    "failure_reason",
                    "",
                )
            )

            energy_penalty = (
                self._calculate_energy_penalty(
                    facts.get(
                        "total_attributable_energy_j",
                        0.0,
                    )
                )
            )

            if failure_reason in {
                "waiting_timeout",
                "等待超时",
            }:
                return -float(
                    float(
                        self.config
                        .timeout_drop_penalty
                    )
                    + energy_penalty
                )

            if failure_reason in {
                "resource_failure",
                "资源不足",
            }:
                return -float(
                    float(
                        self.config
                        .resource_drop_penalty
                    )
                    + energy_penalty
                )

            raise ValueError(
                "未知 Routing drop failure_reason："
                f"{failure_reason}"
            )

        # ------------------------------------------------------
        # Edge / Cloud Remote Routing
        # ------------------------------------------------------

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
                float(
                    facts[
                        "job_duration_s"
                    ]
                ),
                float(
                    self.config.norm_eps
                ),
            )

            remote_latency_cost = (
                self._normalize(
                    value=(
                        transfer_latency_s
                    ),

                    scale=(
                        self.config
                        .max_latency_s
                    ),
                )
            )

            predicted_completion_time_s = (
                elapsed_service_time_s
                + transfer_latency_s
                + job_duration_s
            )

            predicted_sla_violation_degree = (
                self
                ._calculate_sla_violation_degree(
                    job_duration_s=(
                        job_duration_s
                    ),

                    completion_time_s=(
                        predicted_completion_time_s
                    ),
                )
            )

            return -float(
                float(
                    self.config
                    .remote_offload_base_penalty
                )
                + float(
                    self.config
                    .remote_latency_cost_weight
                )
                * remote_latency_cost
                + float(
                    self.config
                    .sla_risk_cost_weight
                )
                * predicted_sla_violation_degree
            )

        raise ValueError(
            "未知 Routing action_type："
            f"{action_type}"
        )

    # ==========================================================
    # Delayed / Terminal Reward
    # ==========================================================

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

        reason = str(
            facts[
                "reason"
            ]
        )

        attributable_energy_j = float(
            facts.get(
                "total_attributable_energy_j",
                0.0,
            )
        )

        energy_penalty = (
            self._calculate_energy_penalty(
                attributable_energy_j
            )
        )

        # ------------------------------------------------------
        # Successful Completion
        # ------------------------------------------------------

        if reason == "completed":

            completion_time_s = max(
                float(
                    facts[
                        "completion_time_s"
                    ]
                ),
                0.0,
            )

            job_duration_s = max(
                float(
                    facts[
                        "job_duration_s"
                    ]
                ),
                float(
                    self.config.norm_eps
                ),
            )

            global_completion_scale_s = max(
                float(
                    self.config
                    .drop_deadline_ratio
                )
                * float(
                    self.config
                    .max_job_duration_s
                ),
                float(
                    self.config.norm_eps
                ),
            )

            completion_time_cost = (
                self._normalize(
                    value=(
                        completion_time_s
                    ),

                    scale=(
                        global_completion_scale_s
                    ),
                )
            )

            sla_violation_degree = (
                self
                ._calculate_sla_violation_degree(
                    job_duration_s=(
                        job_duration_s
                    ),

                    completion_time_s=(
                        completion_time_s
                    ),
                )
            )

            completion_time_penalty = (
                float(
                    self.config
                    .completion_time_cost_weight
                )
                * completion_time_cost
            )

            sla_violation_penalty = (
                float(
                    self.config
                    .sla_violation_cost_weight
                )
                * sla_violation_degree
            )

            return float(
                float(
                    self.config
                    .task_completion_reward
                )
                - completion_time_penalty
                - sla_violation_penalty
                - energy_penalty
            )

        # ------------------------------------------------------
        # Timeout Failure
        # ------------------------------------------------------

        if reason in {
            "waiting_timeout",
            "local_host_arrival_timeout",
            "cloud_arrival_timeout",
        }:
            return -float(
                float(
                    self.config
                    .timeout_drop_penalty
                )
                + energy_penalty
            )

        # ------------------------------------------------------
        # Resource Failure
        # ------------------------------------------------------

        if reason in {
            "local_host_resource_failure",
            "cloud_resource_failure",
        }:
            return -float(
                float(
                    self.config
                    .resource_drop_penalty
                )
                + energy_penalty
            )

        raise ValueError(
            "TrainingRewardModel "
            "收到未知 outcome reason："
            f"{reason}"
        )