"""Host and routing observations with historical feedback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from .experience import FinalizedJobTrace


# Local Host observation
class HostObservationBuilder:
    """
    H-MASAC Local Host Scheduling 专用 Observation Builder。

    Host Observation 只在 Routing Agent 选择 Self 后构造。

    Observation:

        Current Job
        +
        Current Local DC Context
        +
        Every Real Local Host State

    明确不包含：

        Remote DC state
        Routing topology
        Routing history
        Neighbor Historical Feedback
        Cloud
        Host padding
        Action mask

    Host 层不是 PettingZoo Agent。
    """

    JOB_FEAT_DIM = 5

    LOCAL_DC_FEAT_DIM = 5

    HOST_BASE_FEATURE_NAMES = (
        "cpu_capacity",
        "gpu_capacity",
        "cpu_load",
        "gpu_load",
        "available_cpu",
        "available_gpu",
        "waiting_queue_congestion",
        "waiting_workload_ratio",
        "running_queue_congestion",
    )

    HOST_BASE_FEAT_DIM = len(HOST_BASE_FEATURE_NAMES)

    HOST_JOB_EVAL_FEATURE_NAMES = (
        "can_ever_accommodate",
        "can_start_now",
        "estimated_start_delay_ratio",
        "estimated_completion_ratio",
    )

    HOST_JOB_EVAL_FEAT_DIM = len(HOST_JOB_EVAL_FEATURE_NAMES)

    HOST_FEATURE_NAMES = HOST_BASE_FEATURE_NAMES + HOST_JOB_EVAL_FEATURE_NAMES

    HOST_FEAT_DIM = HOST_BASE_FEAT_DIM + HOST_JOB_EVAL_FEAT_DIM

    def __init__(
        self,
        env: Any,
    ) -> None:

        self.env = env

        self.edge_dc_ids = [str(dc_id) for dc_id in env.edge_dc_ids]

        # ======================================================
        # Host 层只管理 Edge DC。
        #
        # Cloud 不进入 Local Host SAC；
        # Cloud 仍沿用环境中的自动执行逻辑。
        # ======================================================

        base_edge_dc_map = {
            str(dc.dc_id): dc for dc in env.base_datacenters if str(dc.dc_id) in self.edge_dc_ids
        }

        # ======================================================
        # 固定每个 DC 的 Host 顺序。
        #
        # 后续：
        #
        #     Host action 0
        #
        # 永远对应：
        #
        #     host_ids_by_dc[dc_id][0]
        #
        # Observation 中 Host block 顺序和 action 顺序
        # 必须严格一致。
        # ======================================================

        self.host_ids_by_dc: Dict[
            str,
            List[str],
        ] = {}

        self.host_count_by_dc: Dict[
            str,
            int,
        ] = {}

        for dc_id in self.edge_dc_ids:

            if dc_id not in base_edge_dc_map:
                raise KeyError("Host Observation 找不到 Edge DC：" f"{dc_id}")

            dc = base_edge_dc_map[dc_id]

            host_ids = [str(host.host_id) for host in dc.host_list]

            if len(host_ids) == 0:
                raise ValueError(f"Edge DC {dc_id} 没有 Host。")

            self.host_ids_by_dc[dc_id] = host_ids

            self.host_count_by_dc[dc_id] = len(host_ids)

        # ======================================================
        # 每个 DC 独立 Observation / Action dimension。
        #
        # 不使用 max_host_num。
        # 不进行 zero padding。
        # ======================================================

        self.obs_dim_by_dc: Dict[
            str,
            int,
        ] = {
            dc_id: (
                self.JOB_FEAT_DIM
                + self.LOCAL_DC_FEAT_DIM
                + self.HOST_FEAT_DIM * self.host_count_by_dc[dc_id]
            )
            for dc_id in self.edge_dc_ids
        }

        self.action_dim_by_dc: Dict[
            str,
            int,
        ] = {dc_id: self.host_count_by_dc[dc_id] for dc_id in self.edge_dc_ids}

    def get_obs_dim(
        self,
        dc_id: str,
    ) -> int:

        dc_id = str(dc_id)

        if dc_id not in self.obs_dim_by_dc:
            raise KeyError(f"未知 Edge DC：{dc_id}")

        return int(self.obs_dim_by_dc[dc_id])

    def get_action_dim(
        self,
        dc_id: str,
    ) -> int:

        dc_id = str(dc_id)

        if dc_id not in self.action_dim_by_dc:
            raise KeyError(f"未知 Edge DC：{dc_id}")

        return int(self.action_dim_by_dc[dc_id])

    def get_host_ids(
        self,
        dc_id: str,
    ) -> List[str]:

        dc_id = str(dc_id)

        if dc_id not in self.host_ids_by_dc:
            raise KeyError(f"未知 Edge DC：{dc_id}")

        return list(self.host_ids_by_dc[dc_id])

    def _normalize(
        self,
        value: float,
        scale: float,
    ) -> float:

        eps = max(
            float(
                getattr(
                    self.env,
                    "norm_eps",
                    1e-8,
                )
            ),
            1e-12,
        )

        value = max(
            float(value),
            0.0,
        )

        scale = max(
            float(scale),
            eps,
        )

        return float(
            np.clip(
                value / scale,
                0.0,
                1.0,
            )
        )

    def _saturating_ratio(
        self,
        value: float,
        scale: float,
    ) -> float:

        eps = max(
            float(
                getattr(
                    self.env,
                    "norm_eps",
                    1e-8,
                )
            ),
            1e-12,
        )

        value = max(
            float(value),
            0.0,
        )

        scale = max(
            float(scale),
            eps,
        )

        return float(value / (value + scale))

    def _validate_host_feature_block(
        self,
        features: List[float],
        expected_dim: int,
        block_name: str,
    ) -> List[float]:
        """
        验证 Host Observation 子块。

        当前所有 Host-level features 都应为：
            finite
            且位于 [0, 1]

        如果未来加入没有归一化的物理量，
        必须先显式修改这里的约束，
        不能静默把 raw value 塞给网络。
        """

        feature_array = np.asarray(
            features,
            dtype=np.float32,
        )

        if feature_array.shape != (int(expected_dim),):
            raise ValueError(
                f"{block_name} 维度错误："
                f"expected={(int(expected_dim),)}, "
                f"actual={feature_array.shape}"
            )

        if not np.all(np.isfinite(feature_array)):
            raise ValueError(f"{block_name} 出现 NaN/Inf：" f"{features}")

        eps = 1e-6

        if np.any(feature_array < -eps) or np.any(feature_array > 1.0 + eps):
            raise ValueError(f"{block_name} 必须位于 [0,1]：" f"{features}")

        return [float(value) for value in features]

    def _encode_job(
        self,
        job: Any,
    ) -> List[float]:

        eps = max(
            float(
                getattr(
                    self.env,
                    "norm_eps",
                    1e-8,
                )
            ),
            1e-12,
        )

        duration = max(
            float(job.duration),
            eps,
        )

        elapsed_time = max(
            float(self.env.current_time) - float(job.arrive_time),
            0.0,
        )

        sla_budget = max(
            float(self.env.sla_deadline_ratio) * duration,
            eps,
        )

        drop_budget = max(
            float(self.env.drop_deadline_ratio) * duration,
            eps,
        )

        features = [
            self._normalize(
                job.cpu_request,
                self.env.max_job_cpu,
            ),
            self._normalize(
                job.gpu_request,
                self.env.max_job_gpu,
            ),
            self._normalize(
                duration,
                self.env.max_job_duration,
            ),
            float(
                np.clip(
                    elapsed_time / sla_budget,
                    0.0,
                    1.0,
                )
            ),
            float(
                np.clip(
                    elapsed_time / drop_budget,
                    0.0,
                    1.0,
                )
            ),
        ]

        if len(features) != self.JOB_FEAT_DIM:
            raise ValueError(
                "Host Job feature dimension error："
                f"expected={self.JOB_FEAT_DIM}, "
                f"actual={len(features)}"
            )

        return features

    def _encode_local_dc(
        self,
        local_dc: Any,
    ) -> List[float]:

        hosts = list(local_dc.host_list)

        host_count = max(
            len(hosts),
            1,
        )

        local_dc.calculate_dc_loads()

        waiting_jobs = sum(len(host.waiting_queue) for host in hosts)

        running_jobs = sum(len(host.running_queue) for host in hosts)

        waiting_workload = sum(float(host.waiting_queue.get_total_duration()) for host in hosts)

        queue_length_scale = float(self.env.queue_length_scale) * host_count

        queue_workload_scale = float(self.env.queue_workload_scale) * host_count

        features = [
            float(
                np.clip(
                    local_dc.dc_cpu_load,
                    0.0,
                    1.0,
                )
            ),
            float(
                np.clip(
                    local_dc.dc_gpu_load,
                    0.0,
                    1.0,
                )
            ),
            self._saturating_ratio(
                waiting_jobs,
                queue_length_scale,
            ),
            self._saturating_ratio(
                running_jobs,
                queue_length_scale,
            ),
            self._saturating_ratio(
                waiting_workload,
                queue_workload_scale,
            ),
        ]

        if len(features) != self.LOCAL_DC_FEAT_DIM:
            raise ValueError(
                "Host Local DC feature dimension error："
                f"expected={self.LOCAL_DC_FEAT_DIM}, "
                f"actual={len(features)}"
            )

        return features

    def _estimate_host_start_delay(
        self,
        host: Any,
        job: Any,
    ) -> Optional[float]:

        # Host 的物理总容量永远无法执行该任务。
        if not host.can_ever_accommodate(job):
            return None

        # Waiting Queue 为空且当前资源足够，
        # 根据现有环境语义可以立即执行。
        if host.waiting_queue.is_empty() and host.can_accommodate(job):
            return 0.0

        max_running_remaining = 0.0

        for running_job in list(
            getattr(
                host.running_queue,
                "_queue",
                [],
            )
        ):

            if running_job.start_time is None:
                remaining_time = float(running_job.duration)

            else:
                remaining_time = max(
                    float(running_job.start_time)
                    + float(running_job.duration)
                    - float(self.env.current_time),
                    0.0,
                )

            max_running_remaining = max(
                max_running_remaining,
                remaining_time,
            )

        waiting_workload = float(host.waiting_queue.get_total_duration())

        return float(max_running_remaining + waiting_workload)

    def _encode_host_base_features(
        self,
        host: Any,
    ) -> List[float]:
        """
        编码单台 Host 自身的 9 维实时状态。

        这里不读取当前 Job，
        只表示 Host 自己当前是什么状态。
        """

        # 更新当前 CPU/GPU load。
        host.calculate_load()

        available_cpu = float(host.get_available_cpu())

        available_gpu = float(host.get_available_gpu())

        waiting_jobs = len(host.waiting_queue)

        running_jobs = len(host.running_queue)

        waiting_workload = float(host.waiting_queue.get_total_duration())

        features = [
            # ------------------------------------------------------
            # 0~1：Host physical capacity
            # ------------------------------------------------------
            self._normalize(
                host.cpu_num,
                self.env.max_host_cpu,
            ),
            self._normalize(
                host.gpu_capacity_num,
                self.env.max_host_gpu,
            ),
            # ------------------------------------------------------
            # 2~3：Current utilization
            # ------------------------------------------------------
            float(
                np.clip(
                    host.cpu_load,
                    0.0,
                    1.0,
                )
            ),
            float(
                np.clip(
                    host.gpu_load,
                    0.0,
                    1.0,
                )
            ),
            # ------------------------------------------------------
            # 4~5：Current available resources
            # ------------------------------------------------------
            self._normalize(
                available_cpu,
                self.env.max_host_cpu,
            ),
            self._normalize(
                available_gpu,
                self.env.max_host_gpu,
            ),
            # ------------------------------------------------------
            # 6~8：Queue / execution congestion
            # ------------------------------------------------------
            self._saturating_ratio(
                waiting_jobs,
                self.env.queue_length_scale,
            ),
            self._saturating_ratio(
                waiting_workload,
                self.env.queue_workload_scale,
            ),
            self._saturating_ratio(
                running_jobs,
                self.env.queue_length_scale,
            ),
        ]

        return self._validate_host_feature_block(
            features=features,
            expected_dim=self.HOST_BASE_FEAT_DIM,
            block_name="Host Base State",
        )

    def _encode_host_job_eval_features(
        self,
        host: Any,
        job: Any,
    ) -> List[float]:
        """
        编码当前 Job 与当前 Host 的 4 维匹配关系。

        这些值只是 Observation Features，
        不会修改 Host action legality。
        """

        eps = max(
            float(
                getattr(
                    self.env,
                    "norm_eps",
                    1e-8,
                )
            ),
            1e-12,
        )

        # ==========================================================
        # 1. Physical feasibility
        #
        # Host 总 CPU/GPU 容量是否永远有可能执行 Job。
        # ==========================================================

        can_ever = bool(host.can_ever_accommodate(job))

        # ==========================================================
        # 2. Immediate-start feasibility
        #
        # 必须同时：
        #
        #     Waiting Queue 为空
        #     +
        #     当前剩余 CPU/GPU 足够
        #
        # 这与当前环境 _execute_job_on_host() 的实际规则一致。
        # ==============================================================

        can_start_now = bool(host.waiting_queue.is_empty() and host.can_accommodate(job))

        # ==========================================================
        # 3. Estimated start delay
        # ==========================================================

        estimated_start_delay = self._estimate_host_start_delay(
            host=host,
            job=job,
        )

        duration = max(
            float(job.duration),
            eps,
        )

        elapsed_time = max(
            float(self.env.current_time) - float(job.arrive_time),
            0.0,
        )

        if estimated_start_delay is None:

            # Host 物理资源永远不支持当前 Job。
            #
            # 不做 Mask，而是把两个风险指标显式置到最高。
            estimated_start_delay_ratio = 1.0
            estimated_completion_ratio = 1.0

        else:

            # ======================================================
            # 可用于等待的预算：
            #
            #     (DROP_RATIO - 1) × duration
            #
            # 因为最终执行本身至少还需要一个 duration。
            # ======================================================

            wait_budget = max(
                (float(self.env.drop_deadline_ratio) - 1.0) * duration,
                eps,
            )

            estimated_start_delay_ratio = float(
                np.clip(
                    float(estimated_start_delay) / wait_budget,
                    0.0,
                    1.0,
                )
            )

            # ======================================================
            # 从最初进入系统到预计完成的总服务时间。
            # ======================================================

            estimated_completion_time = elapsed_time + float(estimated_start_delay) + duration

            drop_completion_limit = max(
                float(self.env.drop_deadline_ratio) * duration,
                eps,
            )

            estimated_completion_ratio = float(
                np.clip(
                    estimated_completion_time / drop_completion_limit,
                    0.0,
                    1.0,
                )
            )

        features = [
            1.0 if can_ever else 0.0,
            1.0 if can_start_now else 0.0,
            estimated_start_delay_ratio,
            estimated_completion_ratio,
        ]

        return self._validate_host_feature_block(
            features=features,
            expected_dim=(self.HOST_JOB_EVAL_FEAT_DIM),
            block_name=("Host Current-Job Evaluation"),
        )

    def _encode_host(
        self,
        host: Any,
        job: Any,
    ) -> List[float]:
        """
        构造单台 Host 的最终固定 13 维特征：

            9-dimensional Host State
            +
            4-dimensional Job-Host Matching
        """

        base_features = self._encode_host_base_features(
            host=host,
        )

        job_eval_features = self._encode_host_job_eval_features(
            host=host,
            job=job,
        )

        features = base_features + job_eval_features

        return self._validate_host_feature_block(
            features=features,
            expected_dim=self.HOST_FEAT_DIM,
            block_name=(f"Host Feature " f"(host_id={host.host_id})"),
        )

    def build(
        self,
        dc_id: str,
        job_id: str,
    ) -> np.ndarray:

        dc_id = str(dc_id)
        job_id = str(job_id)

        # ======================================================
        # Host 层只允许 Edge DC。
        # ======================================================

        if dc_id not in self.edge_dc_ids:
            raise ValueError("Host Observation 只能为 Edge DC 构造：" f"dc_id={dc_id}")

        if dc_id not in self.env.dc_map:
            raise KeyError("当前 Episode 中不存在 DC：" f"{dc_id}")

        if job_id not in self.env.job_map:
            raise KeyError("当前 Episode 中不存在 Job：" f"{job_id}")

        local_dc = self.env.dc_map[dc_id]

        job = self.env.job_map[job_id]

        # ======================================================
        # Host 数量和顺序属于网络结构的一部分。
        #
        # 每个 Episode 的硬件环境必须保持一致。
        # ======================================================

        runtime_host_ids = [str(host.host_id) for host in local_dc.host_list]

        expected_host_ids = self.host_ids_by_dc[dc_id]

        if runtime_host_ids != expected_host_ids:
            raise RuntimeError(
                "Host 列表发生变化，"
                "会破坏 Local Host SAC 的输入/动作映射："
                f"dc={dc_id}, "
                f"expected={expected_host_ids}, "
                f"actual={runtime_host_ids}"
            )

        obs: List[float] = []

        # 1. Current Job
        obs.extend(self._encode_job(job))

        # 2. Local DC context
        obs.extend(self._encode_local_dc(local_dc))

        # 3. Every real local Host
        #
        # 不 padding。
        # 不读取其他 DC。
        for host_index, host in enumerate(local_dc.host_list):

            host_features = self._encode_host(
                host=host,
                job=job,
            )

            if len(host_features) != self.HOST_FEAT_DIM:
                raise ValueError(
                    "Host block dimension error："
                    f"dc={dc_id}, "
                    f"host_index={host_index}, "
                    f"host_id={host.host_id}, "
                    f"expected={self.HOST_FEAT_DIM}, "
                    f"actual={len(host_features)}"
                )

            obs.extend(host_features)

        obs_array = np.asarray(
            obs,
            dtype=np.float32,
        )

        expected_dim = self.get_obs_dim(dc_id)

        if obs_array.shape != (expected_dim,):

            raise ValueError(
                "Host Observation dimension error："
                f"dc={dc_id}, "
                f"expected={(expected_dim,)}, "
                f"actual={obs_array.shape}"
            )

        if not np.all(np.isfinite(obs_array)):

            raise ValueError("Host Observation 出现 NaN/Inf：" f"dc={dc_id}, " f"job={job_id}")

        return obs_array

    def build_pending(
        self,
    ) -> np.ndarray:
        """
        为当前 Routing=Self 后等待 Host 决策的任务
        构造 Host Observation。
        """

        job_id = getattr(
            self.env,
            "pending_host_job_id",
            None,
        )

        dc_id = getattr(
            self.env,
            "pending_host_dc_id",
            None,
        )

        if job_id is None:
            raise RuntimeError("当前不存在 pending Host Job。")

        if dc_id is None:
            raise RuntimeError("当前不存在 pending Host DC。")

        dc_id = str(dc_id)
        job_id = str(job_id)

        # Host Decision 发生时 Routing AEC 应处于暂停状态。
        if self.env.agent_selection is not None:
            raise RuntimeError(
                "当前仍存在 PettingZoo Routing Agent，"
                "不能同时执行 Host Decision："
                f"agent_selection="
                f"{self.env.agent_selection}"
            )

        return self.build(
            dc_id=dc_id,
            job_id=job_id,
        )


# Routing observation
class RoutingObservationBuilder:

    # job特征，CPU请求、GPU请求、执行时长、已消耗SLA预算比例、已消耗丢弃预算比例
    JOB_FEAT_DIM = 5
    # 本地DC状态
    LOCAL_DC_BASE_FEAT_DIM = 8
    LOCAL_DC_ROUTING_FEAT_DIM = 5
    # 任务调度历史
    ROUTE_HISTORY_FEAT_DIM = 3
    # 任务执行反馈
    FEEDBACK_FEAT_PER_DC = 7
    FEEDBACK_FEATURE_NAMES = (
        "success_ewma",
        "sla_success_ewma",
        "drop_ewma",
        "completion_time_ewma",
        "reforward_ewma",
        "feedback_age",
        "sample_confidence",
    )

    def __init__(
        self,
        env: Any,
        use_neighbor_historical_feedback: bool = False,
        neighbor_feedback_provider: Optional[Any] = None,
    ) -> None:

        self.env = env
        self.edge_dc_ids = [str(dc_id) for dc_id in env.edge_dc_ids]
        self.link_target_dc_ids = list(self.edge_dc_ids) + [str(env.cloud_id)]
        # self.job_edge_hop_counts: Dict[str, int] = {}
        # self.job_edge_transfer_latency_s: Dict[str, float,] = {}
        self.job_feat_dim = self.JOB_FEAT_DIM
        self.local_dc_feat_dim = self.LOCAL_DC_BASE_FEAT_DIM + self.LOCAL_DC_ROUTING_FEAT_DIM
        self.route_history_feat_dim = self.ROUTE_HISTORY_FEAT_DIM
        self.link_feat_dim = len(self.link_target_dc_ids)
        self.use_neighbor_historical_feedback = bool(use_neighbor_historical_feedback)
        self.neighbor_feedback_provider = neighbor_feedback_provider
        self.feedback_feat_dim = len(self.edge_dc_ids) * self.FEEDBACK_FEAT_PER_DC
        self.obs_dim = (
            self.job_feat_dim
            + self.local_dc_feat_dim
            + self.route_history_feat_dim
            + self.link_feat_dim
            + self.feedback_feat_dim
        )

    def checkpoint_metadata(self) -> Dict[str, Any]:
        """Protect input semantics as well as dimensions on checkpoint restore."""
        return {
            "schema": "h_masac_local_routing_v1",
            "edge_dc_ids": list(self.edge_dc_ids),
            "link_target_dc_ids": list(self.link_target_dc_ids),
            "obs_dim": self.obs_dim,
            "job_feat_dim": self.job_feat_dim,
            "local_dc_feat_dim": self.local_dc_feat_dim,
            "route_history_feat_dim": self.route_history_feat_dim,
            "feedback_feature_names": list(self.FEEDBACK_FEATURE_NAMES),
            "use_neighbor_historical_feedback": self.use_neighbor_historical_feedback,
        }

    # 注入 Neighbor Historical Feedback 数据提供器
    def set_neighbor_feedback_provider(
        self,
        provider: Optional[Any],
    ) -> None:
        """
        注入 Neighbor Historical Feedback 数据提供器。

        Provider 必须只读取历史统计数据，
        不能读取目标 DC 当前实时资源状态。
        """

        self.neighbor_feedback_provider = provider

    # 归一化辅助函数
    def _normalize(
        self,
        value: float,
        scale: float,
    ) -> float:

        eps = max(
            float(
                getattr(
                    self.env,
                    "norm_eps",
                    1e-8,
                )
            ),
            1e-12,
        )

        value = max(float(value), 0.0)
        scale = max(float(scale), eps)

        return float(
            np.clip(
                value / scale,
                0.0,
                1.0,
            )
        )

    # 无上界归一化函数
    def _saturating_ratio(
        self,
        value: float,
        scale: float,
    ) -> float:

        eps = max(
            float(
                getattr(
                    self.env,
                    "norm_eps",
                    1e-8,
                )
            ),
            1e-12,
        )

        value = max(float(value), 0.0)
        scale = max(float(scale), eps)

        return float(value / (value + scale))

    # 如果选择此host要等多久
    def _estimate_host_start_delay(
        self,
        host: Any,
        job: Any,
    ) -> Optional[float]:

        # 资源不满足
        if not host.can_ever_accommodate(job):
            return None
        # 机器为空
        if host.waiting_queue.is_empty() and host.can_accommodate(job):
            return 0.0

        max_running_remaining = 0.0
        running_jobs = getattr(
            host.running_queue,
            "_queue",
            [],
        )

        for running_job in running_jobs:
            if running_job.start_time is None:
                remaining_time = float(running_job.duration)

            else:
                remaining_time = max(
                    float(running_job.start_time)
                    + float(running_job.duration)
                    - float(self.env.current_time),
                    0.0,
                )

            max_running_remaining = max(
                max_running_remaining,
                remaining_time,
            )

        waiting_workload = float(host.waiting_queue.get_total_duration())

        return float(max_running_remaining + waiting_workload)

    def encode_job_features(
        self,
        job: Any,
    ) -> List[float]:
        return self._encode_job(job)

    def encode_dc_aggregate_features(
        self,
        dc: Any,
        job: Any,
    ) -> List[float]:
        return self._encode_local_dc(
            local_dc=dc,
            job=job,
        )

    def encode_route_history_features(
        self,
        job: Any,
    ) -> List[float]:
        return self._encode_route_history(job)

    #
    def build(
        self,
        agent_id: str,
    ) -> np.ndarray:

        agent_id = str(agent_id)

        job_id = str(self.env.current_job_id)
        job = self.env.job_map[job_id]
        local_dc = self.env.dc_map[agent_id]
        obs: List[float] = []

        # 当前 Job
        obs.extend(self._encode_job(job))

        # 当前 DC 聚合状态
        local_dc_features = self._encode_local_dc(
            local_dc=local_dc,
            job=job,
        )
        obs.extend(local_dc_features)

        # 当前 Job 多跳 Routing 历史
        obs.extend(
            self._encode_route_history(
                job,
            )
        )

        # 当前 DC 到其他 DC / Cloud 的链路
        link_features = self._encode_links(
            agent_id=agent_id,
        )
        obs.extend(link_features)

        # Neighbor Historical Feedback
        feedback_features = self._encode_neighbor_feedback(
            agent_id=agent_id,
        )
        obs.extend(feedback_features)

        obs_array = np.asarray(
            obs,
            dtype=np.float32,
        )

        return obs_array

    #
    def _encode_local_dc(
        self,
        local_dc: Any,
        job: Any,
    ) -> List[float]:

        hosts = list(local_dc.host_list)
        host_count = len(hosts)

        # 更新所有 Host / DC 的当前负载。
        local_dc.calculate_dc_loads()

        total_cpu = sum(float(host.cpu_num) for host in hosts)
        total_gpu = sum(float(host.gpu_capacity_num) for host in hosts)
        used_cpu = sum(float(host.used_cpu) for host in hosts)
        used_gpu = sum(float(host.used_gpu) for host in hosts)

        available_cpu = max(
            total_cpu - used_cpu,
            0.0,
        )
        available_gpu = max(
            total_gpu - used_gpu,
            0.0,
        )

        waiting_jobs = sum(len(host.waiting_queue) for host in hosts)
        running_jobs = sum(len(host.running_queue) for host in hosts)
        waiting_workload = sum(float(host.waiting_queue.get_total_duration()) for host in hosts)

        host_scale = max(
            host_count,
            1,
        )
        dc_queue_length_scale = float(self.env.queue_length_scale) * host_scale
        dc_queue_workload_scale = float(self.env.queue_workload_scale) * host_scale
        immediate_feasible_count = 0
        ever_feasible_count = 0

        best_start_delay = None

        for host in hosts:
            # 物理总容量是否能够运行当前 Job。
            if host.can_ever_accommodate(job):
                ever_feasible_count += 1
                estimated_delay = self._estimate_host_start_delay(
                    host=host,
                    job=job,
                )

                if estimated_delay is not None:
                    if best_start_delay is None or estimated_delay < best_start_delay:
                        best_start_delay = estimated_delay

            if host.waiting_queue.is_empty() and host.can_accommodate(job):
                immediate_feasible_count += 1

        host_count_float = float(max(host_count, 1))
        immediate_feasible_host_ratio = float(immediate_feasible_count) / host_count_float
        ever_feasible_host_ratio = float(ever_feasible_count) / host_count_float

        duration = max(
            float(job.duration),
            float(
                getattr(
                    self.env,
                    "norm_eps",
                    1e-8,
                )
            ),
        )

        elapsed_time = max(
            float(self.env.current_time) - float(job.arrive_time),
            0.0,
        )

        # 如果没有任何 Host 的物理容量能够执行 Job，
        # 直接把两个 local-risk 指标置为最高风险。
        if best_start_delay is None:
            best_local_start_delay_ratio = 1.0
            best_local_completion_ratio = 1.0

        else:
            local_wait_budget = max(
                (float(self.env.drop_deadline_ratio) - 1.0) * duration,
                float(
                    getattr(
                        self.env,
                        "norm_eps",
                        1e-8,
                    )
                ),
            )
            best_local_start_delay_ratio = float(
                np.clip(
                    best_start_delay / local_wait_budget,
                    0.0,
                    1.0,
                )
            )

            # ------------------------------------------------------
            # 从 Job 最初到达到预计执行完成的总时间：
            #
            # elapsed
            # + estimated local waiting
            # + execution duration
            # ------------------------------------------------------
            predicted_local_completion = elapsed_time + float(best_start_delay) + duration

            drop_completion_limit = max(
                float(self.env.drop_deadline_ratio) * duration,
                float(
                    getattr(
                        self.env,
                        "norm_eps",
                        1e-8,
                    )
                ),
            )

            best_local_completion_ratio = float(
                np.clip(
                    predicted_local_completion / drop_completion_limit,
                    0.0,
                    1.0,
                )
            )

        # ==========================================================
        # 5. Final 13-dimensional Local DC Observation
        # ==========================================================

        features = [
            # ---------------- Base 8 ----------------
            self._normalize(
                total_cpu,
                self.env.max_dc_cpu,
            ),
            self._normalize(
                total_gpu,
                self.env.max_dc_gpu,
            ),
            float(
                np.clip(
                    local_dc.dc_cpu_load,
                    0.0,
                    1.0,
                )
            ),
            float(
                np.clip(
                    local_dc.dc_gpu_load,
                    0.0,
                    1.0,
                )
            ),
            self._normalize(
                available_cpu,
                self.env.max_dc_cpu,
            ),
            self._normalize(
                available_gpu,
                self.env.max_dc_gpu,
            ),
            self._saturating_ratio(
                value=float(waiting_jobs),
                scale=dc_queue_length_scale,
            ),
            self._saturating_ratio(
                value=float(running_jobs),
                scale=dc_queue_length_scale,
            ),
            # ------------- Routing-specific 5 -------------
            self._saturating_ratio(
                value=float(waiting_workload),
                scale=dc_queue_workload_scale,
            ),
            float(
                np.clip(
                    immediate_feasible_host_ratio,
                    0.0,
                    1.0,
                )
            ),
            float(
                np.clip(
                    ever_feasible_host_ratio,
                    0.0,
                    1.0,
                )
            ),
            float(
                np.clip(
                    best_local_start_delay_ratio,
                    0.0,
                    1.0,
                )
            ),
            float(
                np.clip(
                    best_local_completion_ratio,
                    0.0,
                    1.0,
                )
            ),
        ]

        # ==========================================================
        # 强制维度检查。
        #
        # Local DC Observation 一旦维度改变，
        # 必须显式修改常量，而不能静默改变 Actor 输入结构。
        # ==========================================================

        if len(features) != self.local_dc_feat_dim:
            raise ValueError(
                "Local DC Routing Observation 维度错误："
                f"expected={self.local_dc_feat_dim}, "
                f"actual={len(features)}"
            )

        return features

    #
    def _encode_route_history(
        self,
        job: Any,
    ) -> List[float]:
        """
        将当前 Job 已经产生的 Edge Routing 历史
        压缩成固定 3 维特征。

        固定顺序：

            [0] routing_hop_ratio
            [1] cumulative_edge_latency_ratio
            [2] cumulative_edge_energy_ratio

        不包含：
            visited_dc
            previous_dc
            route_path
            cycle_flag
        """

        job_id = str(job.job_id)

        eps = max(
            float(
                getattr(
                    self.env,
                    "norm_eps",
                    1e-8,
                )
            ),
            1e-12,
        )

        # ==========================================================
        # 1. Routing hop count
        # ==============================================================
        hop_count = max(
            int(
                getattr(
                    job,
                    "routing_hop_count",
                    0,
                )
            ),
            0,
        )

        # 不存在 MAX_HOPS，因此不能使用：
        #
        #     hop / MAX_HOPS
        #
        # 使用无上界 saturating encoding：
        #
        #     h / (h + 1)
        #
        # 0 -> 0
        # 1 -> 0.5
        # 2 -> 0.667
        # 3 -> 0.75
        # ...
        routing_hop_ratio = float(hop_count / (hop_count + 1.0))

        # ==========================================================
        # 2. Cumulative Edge -> Edge transmission latency
        #
        # 该数值由 Environment 在真实 Edge -> Edge
        # transmission event 被创建时更新。
        # ==========================================================

        cumulative_latency_s = max(
            float(
                getattr(
                    job,
                    "cumulative_transfer_latency_s",
                    0.0,
                )
            ),
            0.0,
        )

        # 使用系统“单链路最大时延”作为参考尺度。
        #
        # 不采用 hard clip：
        # 多跳累计时延超过单链路最大值后仍然能够继续区分。
        latency_scale_s = max(
            float(
                getattr(
                    self.env,
                    "max_latency",
                    1.0,
                )
            ),
            eps,
        )

        cumulative_latency_ratio = self._saturating_ratio(
            value=cumulative_latency_s,
            scale=latency_scale_s,
        )

        # ==========================================================
        # 3. Cumulative Edge -> Edge transmission energy
        #
        # 直接读取 Job 的物理 Energy ledger，
        # 不重复维护第二份能耗累计。
        # ==============================================================
        cumulative_energy_j = max(
            float(
                getattr(
                    job,
                    "edge_edge_transfer_energy_j",
                    0.0,
                )
            ),
            0.0,
        )

        energy_scale_j = max(
            float(
                getattr(
                    self.env,
                    "energy_normalization_j",
                    1.0,
                )
            ),
            eps,
        )

        cumulative_energy_ratio = self._saturating_ratio(
            value=cumulative_energy_j,
            scale=energy_scale_j,
        )

        features = [
            routing_hop_ratio,
            cumulative_latency_ratio,
            cumulative_energy_ratio,
        ]

        # ==========================================================
        # Routing History 必须始终严格保持 3 维。
        # ==============================================================
        if len(features) != self.route_history_feat_dim:
            raise ValueError(
                "Routing History 维度错误："
                f"expected={self.route_history_feat_dim}, "
                f"actual={len(features)}"
            )

        feature_array = np.asarray(
            features,
            dtype=np.float32,
        )

        if not np.all(np.isfinite(feature_array)):
            raise ValueError(
                "Routing History 出现 NaN/Inf：" f"job_id={job_id}, " f"features={features}"
            )

        if np.any(feature_array < 0.0) or np.any(feature_array > 1.0):
            raise ValueError(
                "Routing History 超出 [0,1]：" f"job_id={job_id}, " f"features={features}"
            )

        return features

    #
    def _encode_links(
        self,
        agent_id: str,
    ) -> List[float]:
        """
        构造当前 Routing Agent 的链路观测。

        Routing Actor 对其他 DC 只允许看到 Link Information。

        输出顺序固定为：

            Edge DC1
            Edge DC2
            ...
            Edge DCN
            Cloud

        每个目标只占 1 维：

            normalized transmission latency

        本函数绝对不能访问目标 DC 的：

            CPU/GPU load
            available resource
            waiting queue
            running queue
            Host state

        因此本函数中不会出现：

            self.env.dc_map[target_dc_id]

        这样的远端动态状态读取。
        """

        agent_id = str(agent_id)

        # ==========================================================
        # 1. 基础合法性检查
        # ==========================================================
        if agent_id not in self.edge_dc_ids:
            raise ValueError(
                "Routing Link Observation 只能为 Edge Agent 构造：" f"agent_id={agent_id}"
            )

        if self.env.graph is None:
            raise RuntimeError("当前环境 graph 尚未初始化，" "无法构造 Routing Link Observation。")

        link_features: List[float] = []

        # ==========================================================
        # 2. 固定顺序遍历所有 Routing destination
        #
        # 注意：
        # 这里只访问 graph。
        #
        # 不访问：
        #
        #     env.dc_map[target_dc_id]
        #
        # 因此不存在 Remote DC 实时负载泄漏。
        # ==========================================================
        for target_dc_id in self.link_target_dc_ids:

            target_dc_id = str(target_dc_id)

            # ------------------------------------------------------
            # 当前 DC 到自己的 Routing latency 定义为 0。
            #
            # 对应 Routing Action = Self。
            # ------------------------------------------------------
            if target_dc_id == agent_id:

                latency_s = 0.0

            # ------------------------------------------------------
            # 当前 DC -> Remote Edge / Cloud
            # ------------------------------------------------------
            else:

                if not self.env.graph.has_edge(
                    agent_id,
                    target_dc_id,
                ):
                    raise RuntimeError(
                        "Routing action space 中存在目标 DC，"
                        "但 topology 中缺少对应链路："
                        f"{agent_id} -> {target_dc_id}。"
                        "当前 H-MASAC 不使用 action mask，"
                        "因此所有可选 Routing destination "
                        "必须具有有效物理链路。"
                    )

                latency_s = max(
                    float(
                        self.env.graph[agent_id][target_dc_id].get(
                            "weight",
                            0.0,
                        )
                    ),
                    0.0,
                )

            # ------------------------------------------------------
            # 使用环境现有 max_latency 尺度。
            #
            # Link latency 本身有固定环境尺度，
            # 因此这里继续采用 max-scale normalization：
            #
            #     latency / max_latency
            #
            # 与 Routing History 中“累计多跳 latency”
            # 使用 saturating normalization 不同。
            # ------------------------------------------------------
            normalized_latency = self._normalize(
                value=latency_s,
                scale=float(self.env.max_latency),
            )

            link_features.append(normalized_latency)

        # ==========================================================
        # 3. 维度检查
        # ==========================================================
        if len(link_features) != self.link_feat_dim:
            raise ValueError(
                "Routing Link Observation 维度错误："
                f"expected={self.link_feat_dim}, "
                f"actual={len(link_features)}"
            )

        # ==========================================================
        # 4. 数值检查
        # ==========================================================
        link_array = np.asarray(
            link_features,
            dtype=np.float32,
        )

        if not np.all(np.isfinite(link_array)):
            raise ValueError(
                "Routing Link Observation 出现 NaN/Inf："
                f"agent_id={agent_id}, "
                f"features={link_features}"
            )

        if np.any(link_array < 0.0) or np.any(link_array > 1.0):
            raise ValueError(
                "Routing Link Observation 超出 [0,1]："
                f"agent_id={agent_id}, "
                f"features={link_features}"
            )

        return link_features

    #
    def _encode_job(
        self,
        job: Any,
    ) -> List[float]:
        """
        构造当前 Routing Job 的固定 5 维特征。

        顺序：

            [0] CPU request
            [1] GPU request
            [2] execution duration
            [3] consumed SLA budget ratio
            [4] consumed Drop budget ratio
        """

        eps = max(
            float(
                getattr(
                    self.env,
                    "norm_eps",
                    1e-8,
                )
            ),
            1e-12,
        )

        duration = max(
            float(job.duration),
            eps,
        )

        elapsed_time = max(
            float(self.env.current_time) - float(job.arrive_time),
            0.0,
        )

        sla_budget = max(
            float(self.env.sla_deadline_ratio) * duration,
            eps,
        )

        drop_budget = max(
            float(self.env.drop_deadline_ratio) * duration,
            eps,
        )

        sla_consumed_ratio = float(
            np.clip(
                elapsed_time / sla_budget,
                0.0,
                1.0,
            )
        )

        drop_consumed_ratio = float(
            np.clip(
                elapsed_time / drop_budget,
                0.0,
                1.0,
            )
        )

        features = [
            self._normalize(
                float(job.cpu_request),
                float(self.env.max_job_cpu),
            ),
            self._normalize(
                float(job.gpu_request),
                float(self.env.max_job_gpu),
            ),
            self._normalize(
                duration,
                float(self.env.max_job_duration),
            ),
            sla_consumed_ratio,
            drop_consumed_ratio,
        ]

        if len(features) != self.job_feat_dim:
            raise ValueError(
                "Routing Job Observation 维度错误："
                f"expected={self.job_feat_dim}, "
                f"actual={len(features)}"
            )

        return features

    #
    def _encode_neighbor_feedback(
        self,
        agent_id: str,
    ) -> List[float]:
        """
        构造固定长度的 Neighbor Historical Feedback block。

        固定顺序：

            DC1 的 7维
            DC2 的 7维
            ...
            DCN 的 7维

        USE_NEIGHBOR_HISTORICAL_FEEDBACK=False 时返回全 0；
        为 True 时从 Provider 读取历史统计。两种模式保持相同维度，
        因此切换开关不需要修改 Actor 网络结构。
        """

        agent_id = str(agent_id)

        # ==========================================================
        # Feedback disabled
        #
        # 关闭时不查询 Provider，不读取任何历史结果。
        # ==========================================================
        if not self.use_neighbor_historical_feedback:
            return [0.0 for _ in range(self.feedback_feat_dim)]

        # ==========================================================
        # Feedback enabled
        #
        # 如果用户打开开关却没有安装 Feedback Provider，
        # 必须直接报错，而不是偷偷继续返回全 0。
        # ==========================================================
        if self.neighbor_feedback_provider is None:
            raise RuntimeError(
                "USE_NEIGHBOR_HISTORICAL_FEEDBACK=True，" "但没有配置 Neighbor Feedback Provider。"
            )

        feedback_features: List[float] = []

        # ==========================================================
        # 每个 Edge DC 始终占固定 7 维。
        #
        # 使用全局固定 edge_dc_ids 顺序，
        # 对参数共享 Routing Actor 非常重要。
        # ==========================================================
        for target_dc_id in self.edge_dc_ids:

            target_dc_id = str(target_dc_id)

            # ------------------------------------------------------
            # 当前 Agent 自己并不是 Neighbor。
            #
            # 但仍然保留自己的固定 slot，
            # 这样所有 Agent 的输入布局完全一致。
            #
            # Self block 恒为 0。
            # ------------------------------------------------------
            if target_dc_id == agent_id:
                feedback_features.extend([0.0] * self.FEEDBACK_FEAT_PER_DC)

                continue

            # ------------------------------------------------------
            # 这里只允许 Provider 返回历史统计结果。
            # ------------------------------------------------------
            feedback = self.neighbor_feedback_provider.get_feedback(
                source_dc_id=agent_id,
                target_dc_id=target_dc_id,
            )

            # 尚未收集到任何历史样本：
            #
            # 7维全部置 0，
            # sample_confidence=0 同时能够表达“没有证据”。
            if feedback is None:
                feedback_features.extend([0.0] * self.FEEDBACK_FEAT_PER_DC)

                continue

            if not isinstance(
                feedback,
                Mapping,
            ):
                raise TypeError(
                    "Neighbor Feedback Provider 必须返回 " "Mapping[str, float] 或 None。"
                )

            block = [
                float(
                    feedback.get(
                        feature_name,
                        0.0,
                    )
                )
                for feature_name in self.FEEDBACK_FEATURE_NAMES
            ]

            # ------------------------------------------------------
            # Provider 输出必须已经归一化到 [0,1]。
            #
            # Observation Builder 不负责决定 EWMA 的统计尺度。
            # ------------------------------------------------------
            block_array = np.asarray(
                block,
                dtype=np.float32,
            )

            if not np.all(np.isfinite(block_array)):
                raise ValueError(
                    "Neighbor Feedback 出现 NaN/Inf："
                    f"source={agent_id}, "
                    f"target={target_dc_id}, "
                    f"block={block}"
                )

            if np.any(block_array < 0.0) or np.any(block_array > 1.0):
                raise ValueError(
                    "Neighbor Feedback 必须已经归一化到 [0,1]："
                    f"source={agent_id}, "
                    f"target={target_dc_id}, "
                    f"block={block}"
                )

            feedback_features.extend(block)

        # ==========================================================
        # 最终维度检查
        # ==========================================================
        if len(feedback_features) != self.feedback_feat_dim:
            raise ValueError(
                "Neighbor Feedback Observation 维度错误："
                f"expected={self.feedback_feat_dim}, "
                f"actual={len(feedback_features)}"
            )

        return feedback_features


# Centralized critic state
class RoutingCentralizedStateBuilder:
    """
    H-MASAC Routing Centralized Critic 专用全局状态。

    Centralized State 只在 CTDE training 阶段使用。

    Routing Actor 执行时永远不会读取本 Builder。

    State structure:

        Current Job
        +
        All Edge DC Aggregate States
        +
        Cloud Aggregate State
        +
        Current Job Routing History
        +
        Global Routing Topology

    明确不包含：

        Per-Host raw states
        Neighbor Historical Feedback
        visited DC
        route path
        action mask
    """

    def __init__(
        self,
        env: Any,
        routing_observation_builder: RoutingObservationBuilder,
    ) -> None:

        self.env = env

        self.routing_observation_builder = routing_observation_builder

        # ======================================================
        # Edge Routing Agents
        # ======================================================

        self.edge_dc_ids = [str(dc_id) for dc_id in env.edge_dc_ids]

        # ======================================================
        # 所有实际计算域。
        #
        # 即使 Cloud Action 关闭，
        # Cloud 物理环境仍存在，因此保持 State shape 不变。
        # ======================================================

        self.state_dc_ids = list(self.edge_dc_ids) + [str(env.cloud_id)]

        # ======================================================
        # Feature dimensions
        # ======================================================

        self.job_feat_dim = int(routing_observation_builder.job_feat_dim)

        self.dc_feat_dim = int(routing_observation_builder.local_dc_feat_dim)

        self.route_history_feat_dim = int(routing_observation_builder.route_history_feat_dim)

        # Routing topology:
        #
        # source:
        #     Edge DC only
        #
        # target:
        #     Edge DC + Cloud
        #
        # Cloud 自身不是 Routing Agent，
        # 因此没有 Cloud -> * 这一行。
        self.topology_source_dc_ids = list(self.edge_dc_ids)

        self.topology_target_dc_ids = list(self.state_dc_ids)

        self.topology_feat_dim = len(self.topology_source_dc_ids) * len(self.topology_target_dc_ids)

        # ======================================================
        # Final state dimension
        #
        # Job
        # + (Edge + Cloud) * DC aggregate
        # + Routing history
        # + Routing topology
        # ======================================================

        self.state_dim = (
            self.job_feat_dim
            + len(self.state_dc_ids) * self.dc_feat_dim
            + self.route_history_feat_dim
            + self.topology_feat_dim
        )

    def _encode_topology(
        self,
    ) -> List[float]:
        """
        构造训练阶段完整 Routing topology。

        顺序：

            DC1 -> [DC1 ... DCN Cloud]
            DC2 -> [DC1 ... DCN Cloud]
            ...
            DCN -> [DC1 ... DCN Cloud]

        所有值均为 normalized latency。
        """

        if self.env.graph is None:
            raise RuntimeError("环境 graph 尚未初始化，" "无法构造 Routing Centralized State。")

        features: List[float] = []

        for source_dc_id in self.topology_source_dc_ids:

            for target_dc_id in self.topology_target_dc_ids:

                source_dc_id = str(source_dc_id)

                target_dc_id = str(target_dc_id)

                # Self Routing 网络时延为 0。
                if source_dc_id == target_dc_id:

                    latency_s = 0.0

                else:

                    if not self.env.graph.has_edge(
                        source_dc_id,
                        target_dc_id,
                    ):
                        raise RuntimeError(
                            "Routing topology 缺少链路："
                            f"{source_dc_id}"
                            f" -> "
                            f"{target_dc_id}"
                        )

                    latency_s = max(
                        float(
                            self.env.graph[source_dc_id][target_dc_id].get(
                                "weight",
                                0.0,
                            )
                        ),
                        0.0,
                    )

                # 与 Actor Link Observation
                # 使用相同的 max_latency 归一化。
                normalized_latency = float(
                    np.clip(
                        latency_s
                        / max(
                            float(self.env.max_latency),
                            1e-8,
                        ),
                        0.0,
                        1.0,
                    )
                )

                features.append(normalized_latency)

        if len(features) != self.topology_feat_dim:

            raise ValueError(
                "Routing topology 维度错误："
                f"expected="
                f"{self.topology_feat_dim}, "
                f"actual="
                f"{len(features)}"
            )

        return features

    def build(
        self,
    ) -> np.ndarray:
        """
        构造当前 Routing Decision 对应的
        Centralized Training State。
        """

        if self.env.current_job_id is None:
            raise RuntimeError("当前不存在 Routing Job，" "无法构造 Centralized State。")

        job_id = str(self.env.current_job_id)

        if job_id not in self.env.job_map:
            raise KeyError(f"当前 Job 不存在：{job_id}")

        job = self.env.job_map[job_id]

        state: List[float] = []

        # ======================================================
        # 1. Current Job
        # ======================================================

        job_features = self.routing_observation_builder.encode_job_features(job)

        state.extend(job_features)

        # ======================================================
        # 2. All DC Aggregate States
        #
        # Critic 可以在 centralized training 时看到：
        #
        #     所有 Edge DC
        #     +
        #     Cloud
        #
        # 的实时 aggregate state。
        #
        # Actor 不会读取这里。
        # ======================================================

        for dc_id in self.state_dc_ids:

            dc = self.env.dc_map[str(dc_id)]

            dc_features = self.routing_observation_builder.encode_dc_aggregate_features(
                dc=dc,
                job=job,
            )

            if len(dc_features) != self.dc_feat_dim:
                raise ValueError(
                    "Centralized DC feature "
                    "维度错误："
                    f"dc={dc_id}, "
                    f"expected="
                    f"{self.dc_feat_dim}, "
                    f"actual="
                    f"{len(dc_features)}"
                )

            state.extend(dc_features)

        # ======================================================
        # 3. Current Job Routing History
        # ======================================================

        route_history = self.routing_observation_builder.encode_route_history_features(job)

        state.extend(route_history)

        # ======================================================
        # 4. Global Routing Topology
        # ======================================================

        state.extend(self._encode_topology())

        state_array = np.asarray(
            state,
            dtype=np.float32,
        )

        # ======================================================
        # 5. Final validation
        # ======================================================

        if state_array.shape != (self.state_dim,):
            raise ValueError(
                "Routing Centralized State "
                "维度错误："
                f"expected="
                f"{(self.state_dim,)}, "
                f"actual="
                f"{state_array.shape}"
            )

        if not np.all(np.isfinite(state_array)):
            raise ValueError("Routing Centralized State " "出现 NaN/Inf：" f"job_id={job_id}")

        return state_array


# Historical neighbor feedback
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

        self.edge_dc_ids = [str(dc_id) for dc_id in env.edge_dc_ids]

        self.edge_dc_id_set = set(self.edge_dc_ids)

        self.ewma_alpha = float(ewma_alpha)

        self.age_scale_samples = float(age_scale_samples)

        self.confidence_scale_samples = float(confidence_scale_samples)

        if not (0.0 < self.ewma_alpha <= 1.0):
            raise ValueError(
                "NEIGHBOR_FEEDBACK_EWMA_ALPHA " "必须位于 (0, 1]：" f"{self.ewma_alpha}"
            )

        if self.age_scale_samples <= 0.0:
            raise ValueError("NEIGHBOR_FEEDBACK_AGE_SCALE_SAMPLES " "必须 > 0。")

        if self.confidence_scale_samples <= 0.0:
            raise ValueError("NEIGHBOR_FEEDBACK_CONFIDENCE_SCALE_SAMPLES " "必须 > 0。")

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
            float(env.max_job_duration) * float(env.drop_deadline_ratio),
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

        return float(value / (value + scale))

    def _update_binary_ewma(
        self,
        previous_value: float,
        sample_value: float,
        previous_sample_count: int,
    ) -> float:

        sample_value = self._clip01(sample_value)

        # 第一条样本直接作为初始值，
        # 避免从人为的 0 开始产生额外初始化偏置。
        if previous_sample_count <= 0:
            return sample_value

        return float(
            (1.0 - self.ewma_alpha) * float(previous_value) + self.ewma_alpha * sample_value
        )

    def _get_or_create_pair(
        self,
        source_dc_id: str,
        target_dc_id: str,
    ) -> NeighborPairFeedbackState:

        source_dc_id = str(source_dc_id)

        target_dc_id = str(target_dc_id)

        if source_dc_id not in self.edge_dc_id_set:
            raise ValueError("Neighbor Feedback 收到未知 source DC：" f"{source_dc_id}")

        if target_dc_id not in self.edge_dc_id_set:
            raise ValueError("Neighbor Feedback 收到未知 target DC：" f"{target_dc_id}")

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

        pair_state = self._pair_states.get(pair_key)

        if pair_state is None:

            pair_state = NeighborPairFeedbackState(
                source_dc_id=(source_dc_id),
                target_dc_id=(target_dc_id),
            )

            self._pair_states[pair_key] = pair_state

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

        pair_state = self._get_or_create_pair(
            source_dc_id=(source_dc_id),
            target_dc_id=(target_dc_id),
        )

        previous_sample_count = int(pair_state.sample_count)

        pair_state.success_ewma = self._update_binary_ewma(
            previous_value=(pair_state.success_ewma),
            sample_value=(success_sample),
            previous_sample_count=(previous_sample_count),
        )

        pair_state.sla_success_ewma = self._update_binary_ewma(
            previous_value=(pair_state.sla_success_ewma),
            sample_value=(sla_success_sample),
            previous_sample_count=(previous_sample_count),
        )

        pair_state.drop_ewma = self._update_binary_ewma(
            previous_value=(pair_state.drop_ewma),
            sample_value=(drop_sample),
            previous_sample_count=(previous_sample_count),
        )

        pair_state.reforward_ewma = self._update_binary_ewma(
            previous_value=(pair_state.reforward_ewma),
            sample_value=(reforward_sample),
            previous_sample_count=(previous_sample_count),
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
                float(completion_time_s),
                0.0,
            )

            if pair_state.completion_sample_count <= 0 or pair_state.completion_time_ewma_s is None:
                pair_state.completion_time_ewma_s = completion_time_s

            else:
                pair_state.completion_time_ewma_s = float(
                    (1.0 - self.ewma_alpha) * float(pair_state.completion_time_ewma_s)
                    + self.ewma_alpha * completion_time_s
                )

            pair_state.completion_sample_count += 1

        pair_state.sample_count += 1

        pair_state.last_feedback_clock = int(self._feedback_clock)

        pair_state.last_job_id = str(job_id)

        pair_state.last_terminal_reason = str(terminal_reason)

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

        job_id = str(finalized_trace.job_id)

        job = self.env.job_map.get(job_id)

        if job is None:
            raise RuntimeError("Neighbor Feedback 无法找到 terminal Job：" f"{job_id}")

        terminal_reason = str(finalized_trace.terminal_reason)

        completed = bool(terminal_reason == "completed")

        success_sample = 1.0 if completed else 0.0

        drop_sample = 0.0 if completed else 1.0

        completion_time_s: Optional[float] = None

        sla_success_sample = 0.0

        if completed:

            turnaround_time = job.get_turnaround_time()

            # 正常 completed Job 已有 finish_time。
            # 这里保留 terminal_time fallback，
            # 防止以后环境实现变化。
            if turnaround_time is None:

                turnaround_time = max(
                    float(finalized_trace.terminal_time) - float(job.arrive_time),
                    0.0,
                )

            completion_time_s = max(
                float(turnaround_time),
                0.0,
            )

            sla_limit_s = float(self.env.sla_deadline_ratio) * float(job.duration)

            sla_success_sample = 1.0 if (completion_time_s <= sla_limit_s + 1e-9) else 0.0

        routing_steps = list(finalized_trace.routing_steps)

        pair_update_count = 0

        for step_index, routing_step in enumerate(routing_steps):

            if str(routing_step.action_type) != "edge_dc":
                continue

            source_dc_id = str(routing_step.source_dc_id)

            if routing_step.target_dc_id is None:
                raise RuntimeError(
                    "Finalized Edge Routing "
                    "缺少 target_dc_id："
                    f"job={job_id}, "
                    f"sequence="
                    f"{routing_step.sequence_index}"
                )

            target_dc_id = str(routing_step.target_dc_id)

            # ==================================================
            # Edge Routing 的后继 Routing Decision
            #
            # PendingJobTrace 已经保证：
            #   Edge -> Edge 必须有同 Job successor。
            #
            # 因此这里使用下一 Routing Step 判断：
            #   target 是否再次向外调度。
            # ==================================================

            next_step_index = step_index + 1

            if next_step_index >= len(routing_steps):
                raise RuntimeError(
                    "Neighbor Feedback 找不到 "
                    "Edge Routing 的后继 Routing Step："
                    f"job={job_id}, "
                    f"sequence="
                    f"{routing_step.sequence_index}, "
                    f"source={source_dc_id}, "
                    f"target={target_dc_id}"
                )

            next_routing_step = routing_steps[next_step_index]

            if str(next_routing_step.agent_id) != target_dc_id:
                raise RuntimeError(
                    "Neighbor Feedback 的 Routing "
                    "因果链 target 不一致："
                    f"job={job_id}, "
                    f"expected={target_dc_id}, "
                    f"actual="
                    f"{next_routing_step.agent_id}"
                )

            next_action_type = str(next_routing_step.action_type)

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
                source_dc_id=(source_dc_id),
                target_dc_id=(target_dc_id),
                job_id=(job_id),
                terminal_reason=(terminal_reason),
                success_sample=(success_sample),
                sla_success_sample=(sla_success_sample),
                drop_sample=(drop_sample),
                completion_time_s=(completion_time_s),
                reforward_sample=(reforward_sample),
            )

            pair_update_count += 1

        return int(pair_update_count)

    # ==========================================================
    # RoutingObservation Provider Interface
    # ==========================================================

    def get_feedback(
        self,
        source_dc_id: str,
        target_dc_id: str,
    ) -> Optional[Mapping[str, float]]:
        """
        RoutingObservationBuilder 未来启用 Feedback 后调用。

        返回字段必须与：

            FEEDBACK_FEATURE_NAMES

        完全一致，并且全部已经归一化到 [0,1]。

        第二十九步虽然 Provider 已经安装，
        但 USE_NEIGHBOR_HISTORICAL_FEEDBACK=False，
        所以 ObservationBuilder 当前不会调用本函数。
        """

        source_dc_id = str(source_dc_id)

        target_dc_id = str(target_dc_id)

        pair_state = self._pair_states.get(
            (
                source_dc_id,
                target_dc_id,
            )
        )

        if pair_state is None:
            return None

        feedback_age_samples = max(
            int(self._feedback_clock) - int(pair_state.last_feedback_clock),
            0,
        )

        completion_time_ewma = 0.0

        if pair_state.completion_time_ewma_s is not None:
            completion_time_ewma = self._saturating_ratio(
                value=(pair_state.completion_time_ewma_s),
                scale=(self.completion_time_scale_s),
            )

        feedback_age = self._saturating_ratio(
            value=(feedback_age_samples),
            scale=(self.age_scale_samples),
        )

        sample_confidence = self._saturating_ratio(
            value=(pair_state.sample_count),
            scale=(self.confidence_scale_samples),
        )

        feedback = {
            "success_ewma": self._clip01(pair_state.success_ewma),
            "sla_success_ewma": self._clip01(pair_state.sla_success_ewma),
            "drop_ewma": self._clip01(pair_state.drop_ewma),
            "completion_time_ewma": self._clip01(completion_time_ewma),
            "reforward_ewma": self._clip01(pair_state.reforward_ewma),
            "feedback_age": self._clip01(feedback_age),
            "sample_confidence": self._clip01(sample_confidence),
        }

        return feedback

    # ==========================================================
    # Logging / Analysis Interface
    # ==========================================================

    def get_raw_feedback(
        self,
        source_dc_id: str,
        target_dc_id: str,
    ) -> Optional[Dict[str, Any]]:

        pair_state = self._pair_states.get(
            (
                str(source_dc_id),
                str(target_dc_id),
            )
        )

        if pair_state is None:
            return None

        return {
            "source_dc_id": pair_state.source_dc_id,
            "target_dc_id": pair_state.target_dc_id,
            "sample_count": int(pair_state.sample_count),
            "completion_sample_count": int(pair_state.completion_sample_count),
            "success_ewma": float(pair_state.success_ewma),
            "sla_success_ewma": float(pair_state.sla_success_ewma),
            "drop_ewma": float(pair_state.drop_ewma),
            "completion_time_ewma_s": (
                None
                if (pair_state.completion_time_ewma_s is None)
                else float(pair_state.completion_time_ewma_s)
            ),
            "reforward_ewma": float(pair_state.reforward_ewma),
            "feedback_age_samples": int(
                max(
                    self._feedback_clock - pair_state.last_feedback_clock,
                    0,
                )
            ),
            "last_job_id": pair_state.last_job_id,
            "last_terminal_reason": pair_state.last_terminal_reason,
        }

    def snapshot_for_source(
        self,
        source_dc_id: str,
    ) -> Dict[str, Any]:
        """
        返回指定 Source DC 对所有 Neighbor 的历史认识。

        仅用于日志 / 分析。
        """

        source_dc_id = str(source_dc_id)

        snapshot: Dict[
            str,
            Any,
        ] = {}

        for target_dc_id in self.edge_dc_ids:

            if target_dc_id == source_dc_id:
                continue

            raw_feedback = self.get_raw_feedback(
                source_dc_id=(source_dc_id),
                target_dc_id=(target_dc_id),
            )

            encoded_feedback = self.get_feedback(
                source_dc_id=(source_dc_id),
                target_dc_id=(target_dc_id),
            )

            if raw_feedback is None:
                continue

            snapshot[target_dc_id] = {
                "raw": raw_feedback,
                "encoded": (None if encoded_feedback is None else dict(encoded_feedback)),
            }

        return snapshot

    def source_summary(
        self,
        source_dc_id: str,
    ) -> Dict[str, int]:

        source_dc_id = str(source_dc_id)

        states = [
            pair_state
            for (source_id, _), pair_state in self._pair_states.items()
            if source_id == source_dc_id
        ]

        return {
            "outgoing_pair_count": int(len(states)),
            "outgoing_sample_count": int(sum(pair_state.sample_count for pair_state in states)),
        }

    def summary(
        self,
    ) -> Dict[str, int]:

        return {
            "terminal_jobs_seen": int(self._total_terminal_jobs_seen),
            "total_pair_samples": int(self._total_pair_samples),
            "active_pair_count": int(len(self._pair_states)),
            "episode_terminal_jobs_seen": int(self._episode_terminal_jobs_seen),
            "episode_pair_samples": int(self._episode_pair_samples),
        }
