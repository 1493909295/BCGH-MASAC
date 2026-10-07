"""Host and routing observations with neighbor feedback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from .experience import FinalizedJobTrace


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

        base_edge_dc_map = {
            str(dc.dc_id): dc
            for dc in env.base_datacenters
            if str(dc.dc_id) in self.edge_dc_ids
        }

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

        waiting_workload = sum(
            float(host.waiting_queue.get_total_duration()) for host in hosts
        )

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

        if not host.can_ever_accommodate(job):
            return None

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

        host.calculate_load()

        available_cpu = float(host.get_available_cpu())

        available_gpu = float(host.get_available_gpu())

        waiting_jobs = len(host.waiting_queue)

        running_jobs = len(host.running_queue)

        waiting_workload = float(host.waiting_queue.get_total_duration())

        features = [
            self._normalize(
                host.cpu_num,
                self.env.max_host_cpu,
            ),
            self._normalize(
                host.gpu_capacity_num,
                self.env.max_host_gpu,
            ),
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
            self._normalize(
                available_cpu,
                self.env.max_host_cpu,
            ),
            self._normalize(
                available_gpu,
                self.env.max_host_gpu,
            ),
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

        can_ever = bool(host.can_ever_accommodate(job))

        can_start_now = bool(
            host.waiting_queue.is_empty() and host.can_accommodate(job)
        )

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

            estimated_start_delay_ratio = 1.0
            estimated_completion_ratio = 1.0

        else:

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

            estimated_completion_time = (
                elapsed_time + float(estimated_start_delay) + duration
            )

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

        if dc_id not in self.edge_dc_ids:
            raise ValueError("Host Observation 只能为 Edge DC 构造：" f"dc_id={dc_id}")

        if dc_id not in self.env.dc_map:
            raise KeyError("当前 Episode 中不存在 DC：" f"{dc_id}")

        if job_id not in self.env.job_map:
            raise KeyError("当前 Episode 中不存在 Job：" f"{job_id}")

        local_dc = self.env.dc_map[dc_id]

        job = self.env.job_map[job_id]

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

        obs.extend(self._encode_job(job))

        obs.extend(self._encode_local_dc(local_dc))

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

            raise ValueError(
                "Host Observation 出现 NaN/Inf：" f"dc={dc_id}, " f"job={job_id}"
            )

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


class RoutingObservationBuilder:

    JOB_FEAT_DIM = 5
    LOCAL_DC_BASE_FEAT_DIM = 8
    LOCAL_DC_ROUTING_FEAT_DIM = 5
    ROUTE_HISTORY_FEAT_DIM = 3
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
        self.job_feat_dim = self.JOB_FEAT_DIM
        self.local_dc_feat_dim = (
            self.LOCAL_DC_BASE_FEAT_DIM + self.LOCAL_DC_ROUTING_FEAT_DIM
        )
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

    def set_neighbor_feedback_provider(
        self,
        provider: Optional[Any],
    ) -> None:
        """
        注入 Neighbor Historical Feedback 数据提供器。

        当前阶段训练时保持 provider=None。

        未来 Provider 必须只读取历史统计数据，
        不能读取目标 DC 当前实时资源状态。
        """

        self.neighbor_feedback_provider = provider

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

    def _estimate_host_start_delay(
        self,
        host: Any,
        job: Any,
    ) -> Optional[float]:

        if not host.can_ever_accommodate(job):
            return None
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

    def build(
        self,
        agent_id: str,
    ) -> np.ndarray:

        agent_id = str(agent_id)

        job_id = str(self.env.current_job_id)
        job = self.env.job_map[job_id]
        local_dc = self.env.dc_map[agent_id]
        obs: List[float] = []

        obs.extend(self._encode_job(job))

        local_dc_features = self._encode_local_dc(
            local_dc=local_dc,
            job=job,
        )
        obs.extend(local_dc_features)

        obs.extend(
            self._encode_route_history(
                job,
            )
        )

        link_features = self._encode_links(
            agent_id=agent_id,
        )
        obs.extend(link_features)

        feedback_features = self._encode_neighbor_feedback(
            agent_id=agent_id,
        )
        obs.extend(feedback_features)

        obs_array = np.asarray(
            obs,
            dtype=np.float32,
        )

        return obs_array

    def _encode_local_dc(
        self,
        local_dc: Any,
        job: Any,
    ) -> List[float]:

        hosts = list(local_dc.host_list)
        host_count = len(hosts)

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
        waiting_workload = sum(
            float(host.waiting_queue.get_total_duration()) for host in hosts
        )

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
        immediate_feasible_host_ratio = (
            float(immediate_feasible_count) / host_count_float
        )
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

            predicted_local_completion = (
                elapsed_time + float(best_start_delay) + duration
            )

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

        features = [
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

        if len(features) != self.local_dc_feat_dim:
            raise ValueError(
                "Local DC Routing Observation 维度错误："
                f"expected={self.local_dc_feat_dim}, "
                f"actual={len(features)}"
            )

        return features

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

        routing_hop_ratio = float(hop_count / (hop_count + 1.0))

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
                "Routing History 出现 NaN/Inf："
                f"job_id={job_id}, "
                f"features={features}"
            )

        if np.any(feature_array < 0.0) or np.any(feature_array > 1.0):
            raise ValueError(
                "Routing History 超出 [0,1]："
                f"job_id={job_id}, "
                f"features={features}"
            )

        return features

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

        if agent_id not in self.edge_dc_ids:
            raise ValueError(
                "Routing Link Observation 只能为 Edge Agent 构造："
                f"agent_id={agent_id}"
            )

        if self.env.graph is None:
            raise RuntimeError(
                "当前环境 graph 尚未初始化，" "无法构造 Routing Link Observation。"
            )

        link_features: List[float] = []

        for target_dc_id in self.link_target_dc_ids:

            target_dc_id = str(target_dc_id)

            if target_dc_id == agent_id:

                latency_s = 0.0

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

            normalized_latency = self._normalize(
                value=latency_s,
                scale=float(self.env.max_latency),
            )

            link_features.append(normalized_latency)

        if len(link_features) != self.link_feat_dim:
            raise ValueError(
                "Routing Link Observation 维度错误："
                f"expected={self.link_feat_dim}, "
                f"actual={len(link_features)}"
            )

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

        当前阶段：
            USE_NEIGHBOR_HISTORICAL_FEEDBACK=False

        因此无论系统历史如何变化，都返回全 0。

        这保证：
            1. Actor 输入维度已经稳定；
            2. 当前实验不受历史反馈影响；
            3. 以后打开 Feedback 不需要修改网络结构。
        """

        agent_id = str(agent_id)

        if not self.use_neighbor_historical_feedback:
            return [0.0 for _ in range(self.feedback_feat_dim)]

        if self.neighbor_feedback_provider is None:
            raise RuntimeError(
                "USE_NEIGHBOR_HISTORICAL_FEEDBACK=True，"
                "但没有配置 Neighbor Feedback Provider。"
            )

        feedback_features: List[float] = []

        for target_dc_id in self.edge_dc_ids:

            target_dc_id = str(target_dc_id)

            if target_dc_id == agent_id:
                feedback_features.extend([0.0] * self.FEEDBACK_FEAT_PER_DC)

                continue

            feedback = self.neighbor_feedback_provider.get_feedback(
                source_dc_id=agent_id,
                target_dc_id=target_dc_id,
            )

            if feedback is None:
                feedback_features.extend([0.0] * self.FEEDBACK_FEAT_PER_DC)

                continue

            if not isinstance(
                feedback,
                Mapping,
            ):
                raise TypeError(
                    "Neighbor Feedback Provider 必须返回 "
                    "Mapping[str, float] 或 None。"
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

        if len(feedback_features) != self.feedback_feat_dim:
            raise ValueError(
                "Neighbor Feedback Observation 维度错误："
                f"expected={self.feedback_feat_dim}, "
                f"actual={len(feedback_features)}"
            )

        return feedback_features


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

        self.edge_dc_ids = [str(dc_id) for dc_id in env.edge_dc_ids]

        self.state_dc_ids = list(self.edge_dc_ids) + [str(env.cloud_id)]

        self.job_feat_dim = int(routing_observation_builder.job_feat_dim)

        self.dc_feat_dim = int(routing_observation_builder.local_dc_feat_dim)

        self.route_history_feat_dim = int(
            routing_observation_builder.route_history_feat_dim
        )

        self.topology_source_dc_ids = list(self.edge_dc_ids)

        self.topology_target_dc_ids = list(self.state_dc_ids)

        self.topology_feat_dim = len(self.topology_source_dc_ids) * len(
            self.topology_target_dc_ids
        )

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
            raise RuntimeError(
                "环境 graph 尚未初始化，" "无法构造 Routing Centralized State。"
            )

        features: List[float] = []

        for source_dc_id in self.topology_source_dc_ids:

            for target_dc_id in self.topology_target_dc_ids:

                source_dc_id = str(source_dc_id)

                target_dc_id = str(target_dc_id)

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
            raise RuntimeError(
                "当前不存在 Routing Job，" "无法构造 Centralized State。"
            )

        job_id = str(self.env.current_job_id)

        if job_id not in self.env.job_map:
            raise KeyError(f"当前 Job 不存在：{job_id}")

        job = self.env.job_map[job_id]

        state: List[float] = []

        job_features = self.routing_observation_builder.encode_job_features(job)

        state.extend(job_features)

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

        route_history = self.routing_observation_builder.encode_route_history_features(
            job
        )

        state.extend(route_history)

        state.extend(self._encode_topology())

        state_array = np.asarray(
            state,
            dtype=np.float32,
        )

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
            raise ValueError(
                "Routing Centralized State " "出现 NaN/Inf：" f"job_id={job_id}"
            )

        return state_array


class ShortWindowFeedbackStore:
    """Same seven observation fields, computed only from current-window outcomes.

    Legacy *_ewma names are kept for observation/checkpoint dimensional compatibility;
    their values here are window means. No EWMA or cross-episode history is retained.
    """

    def __init__(self, env, *, window_s, confidence_scale_samples=20.0):
        self.env = env
        self.edge_dc_ids = tuple(map(str, env.edge_dc_ids))
        from .training_support import BayesianBeliefStore

        self.store = BayesianBeliefStore(
            self.edge_dc_ids,
            window_s=window_s,
            confidence_scale=confidence_scale_samples,
        )
        self.completion_time_scale_s = max(
            float(env.max_job_duration) * float(env.drop_deadline_ratio), 1e-8
        )
        self.reset_episode_counters()

    def reset_episode_counters(self):
        self.store.reset()
        self.terminal_jobs_seen = 0
        self.pair_samples = 0

    def update_from_finalized_trace(self, trace):
        self.store.advance_time(float(self.env.current_time))
        self.terminal_jobs_seen += 1
        from .training_support import build_bayesian_evidence_from_finalized_trace

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
        completed = [
            e.completion_time_s for e in records if e.completion_time_s is not None
        ]
        delay = (sum(completed) / len(completed)) if completed else None
        return dict(
            success_ewma=sum(e.success for e in records) / count,
            sla_success_ewma=sum(e.sla_satisfied for e in records) / count,
            drop_ewma=sum(not e.success for e in records) / count,
            completion_time_ewma=(
                delay / (delay + self.completion_time_scale_s)
                if delay is not None
                else 0.5
            ),
            reforward_ewma=sum(e.reforwarded for e in records) / count,
            feedback_age=min(
                1.0,
                (self.store.now - max(e.timestamp for e in records))
                / self.store.window_s,
            ),
            sample_confidence=self.store.get_confidence(source_dc_id, target_dc_id),
        )

    def get_raw_feedback(self, source_dc_id, target_dc_id):
        encoded = self.get_feedback(source_dc_id, target_dc_id)
        if encoded is None:
            return None
        return {
            **encoded,
            "sample_count": len(self.store.records(source_dc_id, target_dc_id)),
            "aggregation": "short_time_window",
        }

    def snapshot_for_source(self, source_dc_id):
        result = {}
        for target in self.edge_dc_ids:
            feedback = self.get_feedback(source_dc_id, target)
            if feedback is not None:
                result[target] = {
                    "raw": self.get_raw_feedback(source_dc_id, target),
                    "encoded": feedback,
                }
        return result

    def source_summary(self, source_dc_id):
        self.store.advance_time(float(self.env.current_time))
        counts = [
            len(self.store.records(source_dc_id, j))
            for j in self.edge_dc_ids
            if j != source_dc_id
        ]
        return dict(
            outgoing_pair_count=sum(n > 0 for n in counts),
            outgoing_sample_count=sum(counts),
        )

    def summary(self):
        counts = [self.source_summary(i) for i in self.edge_dc_ids]
        return dict(
            terminal_jobs_seen=self.terminal_jobs_seen,
            total_pair_samples=sum(c["outgoing_sample_count"] for c in counts),
            active_pair_count=sum(c["outgoing_pair_count"] for c in counts),
            episode_terminal_jobs_seen=self.terminal_jobs_seen,
            episode_pair_samples=self.pair_samples,
        )


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

    sample_count: int = 0

    completion_sample_count: int = 0

    success_ewma: float = 0.0
    sla_success_ewma: float = 0.0
    drop_ewma: float = 0.0

    completion_time_ewma_s: Optional[float] = None

    reforward_ewma: float = 0.0

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

        self.completion_time_scale_s = max(
            float(env.max_job_duration) * float(env.drop_deadline_ratio),
            1e-8,
        )

        self._pair_states: Dict[
            tuple[str, str],
            NeighborPairFeedbackState,
        ] = {}

        self._feedback_clock: int = 0

        self._total_terminal_jobs_seen: int = 0
        self._total_pair_samples: int = 0

        self._episode_terminal_jobs_seen: int = 0
        self._episode_pair_samples: int = 0

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

        if previous_sample_count <= 0:
            return sample_value

        return float(
            (1.0 - self.ewma_alpha) * float(previous_value)
            + self.ewma_alpha * sample_value
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

        if completion_time_s is not None:

            completion_time_s = max(
                float(completion_time_s),
                0.0,
            )

            if (
                pair_state.completion_sample_count <= 0
                or pair_state.completion_time_ewma_s is None
            ):
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

            sla_success_sample = (
                1.0 if (completion_time_s <= sla_limit_s + 1e-9) else 0.0
            )

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
                "encoded": (
                    None if encoded_feedback is None else dict(encoded_feedback)
                ),
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
            "outgoing_sample_count": int(
                sum(pair_state.sample_count for pair_state in states)
            ),
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
