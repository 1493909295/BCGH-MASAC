from __future__ import annotations

from typing import  Dict, List
from typing import Any, Optional
import numpy as np


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

    HOST_BASE_FEAT_DIM = len(
        HOST_BASE_FEATURE_NAMES
    )

    HOST_JOB_EVAL_FEATURE_NAMES = (
        "can_ever_accommodate",
        "can_start_now",
        "estimated_start_delay_ratio",
        "estimated_completion_ratio",
    )

    HOST_JOB_EVAL_FEAT_DIM = len(
        HOST_JOB_EVAL_FEATURE_NAMES
    )

    HOST_FEATURE_NAMES = (
            HOST_BASE_FEATURE_NAMES
            + HOST_JOB_EVAL_FEATURE_NAMES
    )

    HOST_FEAT_DIM = (
        HOST_BASE_FEAT_DIM
        + HOST_JOB_EVAL_FEAT_DIM
    )


    def __init__(
        self,
        env: Any,
    ) -> None:

        self.env = env

        self.edge_dc_ids = [
            str(dc_id)
            for dc_id in env.edge_dc_ids
        ]


        # ======================================================
        # Host 层只管理 Edge DC。
        #
        # Cloud 不进入 Local Host SAC；
        # Cloud 仍沿用环境中的自动执行逻辑。
        # ======================================================

        base_edge_dc_map = {
            str(dc.dc_id): dc
            for dc in env.base_datacenters
            if str(dc.dc_id) in self.edge_dc_ids
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
                raise KeyError(
                    "Host Observation 找不到 Edge DC："
                    f"{dc_id}"
                )

            dc = base_edge_dc_map[dc_id]

            host_ids = [
                str(host.host_id)
                for host in dc.host_list
            ]

            if len(host_ids) == 0:
                raise ValueError(
                    f"Edge DC {dc_id} 没有 Host。"
                )

            self.host_ids_by_dc[
                dc_id
            ] = host_ids

            self.host_count_by_dc[
                dc_id
            ] = len(host_ids)


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
                + self.HOST_FEAT_DIM
                * self.host_count_by_dc[dc_id]
            )
            for dc_id in self.edge_dc_ids
        }


        self.action_dim_by_dc: Dict[
            str,
            int,
        ] = {
            dc_id:
                self.host_count_by_dc[dc_id]
            for dc_id in self.edge_dc_ids
        }


    def get_obs_dim(
        self,
        dc_id: str,
    ) -> int:

        dc_id = str(dc_id)

        if dc_id not in self.obs_dim_by_dc:
            raise KeyError(
                f"未知 Edge DC：{dc_id}"
            )

        return int(
            self.obs_dim_by_dc[dc_id]
        )


    def get_action_dim(
        self,
        dc_id: str,
    ) -> int:

        dc_id = str(dc_id)

        if dc_id not in self.action_dim_by_dc:
            raise KeyError(
                f"未知 Edge DC：{dc_id}"
            )

        return int(
            self.action_dim_by_dc[dc_id]
        )


    def get_host_ids(
        self,
        dc_id: str,
    ) -> List[str]:

        dc_id = str(dc_id)

        if dc_id not in self.host_ids_by_dc:
            raise KeyError(
                f"未知 Edge DC：{dc_id}"
            )

        return list(
            self.host_ids_by_dc[dc_id]
        )

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

        return float(
            value / (
                value + scale
            )
        )

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

        if feature_array.shape != (
                int(expected_dim),
        ):
            raise ValueError(
                f"{block_name} 维度错误："
                f"expected={(int(expected_dim),)}, "
                f"actual={feature_array.shape}"
            )

        if not np.all(
                np.isfinite(feature_array)
        ):
            raise ValueError(
                f"{block_name} 出现 NaN/Inf："
                f"{features}"
            )

        eps = 1e-6

        if (
                np.any(feature_array < -eps)
                or np.any(
            feature_array > 1.0 + eps
        )
        ):
            raise ValueError(
                f"{block_name} 必须位于 [0,1]："
                f"{features}"
            )

        return [
            float(value)
            for value in features
        ]

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
            float(self.env.current_time)
            - float(job.arrive_time),
            0.0,
        )


        sla_budget = max(
            float(
                self.env.sla_deadline_ratio
            )
            * duration,
            eps,
        )


        drop_budget = max(
            float(
                self.env.drop_deadline_ratio
            )
            * duration,
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
                    elapsed_time
                    / sla_budget,
                    0.0,
                    1.0,
                )
            ),

            float(
                np.clip(
                    elapsed_time
                    / drop_budget,
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

        hosts = list(
            local_dc.host_list
        )

        host_count = max(
            len(hosts),
            1,
        )


        local_dc.calculate_dc_loads()


        waiting_jobs = sum(
            len(host.waiting_queue)
            for host in hosts
        )

        running_jobs = sum(
            len(host.running_queue)
            for host in hosts
        )

        waiting_workload = sum(
            float(
                host.waiting_queue
                .get_total_duration()
            )
            for host in hosts
        )


        queue_length_scale = (
            float(
                self.env.queue_length_scale
            )
            * host_count
        )

        queue_workload_scale = (
            float(
                self.env.queue_workload_scale
            )
            * host_count
        )


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


        if (
            len(features)
            != self.LOCAL_DC_FEAT_DIM
        ):
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
        if not host.can_ever_accommodate(
            job
        ):
            return None


        # Waiting Queue 为空且当前资源足够，
        # 根据现有环境语义可以立即执行。
        if (
            host.waiting_queue.is_empty()
            and host.can_accommodate(job)
        ):
            return 0.0


        max_running_remaining = 0.0


        for running_job in list(
            getattr(
                host.running_queue,
                "_queue",
                [],
            )
        ):

            if (
                running_job.start_time
                is None
            ):
                remaining_time = float(
                    running_job.duration
                )

            else:
                remaining_time = max(
                    float(
                        running_job.start_time
                    )
                    + float(
                        running_job.duration
                    )
                    - float(
                        self.env.current_time
                    ),
                    0.0,
                )

            max_running_remaining = max(
                max_running_remaining,
                remaining_time,
            )


        waiting_workload = float(
            host.waiting_queue
            .get_total_duration()
        )


        return float(
            max_running_remaining
            + waiting_workload
        )

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

        available_cpu = float(
            host.get_available_cpu()
        )

        available_gpu = float(
            host.get_available_gpu()
        )

        waiting_jobs = len(
            host.waiting_queue
        )

        running_jobs = len(
            host.running_queue
        )

        waiting_workload = float(
            host.waiting_queue
                .get_total_duration()
        )

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

        can_ever = bool(
            host.can_ever_accommodate(
                job
            )
        )

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

        can_start_now = bool(
            host.waiting_queue.is_empty()
            and host.can_accommodate(job)
        )

        # ==========================================================
        # 3. Estimated start delay
        # ==========================================================

        estimated_start_delay = (
            self._estimate_host_start_delay(
                host=host,
                job=job,
            )
        )

        duration = max(
            float(job.duration),
            eps,
        )

        elapsed_time = max(
            float(self.env.current_time)
            - float(job.arrive_time),
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
                (
                        float(
                            self.env.drop_deadline_ratio
                        )
                        - 1.0
                )
                * duration,
                eps,
            )

            estimated_start_delay_ratio = float(
                np.clip(
                    float(
                        estimated_start_delay
                    )
                    / wait_budget,
                    0.0,
                    1.0,
                )
            )

            # ======================================================
            # 从最初进入系统到预计完成的总服务时间。
            # ======================================================

            estimated_completion_time = (
                    elapsed_time
                    + float(
                estimated_start_delay
            )
                    + duration
            )

            drop_completion_limit = max(
                float(
                    self.env.drop_deadline_ratio
                )
                * duration,
                eps,
            )

            estimated_completion_ratio = float(
                np.clip(
                    estimated_completion_time
                    / drop_completion_limit,
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
            expected_dim=(
                self.HOST_JOB_EVAL_FEAT_DIM
            ),
            block_name=(
                "Host Current-Job Evaluation"
            ),
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

        base_features = (
            self._encode_host_base_features(
                host=host,
            )
        )

        job_eval_features = (
            self._encode_host_job_eval_features(
                host=host,
                job=job,
            )
        )

        features = (
                base_features
                + job_eval_features
        )

        return self._validate_host_feature_block(
            features=features,
            expected_dim=self.HOST_FEAT_DIM,
            block_name=(
                f"Host Feature "
                f"(host_id={host.host_id})"
            ),
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
            raise ValueError(
                "Host Observation 只能为 Edge DC 构造："
                f"dc_id={dc_id}"
            )


        if dc_id not in self.env.dc_map:
            raise KeyError(
                "当前 Episode 中不存在 DC："
                f"{dc_id}"
            )


        if job_id not in self.env.job_map:
            raise KeyError(
                "当前 Episode 中不存在 Job："
                f"{job_id}"
            )


        local_dc = self.env.dc_map[
            dc_id
        ]

        job = self.env.job_map[
            job_id
        ]


        # ======================================================
        # Host 数量和顺序属于网络结构的一部分。
        #
        # 每个 Episode 的硬件环境必须保持一致。
        # ======================================================

        runtime_host_ids = [
            str(host.host_id)
            for host in local_dc.host_list
        ]


        expected_host_ids = (
            self.host_ids_by_dc[
                dc_id
            ]
        )


        if (
            runtime_host_ids
            != expected_host_ids
        ):
            raise RuntimeError(
                "Host 列表发生变化，"
                "会破坏 Local Host SAC 的输入/动作映射："
                f"dc={dc_id}, "
                f"expected={expected_host_ids}, "
                f"actual={runtime_host_ids}"
            )


        obs: List[float] = []


        # 1. Current Job
        obs.extend(
            self._encode_job(job)
        )


        # 2. Local DC context
        obs.extend(
            self._encode_local_dc(
                local_dc
            )
        )


        # 3. Every real local Host
        #
        # 不 padding。
        # 不读取其他 DC。
        for host_index, host in enumerate(
                local_dc.host_list
        ):

            host_features = self._encode_host(
                host=host,
                job=job,
            )

            if (
                    len(host_features)
                    != self.HOST_FEAT_DIM
            ):
                raise ValueError(
                    "Host block dimension error："
                    f"dc={dc_id}, "
                    f"host_index={host_index}, "
                    f"host_id={host.host_id}, "
                    f"expected={self.HOST_FEAT_DIM}, "
                    f"actual={len(host_features)}"
                )

            obs.extend(
                host_features
            )


        obs_array = np.asarray(
            obs,
            dtype=np.float32,
        )


        expected_dim = (
            self.get_obs_dim(
                dc_id
            )
        )


        if obs_array.shape != (
            expected_dim,
        ):

            raise ValueError(
                "Host Observation dimension error："
                f"dc={dc_id}, "
                f"expected={(expected_dim,)}, "
                f"actual={obs_array.shape}"
            )


        if not np.all(
            np.isfinite(obs_array)
        ):

            raise ValueError(
                "Host Observation 出现 NaN/Inf："
                f"dc={dc_id}, "
                f"job={job_id}"
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
            raise RuntimeError(
                "当前不存在 pending Host Job。"
            )


        if dc_id is None:
            raise RuntimeError(
                "当前不存在 pending Host DC。"
            )


        dc_id = str(dc_id)
        job_id = str(job_id)


        # Host Decision 发生时 Routing AEC 应处于暂停状态。
        if (
            self.env.agent_selection
            is not None
        ):
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

