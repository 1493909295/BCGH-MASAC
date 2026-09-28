from __future__ import annotations

from typing import List,Mapping
import numpy as np
from typing import Any, Optional

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

    def __init__(self, env: Any, use_neighbor_historical_feedback: bool = False,neighbor_feedback_provider: Optional[Any] = None,) -> None:

        self.env = env
        self.edge_dc_ids = [
            str(dc_id)
            for dc_id in env.edge_dc_ids
        ]
        self.link_target_dc_ids = (list(self.edge_dc_ids) + [str(env.cloud_id)])
        # self.job_edge_hop_counts: Dict[str, int] = {}
        # self.job_edge_transfer_latency_s: Dict[str, float,] = {}
        self.job_feat_dim = self.JOB_FEAT_DIM
        self.local_dc_feat_dim = (self.LOCAL_DC_BASE_FEAT_DIM + self.LOCAL_DC_ROUTING_FEAT_DIM)
        self.route_history_feat_dim = (self.ROUTE_HISTORY_FEAT_DIM)
        self.link_feat_dim = len(self.link_target_dc_ids)
        self.use_neighbor_historical_feedback = bool(use_neighbor_historical_feedback)
        self.neighbor_feedback_provider = (neighbor_feedback_provider)
        self.feedback_feat_dim = (len(self.edge_dc_ids) * self.FEEDBACK_FEAT_PER_DC)
        self.obs_dim = (
            self.job_feat_dim
            + self.local_dc_feat_dim
            + self.route_history_feat_dim
            + self.link_feat_dim
            + self.feedback_feat_dim
        )

    # 注入 Neighbor Historical Feedback 数据提供器
    def set_neighbor_feedback_provider(self, provider: Optional[Any],) -> None:
        """
        注入 Neighbor Historical Feedback 数据提供器。

        当前阶段训练时保持 provider=None。

        未来 Provider 必须只读取历史统计数据，
        不能读取目标 DC 当前实时资源状态。
        """

        self.neighbor_feedback_provider = provider



    # 归一化辅助函数
    def _normalize(self, value: float,  scale: float,) -> float:

        eps = max(float(getattr(self.env, "norm_eps", 1e-8,)), 1e-12,)

        value = max(float(value), 0.0)
        scale = max(float(scale), eps)

        return float(np.clip(value / scale, 0.0, 1.0,))
    # 无上界归一化函数
    def _saturating_ratio(self, value: float, scale: float,) -> float:

        eps = max(float(getattr(self.env, "norm_eps", 1e-8,)), 1e-12,)

        value = max(float(value), 0.0)
        scale = max(float(scale), eps)

        return float(value / (value + scale))

    # 如果选择此host要等多久
    def _estimate_host_start_delay(self, host: Any, job: Any,) -> Optional[float]:

        # 资源不满足
        if not host.can_ever_accommodate(job):
            return None
        # 机器为空
        if (
                host.waiting_queue.is_empty()
                and host.can_accommodate(job)
        ):
            return 0.0


        max_running_remaining = 0.0
        running_jobs = getattr(host.running_queue, "_queue", [],)

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

            max_running_remaining = max(max_running_remaining, remaining_time,)

        waiting_workload = float(host.waiting_queue.get_total_duration())

        return float(max_running_remaining + waiting_workload)

    def encode_job_features(self, job: Any,) -> List[float]:
        return self._encode_job(job)

    def encode_dc_aggregate_features(self, dc: Any, job: Any,) -> List[float]:
        return self._encode_local_dc(local_dc=dc, job=job,)

    def encode_route_history_features(self, job: Any,) -> List[float]:
        return self._encode_route_history(job)


    #
    def build(self, agent_id: str,) -> np.ndarray:

        agent_id = str(agent_id)

        job_id = str(self.env.current_job_id)
        job = self.env.job_map[job_id]
        local_dc = self.env.dc_map[agent_id]
        obs: List[float] = []

        # 当前 Job
        obs.extend(self._encode_job(job))

        # 当前 DC 聚合状态
        local_dc_features = self._encode_local_dc(local_dc=local_dc, job=job,)
        obs.extend(local_dc_features)

        # 当前 Job 多跳 Routing 历史
        obs.extend(self._encode_route_history( job,))

        # 当前 DC 到其他 DC / Cloud 的链路
        link_features = self._encode_links(agent_id=agent_id,)
        obs.extend(link_features)

        # Neighbor Historical Feedback
        feedback_features = (self._encode_neighbor_feedback(agent_id=agent_id,))
        obs.extend(feedback_features)

        obs_array = np.asarray(obs, dtype=np.float32,)

        return obs_array

    #
    def _encode_local_dc(self, local_dc: Any, job: Any,) -> List[float]:

        hosts = list(local_dc.host_list)
        host_count = len(hosts)

        # 更新所有 Host / DC 的当前负载。
        local_dc.calculate_dc_loads()

        total_cpu = sum(
            float(host.cpu_num)
            for host in hosts
        )
        total_gpu = sum(
            float(host.gpu_capacity_num)
            for host in hosts
        )
        used_cpu = sum(
            float(host.used_cpu)
            for host in hosts
        )
        used_gpu = sum(
            float(host.used_gpu)
            for host in hosts
        )

        available_cpu = max(total_cpu - used_cpu, 0.0,)
        available_gpu = max(total_gpu - used_gpu, 0.0,)

        waiting_jobs = sum(
            len(host.waiting_queue)
            for host in hosts
        )
        running_jobs = sum(
            len(host.running_queue)
            for host in hosts
        )
        waiting_workload = sum(
            float(host.waiting_queue.get_total_duration())
            for host in hosts
        )

        host_scale = max(host_count, 1,)
        dc_queue_length_scale = (float(self.env.queue_length_scale) * host_scale)
        dc_queue_workload_scale = (float(self.env.queue_workload_scale) * host_scale)
        immediate_feasible_count = 0
        ever_feasible_count = 0

        best_start_delay = None

        for host in hosts:
            # 物理总容量是否能够运行当前 Job。
            if host.can_ever_accommodate(job):
                ever_feasible_count += 1
                estimated_delay = (self._estimate_host_start_delay(host=host, job=job,))

                if estimated_delay is not None:
                    if (
                            best_start_delay is None
                            or estimated_delay < best_start_delay
                    ):
                        best_start_delay = (estimated_delay)

            if (
                    host.waiting_queue.is_empty()
                    and host.can_accommodate(job)
            ):
                immediate_feasible_count += 1

        host_count_float = float(max(host_count, 1))
        immediate_feasible_host_ratio = (float(immediate_feasible_count) / host_count_float)
        ever_feasible_host_ratio = (float(ever_feasible_count) / host_count_float)

        duration = max(float(job.duration),float(getattr(self.env, "norm_eps", 1e-8,)),)

        elapsed_time = max(float(self.env.current_time) - float(job.arrive_time), 0.0,)

        # 如果没有任何 Host 的物理容量能够执行 Job，
        # 直接把两个 local-risk 指标置为最高风险。
        if best_start_delay is None:
            best_local_start_delay_ratio = 1.0
            best_local_completion_ratio = 1.0

        else:
            local_wait_budget = max(
                (float(self.env.drop_deadline_ratio) - 1.0) * duration,float(getattr(self.env,"norm_eps", 1e-8,)),)
            best_local_start_delay_ratio = float(
                np.clip(
                    best_start_delay
                    / local_wait_budget,
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
            predicted_local_completion = (
                    elapsed_time
                    + float(best_start_delay)
                    + duration
            )

            drop_completion_limit = max(
                float(
                    self.env.drop_deadline_ratio
                )
                * duration,
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
                    predicted_local_completion
                    / drop_completion_limit,
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
    def _encode_route_history(self, job: Any,) -> List[float]:
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
        routing_hop_ratio = float(
            hop_count
            / (
                    hop_count
                    + 1.0
            )
        )

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

        cumulative_latency_ratio = (
            self._saturating_ratio(
                value=cumulative_latency_s,
                scale=latency_scale_s,
            )
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

        cumulative_energy_ratio = (
            self._saturating_ratio(
                value=cumulative_energy_j,
                scale=energy_scale_j,
            )
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

        if not np.all(
                np.isfinite(feature_array)
        ):
            raise ValueError(
                "Routing History 出现 NaN/Inf："
                f"job_id={job_id}, "
                f"features={features}"
            )

        if (
                np.any(feature_array < 0.0)
                or np.any(feature_array > 1.0)
        ):
            raise ValueError(
                "Routing History 超出 [0,1]："
                f"job_id={job_id}, "
                f"features={features}"
            )

        return features

    #
    def _encode_links(self, agent_id: str,) -> List[float]:
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
                "Routing Link Observation 只能为 Edge Agent 构造："
                f"agent_id={agent_id}"
            )

        if self.env.graph is None:
            raise RuntimeError(
                "当前环境 graph 尚未初始化，"
                "无法构造 Routing Link Observation。"
            )

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
                        self.env.graph[
                            agent_id
                        ][
                            target_dc_id
                        ].get(
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
                scale=float(
                    self.env.max_latency
                ),
            )

            link_features.append(
                normalized_latency
            )

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

        if not np.all(
                np.isfinite(link_array)
        ):
            raise ValueError(
                "Routing Link Observation 出现 NaN/Inf："
                f"agent_id={agent_id}, "
                f"features={link_features}"
            )

        if (
                np.any(link_array < 0.0)
                or np.any(link_array > 1.0)
        ):
            raise ValueError(
                "Routing Link Observation 超出 [0,1]："
                f"agent_id={agent_id}, "
                f"features={link_features}"
            )

        return link_features

    #
    def _encode_job(self, job: Any,) -> List[float]:
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
    def _encode_neighbor_feedback(self, agent_id: str,) -> List[float]:
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

        # ==========================================================
        # Feedback disabled
        #
        # 当前正式实验走这里：
        #
        #     F_i = 0
        #
        # 不查询 Provider；
        # 不访问 Remote DC；
        # 不读取任何历史结果。
        # ==========================================================
        if not self.use_neighbor_historical_feedback:
            return [
                0.0
                for _ in range(
                    self.feedback_feat_dim
                )
            ]

        # ==========================================================
        # Feedback enabled
        #
        # 未来真正启用时才进入。
        #
        # 如果用户打开开关却没有安装 Feedback Provider，
        # 必须直接报错，而不是偷偷继续返回全 0。
        # ==========================================================
        if self.neighbor_feedback_provider is None:
            raise RuntimeError(
                "USE_NEIGHBOR_HISTORICAL_FEEDBACK=True，"
                "但没有配置 Neighbor Feedback Provider。"
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
                feedback_features.extend(
                    [0.0]
                    * self.FEEDBACK_FEAT_PER_DC
                )

                continue

            # ------------------------------------------------------
            # 未来历史数据接口。
            #
            # 这里只允许 Provider 返回历史统计结果。
            # ------------------------------------------------------
            feedback = (
                self.neighbor_feedback_provider
                    .get_feedback(
                    source_dc_id=agent_id,
                    target_dc_id=target_dc_id,
                )
            )

            # 尚未收集到任何历史样本：
            #
            # 7维全部置 0，
            # sample_confidence=0 同时能够表达“没有证据”。
            if feedback is None:
                feedback_features.extend(
                    [0.0]
                    * self.FEEDBACK_FEAT_PER_DC
                )

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
                for feature_name
                in self.FEEDBACK_FEATURE_NAMES
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

            if not np.all(
                    np.isfinite(block_array)
            ):
                raise ValueError(
                    "Neighbor Feedback 出现 NaN/Inf："
                    f"source={agent_id}, "
                    f"target={target_dc_id}, "
                    f"block={block}"
                )

            if (
                    np.any(block_array < 0.0)
                    or np.any(block_array > 1.0)
            ):
                raise ValueError(
                    "Neighbor Feedback 必须已经归一化到 [0,1]："
                    f"source={agent_id}, "
                    f"target={target_dc_id}, "
                    f"block={block}"
                )

            feedback_features.extend(
                block
            )

        # ==========================================================
        # 最终维度检查
        # ==========================================================
        if (
                len(feedback_features)
                != self.feedback_feat_dim
        ):
            raise ValueError(
                "Neighbor Feedback Observation 维度错误："
                f"expected={self.feedback_feat_dim}, "
                f"actual={len(feedback_features)}"
            )

        return feedback_features





