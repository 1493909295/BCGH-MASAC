from __future__ import annotations

from typing import Any, List

import numpy as np

from routing_observation import (
    RoutingObservationBuilder,
)


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

    def __init__(self,env: Any,routing_observation_builder:RoutingObservationBuilder,) -> None:

        self.env = env

        self.routing_observation_builder = (
            routing_observation_builder
        )


        # ======================================================
        # Edge Routing Agents
        # ======================================================

        self.edge_dc_ids = [
            str(dc_id)
            for dc_id in env.edge_dc_ids
        ]


        # ======================================================
        # 所有实际计算域。
        #
        # 即使 Cloud Action 关闭，
        # Cloud 物理环境仍存在，因此保持 State shape 不变。
        # ======================================================

        self.state_dc_ids = (
            list(self.edge_dc_ids)
            + [str(env.cloud_id)]
        )


        # ======================================================
        # Feature dimensions
        # ======================================================

        self.job_feat_dim = int(
            routing_observation_builder
            .job_feat_dim
        )

        self.dc_feat_dim = int(
            routing_observation_builder
            .local_dc_feat_dim
        )

        self.route_history_feat_dim = int(
            routing_observation_builder
            .route_history_feat_dim
        )


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
        self.topology_source_dc_ids = (
            list(self.edge_dc_ids)
        )

        self.topology_target_dc_ids = (
            list(self.state_dc_ids)
        )

        self.topology_feat_dim = (
            len(self.topology_source_dc_ids)
            * len(self.topology_target_dc_ids)
        )


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
            + len(self.state_dc_ids)
            * self.dc_feat_dim
            + self.route_history_feat_dim
            + self.topology_feat_dim
        )

    def _encode_topology(self,) -> List[float]:
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
                "环境 graph 尚未初始化，"
                "无法构造 Routing Centralized State。"
            )

        features: List[float] = []


        for source_dc_id in (
            self.topology_source_dc_ids
        ):

            for target_dc_id in (
                self.topology_target_dc_ids
            ):

                source_dc_id = str(
                    source_dc_id
                )

                target_dc_id = str(
                    target_dc_id
                )


                # Self Routing 网络时延为 0。
                if (
                    source_dc_id
                    == target_dc_id
                ):

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
                            self.env.graph[
                                source_dc_id
                            ][
                                target_dc_id
                            ].get(
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
                            float(
                                self.env.max_latency
                            ),
                            1e-8,
                        ),
                        0.0,
                        1.0,
                    )
                )

                features.append(
                    normalized_latency
                )


        if (
            len(features)
            != self.topology_feat_dim
        ):

            raise ValueError(
                "Routing topology 维度错误："
                f"expected="
                f"{self.topology_feat_dim}, "
                f"actual="
                f"{len(features)}"
            )


        return features

    def build(self,) -> np.ndarray:
        """
        构造当前 Routing Decision 对应的
        Centralized Training State。
        """

        if self.env.current_job_id is None:
            raise RuntimeError(
                "当前不存在 Routing Job，"
                "无法构造 Centralized State。"
            )


        job_id = str(
            self.env.current_job_id
        )

        if job_id not in self.env.job_map:
            raise KeyError(
                f"当前 Job 不存在：{job_id}"
            )

        job = self.env.job_map[
            job_id
        ]


        state: List[float] = []


        # ======================================================
        # 1. Current Job
        # ======================================================

        job_features = (
            self.routing_observation_builder
            .encode_job_features(job)
        )

        state.extend(
            job_features
        )


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

            dc = self.env.dc_map[
                str(dc_id)
            ]

            dc_features = (
                self.routing_observation_builder
                .encode_dc_aggregate_features(
                    dc=dc,
                    job=job,
                )
            )


            if (
                len(dc_features)
                != self.dc_feat_dim
            ):
                raise ValueError(
                    "Centralized DC feature "
                    "维度错误："
                    f"dc={dc_id}, "
                    f"expected="
                    f"{self.dc_feat_dim}, "
                    f"actual="
                    f"{len(dc_features)}"
                )


            state.extend(
                dc_features
            )


        # ======================================================
        # 3. Current Job Routing History
        # ======================================================

        route_history = (
            self.routing_observation_builder
            .encode_route_history_features(
                job
            )
        )

        state.extend(
            route_history
        )


        # ======================================================
        # 4. Global Routing Topology
        # ======================================================

        state.extend(
            self._encode_topology()
        )


        state_array = np.asarray(
            state,
            dtype=np.float32,
        )


        # ======================================================
        # 5. Final validation
        # ======================================================

        if state_array.shape != (
            self.state_dim,
        ):
            raise ValueError(
                "Routing Centralized State "
                "维度错误："
                f"expected="
                f"{(self.state_dim,)}, "
                f"actual="
                f"{state_array.shape}"
            )


        if not np.all(
            np.isfinite(state_array)
        ):
            raise ValueError(
                "Routing Centralized State "
                "出现 NaN/Inf："
                f"job_id={job_id}"
            )


        return state_array



