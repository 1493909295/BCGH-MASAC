"""Routing Actor inputs with instantaneous load information from every DC.

Only the current job, live DC aggregates and observed history are read. The
pre-generated future workload and future arrival events are not inspected.
Cloud remains in the observation even when its routing action is disabled.
"""
from __future__ import annotations

import numpy as np

from ._base import observations


class RoutingObservationBuilder(observations.RoutingObservationBuilder):
    SCHEMA_VERSION = 1
    DC_FEATURE_NAMES = (
        "cpu_capacity", "gpu_capacity", "cpu_load", "gpu_load",
        "available_cpu", "available_gpu", "waiting_jobs", "running_jobs",
        "waiting_workload", "immediate_feasible_host_ratio",
        "ever_feasible_host_ratio", "best_start_delay_ratio",
        "best_completion_ratio",
    )

    def __init__(self, env, use_neighbor_historical_feedback=False,
                 neighbor_feedback_provider=None):
        super().__init__(env, use_neighbor_historical_feedback, neighbor_feedback_provider)
        self.state_dc_ids = list(self.edge_dc_ids) + [str(env.cloud_id)]
        if len(set(self.state_dc_ids)) != len(self.state_dc_ids):
            raise ValueError("OPT DC identifiers must be unique")
        if len(self.DC_FEATURE_NAMES) != self.local_dc_feat_dim:
            raise ValueError("OPT DC feature schema must match the shared encoder")
        self.global_dc_feat_dim = len(self.state_dc_ids) * self.local_dc_feat_dim
        self.obs_dim = (self.job_feat_dim + self.global_dc_feat_dim
                        + self.route_history_feat_dim + self.link_feat_dim + self.feedback_feat_dim)
        self.dc_slices = {
            dc_id: slice(self.job_feat_dim + index * self.local_dc_feat_dim,
                         self.job_feat_dim + (index + 1) * self.local_dc_feat_dim)
            for index, dc_id in enumerate(self.state_dc_ids)
        }

    def build(self, agent_id: str) -> np.ndarray:
        agent_id = str(agent_id)
        if agent_id not in self.edge_dc_ids:
            raise ValueError(f"Unknown OPT routing agent: {agent_id}")
        if self.env.current_job_id is None:
            raise RuntimeError("OPT observation requires a current routing decision")
        if list(map(str, self.env.edge_dc_ids)) != self.edge_dc_ids:
            raise RuntimeError("OPT DC order changed after initialization")
        job = self.env.job_map[str(self.env.current_job_id)]
        features = list(self.encode_job_features(job))
        # Fixed DC order is shared by every agent; agent identity stays in the network.
        for dc_id in self.state_dc_ids:
            features.extend(self.encode_dc_aggregate_features(self.env.dc_map[dc_id], job))
        features.extend(self.encode_route_history_features(job))
        features.extend(self._encode_links(agent_id))
        features.extend(self._encode_neighbor_feedback(agent_id))
        result = np.asarray(features, dtype=np.float32)
        if result.shape != (self.obs_dim,) or not np.all(np.isfinite(result)):
            raise ValueError(f"Invalid OPT observation: shape={result.shape}, expected={self.obs_dim}")
        return result

    def checkpoint_metadata(self) -> dict:
        provider = self.neighbor_feedback_provider
        return {
            "name": "opt_global_dc_load",
            "version": self.SCHEMA_VERSION,
            "blocks": ["job", "all_dc_loads", "route_history", "source_links", "neighbor_feedback"],
            "dc_ids": list(self.state_dc_ids),
            "dc_features": list(self.DC_FEATURE_NAMES),
            "link_target_dc_ids": list(self.link_target_dc_ids),
            "feedback_features": list(self.FEEDBACK_FEATURE_NAMES),
            "use_neighbor_historical_feedback": self.use_neighbor_historical_feedback,
            "feedback_parameters": {
                name: float(getattr(provider, name))
                for name in ("ewma_alpha", "age_scale_samples", "confidence_scale_samples")
                if provider is not None and hasattr(provider, name)
            },
            "obs_dim": self.obs_dim,
        }
