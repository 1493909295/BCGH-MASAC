"""Synthetic Host bundles for compatibility tests, never production training."""
import importlib
from pathlib import Path

from schedulers.shared_host import HostObservationBuilder, HostSACConfig, LocalHostSAC, save_host_bundle


def make_host_bundle(env, destination, device="cpu", status="converged"):
    builder = HostObservationBuilder(env)
    config = HostSACConfig(actor_hidden_dim=16, critic_hidden_dim=16,
                           device=device, allow_cpu=device == "cpu")
    agents = {dc: LocalHostSAC(builder.get_obs_dim(dc), builder.get_action_dim(dc), config)
              for dc in builder.edge_dc_ids}
    return save_host_bundle(agents, env, Path(destination), status=status,
                            training={"purpose": "synthetic test fixture; not a trained model"})
