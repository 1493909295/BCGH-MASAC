"""Same discrete SAC updates as H-MASAC, with an explicit OPT model identity."""
from ._base import agents


class OPTRoutingMASAC(agents.RoutingMASAC):
    algorithm_role = "opt_routing_masac_global_load"
    checkpoint_architecture = "opt_two_layer_global_load_v1"
    scheduler_name = "OPT"

    def save(self, file_path):
        if self.observation_metadata is None:
            raise RuntimeError("Bind the OPT observation schema before saving a checkpoint")
        super().save(file_path)

    def load(self, file_path, load_optimizers=True):
        if self.observation_metadata is None:
            raise RuntimeError("Bind the OPT observation schema before loading a checkpoint")
        super().load(file_path, load_optimizers=load_optimizers)
