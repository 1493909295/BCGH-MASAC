"""Qualified imports keep the shared H-MASAC implementation isolated from BGH.

The existing directory name contains a hyphen, so use importlib instead of
adding it to sys.path or importing ambiguous names such as h_masac_agent.
"""
from importlib import import_module

trainer = import_module("schedulers.H-MASAC.train_h_masac")
observations = import_module("schedulers.H-MASAC.routing_observation")
agents = import_module("schedulers.H-MASAC.h_masac_agent")

RoutingMASACConfig = agents.RoutingMASACConfig
HostSACConfig = agents.HostSACConfig
LocalHostSAC = agents.LocalHostSAC
HostObservationBuilder = trainer.HostObservationBuilder
RoutingCentralizedStateBuilder = trainer.RoutingCentralizedStateBuilder
NeighborHistoricalFeedbackStore = trainer.NeighborHistoricalFeedbackStore
PendingJobTraceStore = trainer.PendingJobTraceStore
TransitionCollector = trainer.TransitionCollector
TrainingRewardConfig = trainer.TrainingRewardConfig
TrainingRewardModel = trainer.HMasacTrainingRewardModel
