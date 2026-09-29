"""CPU-only checks for deferred training metrics and Host replay validation."""

import importlib
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
H_TRAINER = importlib.import_module("schedulers.H-MASAC.train_h_masac")
H_AGENTS = importlib.import_module("schedulers.H-MASAC.h_masac_agent")
BCGH_TRAINER = importlib.import_module("schedulers.BCGH-MASAC.train_bcgh_masac")
BCGH_AGENTS = importlib.import_module("schedulers.BCGH-MASAC.h_masac_agent")
OPT_BASE = importlib.import_module("schedulers.OPT._base")


class TransferOptimizationTests(unittest.TestCase):
    def test_opt_uses_the_shared_h_masac_trainer_and_agents(self):
        self.assertIs(OPT_BASE.trainer, H_TRAINER)
        self.assertIs(OPT_BASE.agents, H_AGENTS)

    def test_nonterminal_host_batch_is_rejected_before_device_transfer(self):
        replay = SimpleNamespace(
            sample=lambda **_: SimpleNamespace(done=np.array([1.0, 0.0], dtype=np.float32))
        )
        for agents in (H_AGENTS, BCGH_AGENTS):
            with self.subTest(agent=agents.__name__):
                agent = agents.LocalHostSAC.__new__(agents.LocalHostSAC)
                with self.assertRaisesRegex(RuntimeError, "non-terminal Host transition"):
                    agent.update(replay, batch_size=2)

    def test_deferred_metrics_match_immediate_statistics(self):
        for trainer in (H_TRAINER, BCGH_TRAINER):
            with self.subTest(trainer=trainer.__name__):
                names = trainer.UPDATE_TENSOR_METRIC_NAMES
                first = {
                    name: torch.tensor(float(i + 1), requires_grad=True)
                    for i, name in enumerate(names)
                }
                second = {
                    name: torch.tensor(float(i + 2), requires_grad=True)
                    for i, name in enumerate(names)
                }
                first["actor_loss"] = torch.tensor(float("nan"), requires_grad=True)

                def stats():
                    return trainer.EpisodeStatistics(
                        episode=1,
                        episode_seed=42,
                        training_stage="routing_train",
                        per_agent_returns={},
                    )

                expected = stats()
                for row in trainer._convert_update_block_to_cpu([first, second]):
                    expected.record_routing_update(row)
                for dc_id, row in zip(
                    ("DC-1", "DC-2"),
                    trainer._convert_update_block_to_cpu([first, second]),
                ):
                    expected.record_host_update(dc_id, row)

                actual = stats()
                trainer.record_routing_update_block(actual, [first, second])
                trainer.record_host_update_block(actual, "DC-1", [first])
                trainer.record_host_update_block(actual, "DC-2", [second])
                self.assertEqual(actual.routing_update_count, 0)
                self.assertEqual(actual.host_update_count, 0)
                self.assertFalse(actual.pending_routing_update_infos[0][names[0]].requires_grad)
                trainer.flush_pending_update_metrics(actual)
                for key in (
                    "routing_update_count",
                    "host_update_count",
                    "routing_update_metric_sums",
                    "routing_update_metric_counts",
                    "host_update_metric_sums",
                    "host_update_metric_counts",
                    "host_update_metric_sums_by_dc",
                    "host_update_metric_counts_by_dc",
                    "dc_counters",
                ):
                    self.assertEqual(getattr(actual, key), getattr(expected, key), key)
                trainer.flush_pending_update_metrics(actual)
                self.assertEqual(actual.routing_update_count, 2)
                self.assertEqual(actual.host_update_count, 2)


if __name__ == "__main__":
    unittest.main()
