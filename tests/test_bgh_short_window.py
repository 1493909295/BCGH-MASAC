"""Run: python -B -m unittest discover -s tests -p 'test_bgh_short_window.py' -v"""
import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "schedulers" / "BGH-MASAC"))
sys.path.insert(0, str(ROOT))

from bayesian_game import BayesianHistoricalEvidence
from bayesian_congestion_game import (BayesianBeliefStore, BayesianCongestionGame,
    AbsorptionEvidence, RoutingActionFeatures, StaticDCCapacity, LocalSourceSignal)
from short_window_runtime import ShortWindowRuntime
from guidance_schedule import GuidanceLambdaSchedule
from neighbor_feedback import ShortWindowFeedbackStore


def game(window=10):
    return BayesianCongestionGame(NS(player_ids=("A", "B", "C")),
        capacities={j: StaticDCCapacity(100, 10, ((100, 10),)) for j in ("A", "B", "C")},
        window_s=window)


def evidence(key="one", routed=0.0, terminal=1.0, bad=True):
    return BayesianHistoricalEvidence("A", "B", "risky" if bad else "good",
        not bad, not bad, completion_time_s=None if bad else terminal-routed,
        timestamp=terminal, routing_time=routed, evidence_id=key)


@contextlib.contextmanager
def writable_test_directory():
    """Use inherited workspace ACLs; tempfile's restrictive ACL is unusable on this host."""
    path = ROOT / "tests" / f"bgh-window-smoke-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield str(path)
    finally:
        shutil.rmtree(path)


class WindowTests(unittest.TestCase):
    def test_reforward_is_not_double_counted_as_terminal_congestion(self):
        item = replace(evidence(bad=False), reforwarded=True)
        self.assertFalse(item.congestion_observed)

    def test_result_expiry_and_dedup(self):
        store = BayesianBeliefStore(("A", "B"), window_s=10)
        store.update(evidence())
        store.update(evidence())
        self.assertEqual(store.get_state("A", "B").sample_weight, 1)
        self.assertAlmostEqual(store.get_congestion_probability("A", "B"), 2/3)
        store.advance_time(10)
        self.assertEqual(store.get_congestion_probability("A", "B"), .5)
        self.assertEqual(store.get_confidence("A", "B"), 0)

    def test_late_outcome_does_not_revive_old_route(self):
        store = BayesianBeliefStore(("A", "B"), window_s=10)
        store.update(evidence(terminal=11))
        self.assertEqual(store.get_state("A", "B").sample_weight, 0)
        with self.assertRaises(ValueError):
            store.update(evidence(routed=11, terminal=13), update_clock=12)

    def test_boundary_fractional_seconds_and_reset(self):
        store = BayesianBeliefStore(("A", "B"), window_s=10)
        store.update(evidence(routed=.25, terminal=.5))
        store.advance_time(10.24)
        self.assertEqual(store.get_state("A", "B").sample_weight, 1)
        store.advance_time(10.25)
        self.assertEqual(store.get_state("A", "B").sample_weight, 0)
        with self.assertRaises(ValueError):
            store.advance_time(0)
        store.reset()
        store.advance_time(0)

    def test_feedback_expires_and_empty_delay_unknown(self):
        env = NS(edge_dc_ids=("A", "B"), max_job_duration=10,
                 drop_deadline_ratio=2, current_time=1)
        feedback = ShortWindowFeedbackStore(env, window_s=10)
        feedback.store.update(evidence())
        self.assertEqual(feedback.get_feedback("A", "B")["completion_time_ewma"], .5)
        env.current_time = 10
        self.assertIsNone(feedback.get_feedback("A", "B"))
        self.assertEqual(feedback.summary()["active_pair_count"], 0)
        feedback.reset_episode_counters()
        env.current_time = 0
        self.assertIsNone(feedback.get_feedback("A", "B"))

    def test_resource_cost_increases_with_demand_and_competition(self):
        g = game()
        small = g.get_congestion_cost("B", 10, 1)
        self.assertGreater(g.get_congestion_cost("B", 20, 2), small)
        g.advance_time(1)
        kwargs = dict(event_id="transfer", event_time=1, cpu_request=40, gpu_request=4)
        g.update_pressure("A", "B", **kwargs)
        g.update_pressure("A", "B", **kwargs)
        self.assertEqual(g.get_pressure("B"), (.4, .4))
        self.assertGreater(g.get_congestion_cost("B", 10, 1), small)
        g.advance_time(11)
        self.assertEqual(g.get_congestion_cost("B", 10, 1), small)

    def test_impossible_static_resource_and_host_fragmentation(self):
        g = game()
        g.pressure_store.capacities["B"] = StaticDCCapacity(100, 10, ((100, 0), (0, 10)))
        rows = g.evaluate_actions("A", {"A": RoutingActionFeatures("A"),
            "B": RoutingActionFeatures("B", cpu_request=10, gpu_request=1)})
        self.assertFalse(rows[1].feasible)
        self.assertTrue(rows[0].feasible)

    def test_source_evidence_works_without_outbound_samples_and_is_bounded(self):
        g = game()
        context = {j: RoutingActionFeatures(j) for j in ("A", "B", "C")}
        baseline = g.evaluate_actions("A", context)
        self.assertTrue(all(r.bias == 0 for r in baseline))
        context["B"] = replace(context["B"], source_signal=LocalSourceSignal(2, 1, 8))
        rows = g.evaluate_actions("A", context)
        self.assertGreater(rows[1].congestion_probability, .5)
        self.assertGreater(rows[1].source_confidence, 0)
        self.assertLess(rows[1].bias, rows[2].bias)
        self.assertLess(rows[1].source_pseudo_count, g.source_evidence_weight)
        self.assertEqual(rows, g.evaluate_actions("A", context))
        self.assertEqual(g.belief_store.get_state("A", "B").sample_weight, 0)
        self.assertTrue(all(abs(r.bias) <= g.max_logit_bias for r in rows))

    def test_pressure_not_silenced_by_zero_outcome_confidence(self):
        g = game()
        g.advance_time(1)
        g.update_pressure("C", "B", event_id="x", event_time=1, cpu_request=80, gpu_request=8)
        context = {j: RoutingActionFeatures(j, cpu_request=10, gpu_request=1) for j in ("A", "B", "C")}
        rows = g.evaluate_actions("A", context)
        self.assertLess(rows[1].bias, rows[2].bias)

    def test_local_queue_receipt_previous_hop_expiry_and_no_double_count(self):
        runtime = ShortWindowRuntime(game())
        job = NS(job_id="job", origin_datacenter="C", cpu_request=20, gpu_request=2)
        decision = NS(job_id="job", agent_id="B", env_time=0)
        runtime.record_routing(decision, "edge_dc", "A", "policy", job)
        local = NS(host_list=[NS(waiting_queue=NS(_queue=[job]), running_queue=NS(_queue=[]))])
        # Before actual arrival the pending transfer must not be visible.
        self.assertEqual(runtime.local_source_signals("A", local, 0), {})
        runtime.record_arrival("A", "job", 1)
        first = runtime.local_source_signals("A", local, 1)
        self.assertIn("B", first)
        self.assertNotIn("C", first)
        self.assertEqual(first, runtime.local_source_signals("A", local, 1))
        local.host_list[0].waiting_queue._queue.clear()
        local.host_list[0].running_queue._queue.append(job)
        self.assertGreater(runtime.local_source_signals("A", local, 2)["B"].running_volume, 0)
        self.assertEqual(runtime.local_source_signals("A", local, 11), {})
        runtime.reset()
        self.assertEqual(runtime.local_source_signals("A", local, 0), {})

    def test_random_offload_is_competition_but_not_source_evidence(self):
        runtime = ShortWindowRuntime(game())
        job = NS(job_id="j", cpu_request=20, gpu_request=2)
        runtime.record_routing(NS(job_id="j", agent_id="B", env_time=0), "edge_dc", "A", "random", job)
        runtime.record_arrival("A", "j", 1)
        dc = NS(host_list=[NS(waiting_queue=NS(_queue=[job]), running_queue=NS(_queue=[]))])
        self.assertEqual(runtime.local_source_signals("A", dc, 1), {})
        self.assertEqual(runtime.game.get_pressure("A"), (.2, .2))

    def test_edge_to_cloud_updates_directed_absorption_once(self):
        runtime = ShortWindowRuntime(game())
        job = NS(job_id="j", cpu_request=20, gpu_request=2)
        runtime.record_routing(NS(job_id="j", agent_id="A", env_time=0),
            "edge_dc", "B", "policy", job)
        runtime.record_arrival("B", "j", 1)
        runtime.record_routing(NS(job_id="j", agent_id="B", env_time=1),
            "cloud", "cloud", "policy", job)
        belief = runtime.game.get_absorption_belief("A", "B", 20, 2)
        self.assertEqual(belief.pair_samples, 1)
        self.assertGreater(belief.p_cloud, .25)
        self.assertEqual(runtime.absorption_summary()["counts"]["cloud"], 1)
        self.assertEqual(runtime.game.get_absorption_belief("B", "A", 20, 2).pair_samples, 0)

    def test_self_is_absorbed_only_after_real_host_admission(self):
        for result, expected in (("started", "self"), ("queued", "self"), ("dropped", "drop")):
            with self.subTest(result=result):
                runtime = ShortWindowRuntime(game())
                job = NS(job_id="j", cpu_request=10, gpu_request=0)
                runtime.record_routing(NS(job_id="j", agent_id="A", env_time=0),
                    "edge_dc", "B", "policy", job)
                runtime.record_arrival("B", "j", 1)
                runtime.record_routing(NS(job_id="j", agent_id="B", env_time=1),
                    "self", "B", "policy", job)
                self.assertEqual(runtime.game.get_absorption_belief("A", "B", 10, 0).pair_samples, 0)
                runtime.record_host_result("j", "B", result, 1)
                summary = runtime.absorption_summary()["counts"]
                self.assertEqual(summary[expected], 1)

    def test_non_policy_chain_is_logged_but_does_not_calibrate_belief(self):
        runtime = ShortWindowRuntime(game())
        job = NS(job_id="j", cpu_request=10, gpu_request=0)
        runtime.record_routing(NS(job_id="j", agent_id="A", env_time=0),
            "edge_dc", "B", "random", job)
        runtime.record_arrival("B", "j", 1)
        runtime.record_routing(NS(job_id="j", agent_id="B", env_time=1),
            "cloud", "cloud", "policy", job)
        summary = runtime.absorption_summary()
        self.assertEqual(summary["counts"]["cloud"], 1)
        self.assertEqual(summary["informative_total"], 0)
        self.assertEqual(runtime.game.get_absorption_belief("A", "B", 10, 0).pair_samples, 0)

    def test_absorption_expiry_and_cloud_penalty(self):
        g = game()
        context = {j: RoutingActionFeatures(j, cpu_request=10, gpu_request=1)
                   for j in ("A", "B", "C", "cloud")}
        baseline = g.evaluate_actions("A", context)
        self.assertTrue(all(r.transit_penalty == 0 for r in baseline))
        resource_class = g.absorption_store.resource_class("B", 10, 1)
        for index in range(8):
            g.record_absorption(AbsorptionEvidence(str(index), f"j{index}", "A", "B",
                float(index) / 10, 1.0 + float(index) / 10, "cloud", 10, 1,
                resource_class, "policy", "policy"))
        rows = {r.target_dc_id: r for r in g.evaluate_actions("A", context)}
        self.assertGreater(rows["B"].transit_penalty, 0)
        self.assertLess(rows["B"].bias, rows["C"].bias)
        self.assertEqual(rows["cloud"].transit_penalty, 0)
        g.advance_time(12)
        expired = g.get_absorption_belief("A", "B", 10, 1)
        self.assertEqual(expired.pair_samples, 0)
        self.assertEqual(expired.confidence, 0)

    def test_absorption_dedup_survives_window_expiry(self):
        g = game()
        resource_class = g.absorption_store.resource_class("B", 10, 0)
        item = AbsorptionEvidence("same", "j", "A", "B", 0, 1, "cloud", 10, 0,
                                  resource_class, "policy", "policy")
        self.assertTrue(g.record_absorption(item))
        g.advance_time(11)
        self.assertFalse(g.record_absorption(item))
        self.assertEqual(g.absorption_store.episode_summary()["counts"]["cloud"], 1)

    def test_episode_statistics_classifies_transit_pattern(self):
        import train_bgh_masac as trainer
        stats = trainer.EpisodeStatistics(1, 1, "routing_train", {})
        edge = NS(action_type="edge_dc", source_dc_id="A", target_dc_id="B",
                  agent_id="A", sequence_index=0)
        cloud = NS(action_type="cloud", source_dc_id="B", target_dc_id="cloud",
                   agent_id="B", sequence_index=1)
        trace = NS(job_id="j", routing_steps=(edge, cloud), routing_transitions=(),
                   host_transition=None)
        stats.record_finalized_trace(trace)
        self.assertEqual(stats.edge_successor_cloud_count, 1)
        self.assertEqual(stats.absorption_pair_counts["A->B"]["cloud"], 1)
        self.assertEqual(stats.first_edge_absorbed_count, 0)

    def test_schedule_end_and_single_episode(self):
        schedule = GuidanceLambdaSchedule()
        with self.assertRaises(ValueError):
            schedule.lambda_for_episode(stage="invalid", episode=1,
                stage_start_episode=1, stage_end_episode=1)
        self.assertEqual(schedule.lambda_for_episode(stage="joint_finetune", episode=10,
            stage_start_episode=5, stage_end_episode=10), .3)
        self.assertEqual(schedule.lambda_for_episode(stage="joint_finetune", episode=3,
            stage_start_episode=3, stage_end_episode=3), .3)

    def test_guided_policy_masks_impossible_remote(self):
        import numpy as np
        import torch
        from guided_policy import GuidedRoutingPolicy
        class Actor:
            def forward(self, **_):
                return torch.tensor([[0., 100., 0.]])
        base = NS(action_dim=3, device="cpu", actor=Actor())
        g = game()
        g.pressure_store.capacities["B"] = StaticDCCapacity(10, 0, ((10, 0),))
        policy = GuidedRoutingPolicy(base, ("A", "B", "C"), enabled=True, bayesian_game=g)
        context = policy.build_action_context("A", cpu_request=20, gpu_request=1)
        action = policy.select_action(np.zeros(1), 0, "A", action_context=context,
                                      deterministic=True)
        self.assertNotEqual(action, 1)


class TrainingSmokeTest(unittest.TestCase):
    def test_three_stage_training_updates_and_metadata(self):
        import torch
        import config
        import train_bgh_masac as trainer
        if not torch.cuda.is_available():
            self.skipTest("Project training requires CUDA")
        torch.set_num_threads(1)
        environments = []
        original_build = trainer.build_environment
        def capture_environment(*args, **kwargs):
            env = original_build(*args, **kwargs)
            environments.append(env)
            return env
        with writable_test_directory() as temp:
            with patch.object(config, "NUM_JOBS", 24), patch.object(config, "ENV_KEEP_PATH", temp + "/env"), \
                 patch.object(trainer, "build_environment", side_effect=capture_environment), \
                 contextlib.redirect_stdout(io.StringIO()):
                cfg = trainer.TrainConfig(num_episodes=3, host_pretrain_episodes=1,
                    routing_train_episodes=1, joint_finetune_episodes=1,
                    checkpoint_dir=temp + "/checkpoints", episode_log_csv_path=temp + "/episode.csv",
                    dc_log_csv_path=temp + "/dc.csv", old_env_path=None, resume_checkpoint=None,
                    checkpoint_interval=3, routing_batch_size=4, host_batch_size=4,
                    routing_replay_capacity=512, host_replay_capacity=512,
                    routing_random_warmup_steps=0, host_random_warmup_steps=0,
                    routing_learning_starts=4, host_learning_starts=4,
                    routing_train_every=1, host_train_every=1, short_window_s=1200)
                routing, hosts = trainer.train(cfg,
                    trainer.RoutingMASACConfig(actor_hidden_dim=16, critic_hidden_dim=16, device="cuda:0"),
                    trainer.HostSACConfig(actor_hidden_dim=16, critic_hidden_dim=16, device="cuda:0"))
            checkpoint_dir = Path(temp) / "checkpoints"
            self.assertEqual(
                {path.name for path in checkpoint_dir.glob("*.pt")},
                {"joint_finetune_start.pt", "final.pt"},
            )
            start_metadata = json.loads(
                (checkpoint_dir / "joint_finetune_start.trainer.json").read_text(encoding="utf-8")
            )
            self.assertEqual(start_metadata["saved_training_stage"], "joint_finetune")
            self.assertEqual(start_metadata["trainer_state"]["next_episode"], 3)
            self.assertEqual(
                {path.name for path in checkpoint_dir.iterdir() if path.is_dir()},
                {"joint_finetune_start_hosts", "final_hosts"},
            )
            metadata = json.loads((checkpoint_dir / "final.trainer.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["trainer_state"]["next_episode"], 4)
            self.assertEqual(metadata["bgh_guidance"]["version"], 3)
            self.assertTrue(metadata["bgh_guidance"]["parameters"]["absorption_enabled"])
            self.assertEqual(metadata["bgh_guidance"]["parameters"]["guidance_lambda_stage3_end"], .3)
            self.assertEqual(metadata["saved_training_stage"], "joint_finetune")
            records = [json.loads(line) for path in Path(temp).glob("*.guidance.jsonl")
                       for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(records)
            self.assertEqual(records[-1]["guidance_lambda"], .3)
            self.assertTrue(any(c["resource_cost_raw"] > 0 for r in records for c in r["candidates"]))
            self.assertTrue(all("cloud_escape_probability" in c and "transit_penalty" in c
                                for r in records for c in r["candidates"]))
            self.assertEqual(len(hosts), 5)
            self.assertTrue(any(c["source_signal"] > 0 for r in records for c in r["candidates"]))
            import csv
            episode_files = [p for p in Path(temp).glob("episode*.csv")]
            with episode_files[0].open(encoding="utf-8-sig", newline="") as log:
                logs = list(csv.DictReader(log))
            self.assertEqual(len(logs), 3)
            self.assertIn("edge_to_cloud_transit_rate", logs[-1])
            self.assertIn("first_edge_absorption_rate", logs[-1])
            self.assertIn("p95_routing_edge_hops", logs[-1])
            dc_files = [p for p in Path(temp).glob("dc*.csv")]
            with dc_files[0].open(encoding="utf-8-sig", newline="") as log:
                dc_logs = list(csv.DictReader(log))
            self.assertIn("incoming_edge_absorption_rate", dc_logs[-1])
            before = {k: v.clone() for k, v in routing.actor.state_dict().items()}
            from evaluate_bgh_masac import evaluate_episode
            with contextlib.redirect_stdout(io.StringIO()):
                metrics = evaluate_episode(environments[0], routing, hosts, cfg, deterministic=False)
            self.assertEqual(metrics["guidance_lambda"], .3)
            self.assertGreater(metrics["routing_decisions"], 0)
            self.assertIn("edge_to_cloud_transit_rate", metrics["absorption"])
            self.assertTrue(all(torch.equal(v, before[k]) for k, v in routing.actor.state_dict().items()))

    def test_cloud_enabled_training_and_evaluation(self):
        import config
        with patch.object(config, "ENABLE_CLOUD_ACTION", True):
            self.test_three_stage_training_updates_and_metadata()


if __name__ == "__main__":
    unittest.main()
