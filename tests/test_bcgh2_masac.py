"""BCGH2 state capture, TD targets, CUDA training, evaluation and resume."""

import ast
import contextlib
import csv
from dataclasses import replace
import importlib
import io
import json
import math
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import uuid

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
support = importlib.import_module("schedulers.BCGH2-MASAC.training_support")
agent_module = importlib.import_module("schedulers.BCGH2-MASAC.h_masac_agent")
experience = importlib.import_module("schedulers.BCGH2-MASAC.experience")
trainer = importlib.import_module("schedulers.BCGH2-MASAC.train_bcgh2_masac")


@contextlib.contextmanager
def test_directory():
    directory = ROOT / "tests" / f"bcgh2-smoke-{uuid.uuid4().hex}"
    directory.mkdir()
    try:
        yield directory
    finally:
        shutil.rmtree(directory)


def game():
    return support.BayesianCongestionGame(
        NS(player_ids=("A", "B", "C")),
        capacities={
            dc: support.StaticDCCapacity(100, 10, ((100, 10),))
            for dc in ("A", "B", "C")
        },
    )


def route(
    store, agent="A", time=0, kind="edge_dc", target="B", source="policy", job="J"
):
    store.record_routing_step(
        job_id=job,
        agent_id=agent,
        agent_index={"A": 0, "B": 1}[agent],
        env_time=time,
        local_obs=np.array([time], dtype=np.float32),
        global_state=np.array([time, 1], dtype=np.float32),
        action=0,
        action_type=kind,
        action_source=source,
        immediate_reward=-0.1,
        target_dc_id=target,
    )


def successor(store, score=0.4, job="J", valid=True):
    return store.resolve_routing_successor(
        job_id=job,
        next_agent_id="B",
        next_agent_index=1,
        next_env_time=2,
        next_local_obs=np.array([2], dtype=np.float32),
        next_global_state=np.array([2, 1], dtype=np.float32),
        next_heuristic_score=score,
        heuristic_valid=valid,
    )


from host_test_support import make_host_bundle

class StateScoreTests(unittest.TestCase):
    def test_six_modules_and_no_cross_scheduler_imports(self):
        folder = Path(support.__file__).parent
        self.assertEqual(
            {p.name for p in folder.glob("*.py")},
            {
                "__init__.py",
                "train_bcgh2_masac.py",
                "training_support.py",
                "h_masac_agent.py",
                "experience.py",
                "observations.py",
            },
        )
        for path in folder.glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn(
                        node.module,
                        {"bayesian_game", "bayesian_congestion_game", "guided_policy"},
                    )

    def test_neutral_score_and_uncentered_utility(self):
        cfg = support.TrainConfig()
        rows = [NS(feasible=True, guidance_utility=0.5, bias=100)]
        self.assertEqual(support.state_score_from_utilities(rows, cfg), 0)
        rows.append(NS(feasible=True, guidance_utility=0.9, bias=-100))
        self.assertAlmostEqual(
            support.state_score_from_utilities(rows, cfg), math.tanh(0.4) / 2
        )
        rows.append(NS(feasible=False, guidance_utility=float("nan"), bias=0))
        self.assertAlmostEqual(
            support.state_score_from_utilities(rows, cfg), math.tanh(0.4) / 2
        )

    def test_score_bounds_and_monotonicity(self):
        cfg = replace(support.TrainConfig(), heuristic_scale=2)
        rows = [NS(feasible=True, guidance_utility=1e6)]
        self.assertEqual(support.state_score_from_utilities(rows, cfg), 2)
        rows[0].guidance_utility = -1e6
        self.assertEqual(support.state_score_from_utilities(rows, cfg), -2)
        rows[0].guidance_utility = float("nan")
        with self.assertRaises(ValueError):
            support.state_score_from_utilities(rows, cfg)

    def test_competition_changes_score_without_future_outcome(self):
        g = game()
        runtime = support.ShortWindowRuntime(g)
        cfg = support.TrainConfig()
        scorer = support.RoutingStateHeuristic(runtime, None, cfg, ("A", "B", "C"))
        decision = NS(agent_id="A", job_id="J", env_time=1, forced_action=None)
        env = NS(
            dc_map={"A": NS(host_list=[])},
            job_map={"J": NS(cpu_request=10, gpu_request=1)},
        )
        first = scorer.capture(env, decision)
        g.update_pressure(
            "C", "B", event_id="transfer", event_time=1, cpu_request=80, gpu_request=8
        )
        self.assertLess(scorer.capture(env, decision), first)
        self.assertEqual(g.belief_store.get_state("A", "B").sample_weight, 0)

    def test_trace_captures_same_job_score_and_terminal_zero(self):
        traces = experience.PendingJobTraceStore()
        route(traces)
        self.assertFalse(successor(traces, job="OTHER"))
        self.assertTrue(successor(traces))
        # Resolving again with a later belief cannot overwrite the original score.
        self.assertFalse(successor(traces, score=-0.9))
        route(traces, agent="B", time=2, kind="cloud", target="cloud")
        traces.record_reward_event(
            job_id="J", env_time=5, reward_delta=2, reason="completed", terminal=True
        )
        finalized = traces.finalize_terminal_trace("J")
        first, terminal = finalized.routing_transitions
        self.assertFalse(first.done)
        self.assertEqual(first.next_agent_id, "B")
        self.assertEqual(first.next_heuristic_score, 0.4)
        self.assertEqual(first.heuristic_timestamp, 2)
        self.assertAlmostEqual(first.reward, -0.1)
        self.assertTrue(terminal.done)
        self.assertEqual(terminal.next_heuristic_score, 0)
        self.assertTrue(terminal.heuristic_valid)
        self.assertTrue(np.all(terminal.next_global_state == 0))
        self.assertAlmostEqual(terminal.reward, 1.9)
        replay = experience.RoutingReplayBuffer(8, 1, 2, seed=1)
        for transition in finalized.routing_transitions:
            replay.add(transition)
        batch = replay.sample(2)
        self.assertEqual(set(batch.next_heuristic_scores), {np.float32(0.4), 0})
        self.assertTrue(batch.heuristic_valid.all())
        with self.assertRaises(ValueError):
            replay.add(replace(first, heuristic_valid=False))
        with self.assertRaises(ValueError):
            replay.add(replace(terminal, next_heuristic_score=0.1))
        replay.clear()
        self.assertEqual(len(replay), 0)

    def test_missing_score_rejected_at_successor_capture(self):
        traces = experience.PendingJobTraceStore()
        route(traces)
        with self.assertRaises(ValueError):
            successor(traces, valid=False)
        collector = support.TransitionCollector.__new__(support.TransitionCollector)
        collector.env = NS(
            current_job_id="J", current_time=2, agent_name_mapping={"B": 1}
        )
        collector.short_window_runtime = None
        collector.state_heuristic = None
        collector.pending_trace_store = traces
        collector.routing_observation_builder = NS(
            build=lambda _: np.array([2], dtype=np.float32)
        )
        collector.routing_state_builder = NS(
            build=lambda: np.array([2, 1], dtype=np.float32)
        )
        collector._require_reset = lambda: None
        collector._get_live_selected_agent = lambda: "B"
        collector._get_forced_action = lambda _: None
        with self.assertRaises(ValueError):
            collector.capture_decision()

    def test_validation_and_independent_default_paths(self):
        cfg = support.TrainConfig()
        self.assertIn("BCGH2-MASAC", cfg.checkpoint_dir)
        for change in (
            {"heuristic_beta": 2},
            {"heuristic_scale": -1},
            {"heuristic_reference_distribution": "actor"},
            {"enable_bayesian_game": False},
        ):
            with self.assertRaises(ValueError):
                support.validate_bcgh_feature_config(replace(cfg, **change))
        support.validate_bcgh_feature_config(
            replace(cfg, enable_bayesian_game=False, enable_heuristic_guidance=False)
        )


class TargetTests(unittest.TestCase):
    def target(self, **changes):
        cfg = replace(agent_module.RoutingMASACConfig(), **changes)
        return agent_module.routing_td_target(
            torch.tensor([1.0, -2.0]),
            torch.tensor([0.5, 0.0]),
            torch.tensor([0.0, 1.0]),
            torch.tensor([4.0, 999.0]),
            cfg,
            torch.tensor([True, True]),
        )

    def test_complete_reward_discount_target(self):
        target, bonus, modified, gamma = self.target()
        torch.testing.assert_close(target, torch.tensor([4.8907, -2.0]))
        torch.testing.assert_close(bonus, torch.tensor([0.0099, 0]))
        torch.testing.assert_close(modified, torch.tensor([1.0099, -2]))
        self.assertAlmostEqual(gamma, 0.9702)

    def test_disabled_and_beta_one_recover_original_target(self):
        for changes in ({"enable_heuristic_guidance": False}, {"heuristic_beta": 1}):
            target, bonus, _, gamma = self.target(**changes)
            torch.testing.assert_close(target, torch.tensor([4.96, -2]))
            self.assertEqual(float(bonus.abs().sum()), 0)
            self.assertEqual(gamma, 0.99)

    def test_reward_only_and_discount_only_are_separate(self):
        target, _, _, gamma = self.target(enable_discount_reduction=False)
        torch.testing.assert_close(target, torch.tensor([4.9699, -2]))
        self.assertEqual(gamma, 0.99)
        target, bonus, _, _ = self.target(enable_reward_shaping=False)
        torch.testing.assert_close(target, torch.tensor([4.8808, -2]))
        self.assertEqual(float(bonus.abs().sum()), 0)

    def test_invalid_terminal_score_rejected(self):
        with self.assertRaises(ValueError):
            agent_module.routing_td_target(
                torch.ones(1),
                torch.ones(1),
                torch.ones(1),
                torch.ones(1),
                agent_module.RoutingMASACConfig(),
            )


@unittest.skipUnless(torch.cuda.is_available(), "Project agents require CUDA")
class TrainingTests(unittest.TestCase):
    def test_disabled_guidance_matches_original_network_updates(self):
        old_agent = importlib.import_module("schedulers.BCGH-MASAC.h_masac_agent")
        old_experience = importlib.import_module("schedulers.BCGH-MASAC.experience")
        torch.set_num_threads(1)
        common = dict(actor_hidden_dim=8, critic_hidden_dim=8, device="cuda:0", seed=7)
        original = old_agent.RoutingMASAC(
            1, 2, 2, 2, old_agent.RoutingMASACConfig(**common)
        )
        updated = agent_module.RoutingMASAC(
            1,
            2,
            2,
            2,
            agent_module.RoutingMASACConfig(**common, enable_heuristic_guidance=False),
        )
        traces = experience.PendingJobTraceStore()
        route(traces)
        successor(traces)
        route(traces, agent="B", time=2, kind="cloud", target="cloud")
        traces.record_reward_event(
            job_id="J", env_time=5, reward_delta=2, reason="completed", terminal=True
        )
        transitions = traces.finalize_terminal_trace("J").routing_transitions
        old_replay = old_experience.RoutingReplayBuffer(8, 1, 2, seed=13)
        new_replay = experience.RoutingReplayBuffer(8, 1, 2, seed=13)
        for item in transitions:
            old_replay.add(item)
            new_replay.add(item)
        observation = np.array([0.7], dtype=np.float32)
        self.assertEqual(
            original.select_action(observation, 0, deterministic=True),
            updated.select_action(observation, 0, deterministic=True),
        )
        original.update(old_replay, 2)
        updated.update(new_replay, 2)
        for module in ("actor", "critic", "target_critic"):
            for name, value in getattr(original, module).state_dict().items():
                torch.testing.assert_close(
                    value, getattr(updated, module).state_dict()[name]
                )
        torch.testing.assert_close(original.log_alpha, updated.log_alpha)

    def test_frozen_host_logs_evaluation_and_checkpoint_resume(self):
        import config

        torch.set_num_threads(1)
        environments = []
        original_build = trainer.build_environment

        def build(*args, **kwargs):
            env = original_build(*args, **kwargs)
            environments.append(env)
            return env

        with test_directory() as temp:
            cfg = support.TrainConfig(
                num_episodes=3,
                host_pretrain_episodes=0,
                routing_train_episodes=3,
                joint_finetune_episodes=0,
                checkpoint_dir=str(temp / "checkpoints"),
                episode_log_csv_path=str(temp / "episode.csv"),
                dc_log_csv_path=str(temp / "dc.csv"),
                routing_batch_size=4,
                host_batch_size=4,
                routing_replay_capacity=512,
                host_replay_capacity=512,
                routing_random_warmup_steps=0,
                host_random_warmup_steps=0,
                routing_learning_starts=4,
                host_learning_starts=4,
                routing_train_every=1,
                host_train_every=1,
            )
            routing_cfg = agent_module.RoutingMASACConfig(
                actor_hidden_dim=16, critic_hidden_dim=16, device="cuda:0"
            )
            host_cfg = agent_module.HostSACConfig(
                actor_hidden_dim=16, critic_hidden_dim=16, device="cuda:0"
            )
            with patch.object(config, "NUM_JOBS", 24), patch.object(
                config, "ENV_KEEP_PATH", str(temp / "env")
            ), patch.object(
                trainer, "build_environment", side_effect=build
            ), patch.object(
                trainer,
                "StateScoreLog",
                side_effect=AssertionError("Score JSONL logging is temporarily disabled"),
            ), patch.object(
                support.StateScoreLog,
                "record",
                side_effect=AssertionError("Score JSONL records must not be collected"),
            ), contextlib.redirect_stdout(
                io.StringIO()
            ):
                cfg = replace(cfg, host_checkpoint=str(make_host_bundle(original_build(seed=42), temp / "host", "cuda:0")))
                routing, hosts = trainer.train(cfg, routing_cfg, host_cfg)
                self.assertGreater(routing.update_step, 0)
                self.assertTrue(all(host.update_step == 0 for host in hosts.values()))
                self.assertTrue(routing.config.enable_heuristic_guidance)
                self.assertEqual(routing.config.heuristic_beta, 0.98)
                before = {k: v.clone() for k, v in routing.actor.state_dict().items()}
                critic_before = {
                    k: v.clone() for k, v in routing.critic.state_dict().items()
                }
                steps = routing.update_step
                with patch.object(
                    routing,
                    "update",
                    side_effect=AssertionError("Evaluation must not train"),
                ):
                    metrics = support.evaluate_episode(
                        environments[0], routing, hosts, cfg, deterministic=False
                    )
                    disabled_metrics = support.evaluate_episode(
                        environments[0],
                        routing,
                        hosts,
                        replace(
                            cfg,
                            enable_bayesian_game=False,
                            enable_heuristic_guidance=False,
                        ),
                        deterministic=False,
                    )
                    self.assertGreater(disabled_metrics["routing_decisions"], 0)
                self.assertGreater(metrics["routing_decisions"], 0)
                self.assertFalse(metrics["actor_logit_bias"])
                self.assertEqual(routing.update_step, steps)
                self.assertTrue(
                    all(
                        torch.equal(before[k], value)
                        for k, value in routing.actor.state_dict().items()
                    )
                )
                self.assertTrue(
                    all(
                        torch.equal(critic_before[k], value)
                        for k, value in routing.critic.state_dict().items()
                    )
                )
                checkpoint = temp / "checkpoints/final.pt"
                # Explicit weights-only initialization accepts compatible old role;
                # ordinary load must reject it even when tensor shapes match.
                legacy = torch.load(checkpoint, weights_only=False)
                legacy["algorithm_role"] = "routing_masac_ctde"
                legacy_path = temp / "legacy.pt"
                torch.save(legacy, legacy_path)
                with self.assertRaises(RuntimeError):
                    routing.load(legacy_path)
                routing.initialize_weights(legacy_path)
                with self.assertRaises(RuntimeError):
                    routing.config = replace(routing.config, heuristic_beta=0.9)
                    routing.load(checkpoint)
                routing.config = replace(routing.config, heuristic_beta=0.98)
                routing.load(checkpoint)
                resume_cfg = replace(
                    cfg,
                    resume_checkpoint=str(temp / "checkpoints/final.pt"),
                    num_episodes=4, routing_train_episodes=4,
                )
                resumed, _ = trainer.train(resume_cfg, routing_cfg, host_cfg)
                self.assertGreater(resumed.update_step, 0)
                with self.assertRaises(RuntimeError):
                    trainer.train(
                        replace(resume_cfg, heuristic_beta=0.9), routing_cfg, host_cfg
                    )
            metadata = json.loads(
                (temp / "checkpoints/final.trainer.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["bcgh2_guidance"]["scheduler"], "BCGH2-MASAC")
            self.assertFalse(metadata["bcgh2_guidance"]["actor_logit_bias"])
            self.assertFalse(metadata["bcgh2_guidance"]["data_augmentation"])
            self.assertEqual(metadata["trainer_state"]["next_episode"], 5)
            files = [p for p in temp.glob("episode*.csv") if "current" not in p.name]
            with files[0].open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(rows)
            self.assertIn("bcgh2_heuristic_reward_sum", rows[0])
            self.assertIn("bcgh2_effective_gamma", rows[0])
            self.assertEqual(list(temp.glob("*.scores.jsonl")), [])
            self.assertEqual(list(temp.glob("*.guidance.jsonl")), [])


if __name__ == "__main__":
    unittest.main()
