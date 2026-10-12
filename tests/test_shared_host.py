"""Shared Host contract, convergence gates, and five real Routing training paths."""
import contextlib
import copy
from dataclasses import replace
import importlib
import io
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

from host_test_support import make_host_bundle
from test_mr_h_masac import make_env, workspace_temp
from schedulers import shared_host as shared
from schedulers import pretrain_host as pretrain


class SharedHostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_validation_and_immutable_policy(self):
        env = make_env(jobs=15)
        with workspace_temp() as temp:
            bundle = make_host_bundle(env, temp / "host")
            frozen = shared.load_frozen_hosts(env, bundle, "cpu")
            for agent in frozen.values():
                self.assertFalse(hasattr(agent, "actor_optimizer"))
                self.assertFalse(any(p.requires_grad for p in agent.actor.parameters()))
                with self.assertRaises(RuntimeError):
                    agent.update(None, 2)
                with self.assertRaises(RuntimeError):
                    agent.train_mode()
                with self.assertRaises(RuntimeError):
                    agent.select_action(np.zeros(agent.obs_dim), deterministic=False)
                agent.assert_unchanged()
            env.base_datacenters[0].host_list.reverse()
            with self.assertRaisesRegex(ValueError, "不兼容"):
                shared.load_frozen_hosts(env, bundle, "cpu")
            env.base_datacenters[0].host_list.reverse()
            make_host_bundle(env, temp / "unfinished", status="unconverged")
            with self.assertRaisesRegex(ValueError, "收敛"):
                shared.load_frozen_hosts(env, temp / "unfinished", "cpu")
            path = bundle / "hosts" / f"{env.edge_dc_ids[-1]}.pt"
            path.write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "校验"):
                shared.load_frozen_hosts(env, bundle, "cpu")

    def test_manual_acceptance_preserves_weights_and_validation(self):
        env = make_env(jobs=15)
        with workspace_temp() as temp:
            source = make_host_bundle(env, temp / "source", status="unconverged")
            metadata = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
            metadata["training"].update(episode=1550, last_evaluation={"completion_rate": .32})
            shared.atomic_json(metadata, source / "manifest.json")
            source_hash = shared.sha256(source / "manifest.json")
            target = shared.accept_host_bundle(source, temp / "final", reason="User accepts episode 1550")
            _, accepted = shared.read_manifest(target)
            self.assertEqual(accepted["status"], "accepted_manual")
            self.assertEqual(accepted["training"], metadata["training"])
            self.assertEqual(accepted["files"], metadata["files"])
            self.assertEqual(accepted["acceptance"]["source_episode"], 1550)
            self.assertFalse(accepted["acceptance"]["automatic_convergence_passed"])
            self.assertEqual(shared.sha256(source / "manifest.json"), source_hash)
            for agent in shared.load_frozen_hosts(env, target, "cpu").values():
                agent.assert_unchanged()
            with self.assertRaises(FileExistsError):
                shared.accept_host_bundle(source, target, reason="Do not overwrite")
            with self.assertRaises(ValueError):
                shared.accept_host_bundle(source, temp / "empty_reason", reason=" ")
            accepted.pop("acceptance")
            shared.atomic_json(accepted, target / "manifest.json")
            with self.assertRaisesRegex(ValueError, "验收"):
                shared.read_manifest(target)
            (source / "hosts" / f"{env.edge_dc_ids[0]}.pt").write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "校验"):
                shared.accept_host_bundle(source, temp / "bad_source", reason="Reject bad source")

    def test_five_routing_trainers_share_identical_host_and_resume(self):
        schedulers = [("H-MASAC", "train_h_masac", "h_masac_agent"),
                      ("MR-H-MASAC", "train_mr_h_masac", "mr_h_masac_agent"),
                      ("BCGH-MASAC", "train_bcgh_masac", "h_masac_agent"),
                      ("BCGH2-MASCA", "train_bcgh2_masac", "h_masac_agent"),
                      ("MR-BCGH-MASAC", "train_mr_bcgh_masac", "h_masac_agent")]
        env = make_env(jobs=30)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        with workspace_temp() as temp:
            source = make_host_bundle(env, temp / "host_source", device, status="unconverged")
            metadata = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
            metadata["training"]["episode"] = 1550
            shared.atomic_json(metadata, source / "manifest.json")
            bundle = shared.accept_host_bundle(source, temp / "host", reason="Explicit manual acceptance test")
            expected = shared.load_frozen_hosts(env, bundle, device)
            expected_digests = {dc: shared.parameter_digest(agent) for dc, agent in expected.items()}
            reference = shared.host_reference(expected)
            for name, entry, networks in schedulers:
                if device == "cpu" and "BCGH" in name:
                    continue  # These existing Routing implementations require CUDA.
                with self.subTest(scheduler=name):
                    trainer = importlib.import_module(f"schedulers.{name}.{entry}")
                    agents = importlib.import_module(f"schedulers.{name}.{networks}")
                    cfg = trainer.TrainConfig(host_checkpoint=str(bundle), num_episodes=2,
                        host_pretrain_episodes=0, routing_train_episodes=2, joint_finetune_episodes=0,
                        routing_batch_size=2, routing_replay_capacity=160, routing_random_warmup_steps=1,
                        routing_learning_starts=2, routing_train_every=1, routing_updates_per_train=1,
                        checkpoint_dir=str(temp / name), episode_log_csv_path=str(temp / name / "episode.csv"),
                        dc_log_csv_path=str(temp / name / "dc.csv"))
                    options = dict(actor_hidden_dim=16, critic_hidden_dim=16, device=device)
                    if "BCGH" not in name:
                        options["allow_cpu"] = device == "cpu"
                    route_config = agents.RoutingMASACConfig(**options)
                    host_config = shared.HostSACConfig(actor_hidden_dim=16, critic_hidden_dim=16,
                                                       device=device, allow_cpu=device == "cpu")
                    with patch.object(trainer, "build_environment", return_value=env), contextlib.redirect_stdout(io.StringIO()):
                        routing, hosts = trainer.train(cfg, route_config, host_config)
                    self.assertGreater(routing.update_step, 0)
                    self.assertEqual(shared.host_reference(hosts), reference)
                    self.assertEqual({dc: shared.parameter_digest(agent) for dc, agent in hosts.items()}, expected_digests)
                    path = temp / name / "final.pt"
                    metadata = json.loads(path.with_suffix(".trainer.json").read_text(encoding="utf-8"))
                    self.assertEqual(metadata["shared_host"], reference)
                    self.assertEqual(metadata["trainer_state"]["host_global_steps"], 0)
                    with patch.object(trainer, "build_environment", return_value=env), contextlib.redirect_stdout(io.StringIO()):
                        resumed, resumed_hosts = trainer.train(replace(cfg, num_episodes=3, routing_train_episodes=3,
                                                                      resume_checkpoint=str(path)), route_config, host_config)
                    self.assertGreater(resumed.update_step, routing.update_step)
                    self.assertEqual({dc: shared.parameter_digest(agent) for dc, agent in resumed_hosts.items()}, expected_digests)
                    before = copy.deepcopy(resumed.actor.state_dict()) if hasattr(resumed, "actor") else None
                    (path.with_name("final_hosts") / f"{env.edge_dc_ids[-1]}.pt").write_bytes(b"corrupted")
                    with self.assertRaises(ValueError):
                        trainer.load_two_layer_checkpoint_if_needed(env, cfg, resumed, resumed_hosts, str(path))
                    if before:
                        for key, value in resumed.actor.state_dict().items():
                            self.assertTrue(torch.equal(value, before[key]))

    def test_convergence_requires_all_dcs_quality_stability_and_training(self):
        cfg = pretrain.HostPretrainConfig(min_episodes=2, min_samples_per_dc=2, min_updates_per_dc=2,
                                         stable_windows=2)
        row = dict(samples=10, reward=1.0, completion_rate=.8, sla_rate=.7,
                   completion_time=2.0, energy_per_job=3.0)
        baseline = {"A@1": {**row, "reward": .5}, "B@1": {**row, "reward": .5}}
        history = [{"A@1": row.copy(), "B@1": row.copy()} for _ in range(2)]
        self.assertTrue(pretrain.convergence_passed(history, baseline, cfg, 2, {"A":2,"B":2}, {"A":2,"B":2}))
        self.assertFalse(pretrain.convergence_passed(history, baseline, cfg, 2, {"A":2,"B":1}, {"A":2,"B":2}))
        history[-1]["B@1"]["reward"] = -10
        self.assertFalse(pretrain.convergence_passed(history, baseline, cfg, 2, {"A":2,"B":2}, {"A":2,"B":2}))

    def test_pretraining_collects_terminal_samples_and_validation_does_not_update(self):
        env = make_env(jobs=30)
        builder = shared.HostObservationBuilder(env)
        cfg = pretrain.HostPretrainConfig(device="cpu", min_episodes=1, max_episodes=2,
            batch_size=2, replay_capacity=64, random_warmup_steps=0, learning_starts=2,
            train_every=1, updates_per_train=1, eval_seeds=(1000001,), load_factors=(1.0,))
        agents = {dc: shared.LocalHostSAC(builder.get_obs_dim(dc), builder.get_action_dim(dc),
                    shared.HostSACConfig(actor_hidden_dim=16, critic_hidden_dim=16, device="cpu", allow_cpu=True))
                  for dc in env.edge_dc_ids}
        replay = {dc: pretrain.support.HostReplayBuffer(dc_id=dc,capacity=64,obs_dim=a.obs_dim,action_dim=a.action_dim,seed=42)
                  for dc,a in agents.items()}
        counters = {dc:0 for dc in agents}
        rngs = {dc:np.random.default_rng(42) for dc in agents}
        metrics = pretrain.run_host_episode(env, agents, cfg, 42, replay=replay, counters=counters, rngs=rngs)
        self.assertEqual(sum(row["samples"] for row in metrics.values()), 30)
        self.assertTrue(all(a.update_step > 0 for a in agents.values()))
        before = {dc:shared.parameter_digest(a) for dc,a in agents.items()}
        samples = {dc:buffer.total_added for dc,buffer in replay.items()}
        pretrain.evaluate_hosts(env, agents, cfg)
        self.assertEqual(before, {dc:shared.parameter_digest(a) for dc,a in agents.items()})
        self.assertEqual(samples, {dc:buffer.total_added for dc,buffer in replay.items()})

    def test_pretraining_limit_resume_and_publish_last_confirmed_weights(self):
        env = make_env(jobs=30)
        row = dict(samples=10, reward=1.0, completion_rate=.9, sla_rate=.8,
                   completion_time=2.0, energy_per_job=3.0)
        evaluation = {f"{dc}@1": row.copy() for dc in env.edge_dc_ids}
        baseline = {key: {**value, "reward": 0.0} for key,value in evaluation.items()}
        with workspace_temp() as temp:
            cfg = pretrain.HostPretrainConfig(output_dir=str(temp / "pretrain"), device="cpu",
                min_episodes=2, max_episodes=2, min_samples_per_dc=2, min_updates_per_dc=2,
                eval_interval=1, eval_seeds=(1000001,), load_factors=(1.0,), stable_windows=2,
                confirmation_evaluations=1, checkpoint_interval=2, replay_capacity=64,
                batch_size=2, random_warmup_steps=0, learning_starts=2, train_every=1,
                updates_per_train=1, save_replay=True)
            agent_config = shared.HostSACConfig(actor_hidden_dim=16, critic_hidden_dim=16,
                                               device="cpu", allow_cpu=True)
            with patch.object(pretrain, "evaluate_hosts", side_effect=[baseline,evaluation,evaluation]), contextlib.redirect_stdout(io.StringIO()):
                unfinished = pretrain.train(cfg, env=env, agent_config=agent_config)
            self.assertFalse((temp / "pretrain/final").exists())
            with self.assertRaisesRegex(ValueError, "收敛"):
                shared.load_frozen_hosts(env, unfinished, "cpu")
            cfg = replace(cfg, max_episodes=3, resume=str(unfinished))
            with patch.object(pretrain, "evaluate_hosts", return_value=evaluation), contextlib.redirect_stdout(io.StringIO()):
                final = pretrain.train(cfg, env=make_env(jobs=30), agent_config=agent_config)
            _, metadata = shared.read_manifest(final)
            self.assertEqual(metadata["training"]["episode"], 3)
            self.assertTrue(all(value > 0 for value in metadata["training"]["updates"].values()))
            state = torch.load(final / "training_state.pt", weights_only=False)
            self.assertEqual(state["episode"], 3)
            self.assertEqual(state["confirmations"], 2)
            self.assertEqual(state["replay_policy"], "restore")
            shared.load_frozen_hosts(env, final, "cpu")

    def test_routing_guidance_uses_full_budget(self):
        settings = importlib.import_module("schedulers.BCGH-MASAC.training_support").TrainConfig(
            routing_train_episodes=5, guidance_lambda_stage2_start=.2,
            guidance_lambda_stage2_peak=1., guidance_lambda_stage3_end=.3)
        self.assertEqual(shared.routing_guidance_lambda(settings, 1), .2)
        self.assertEqual(shared.routing_guidance_lambda(settings, 3), 1.)
        self.assertAlmostEqual(shared.routing_guidance_lambda(settings, 5), .3)


if __name__ == "__main__":
    unittest.main()
