"""Run: python -B -m unittest discover -s tests -p test_opt_scheduler.py -v"""
import contextlib
import copy
import csv
from dataclasses import dataclass, replace
import io
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import networkx as nx
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from environment.cloud_edge_env import CloudEdgeEnv
from environment.datacenter import DataCenter, Host
from environment.datacenter import hosts_generate
from environment.job import Job
from schedulers.OPT._base import trainer, agents, observations
from schedulers.OPT.routing_observation import RoutingObservationBuilder
from schedulers.OPT.opt_agent import OPTRoutingMASAC
from schedulers.OPT.train_opt import (
    TrainConfig, train, initialize_hosts, model_config,
    validate_opt_device, choose_opt_device,
)
from schedulers.OPT.evaluate_opt import build_observers, evaluate_episode, load_models


@contextlib.contextmanager
def workspace_temp():
    parent = (ROOT / "tests").resolve()
    path = parent / f"opt-smoke-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        # Resolve and verify before recursive cleanup, including on Windows.
        target = path.resolve()
        if target.parent != parent or not target.name.startswith("opt-smoke-"):
            raise RuntimeError(f"Unsafe test cleanup path: {target}")
        shutil.rmtree(target)


def make_env(cloud=False, jobs=30):
    dcs = []
    graph = nx.Graph()
    for index in range(5):
        dc = DataCenter(f"DC-{index + 1}", 0.5)
        dc.host_list = [Host(f"host-{index}-{h}", 4, 16) for h in range(2)]
        dcs.append(dc)
    cloud_dc = DataCenter("cloud", 0.0)
    cloud_dc.host_list = [Host("cloud-host", 999999, 999999)]
    dcs.append(cloud_dc)
    for dc in dcs:
        graph.add_node(dc.dc_id, dc_instance=dc)
    for index, dc in enumerate(dcs):
        for target in dcs[index + 1:]:
            graph.add_edge(dc.dc_id, target.dc_id, weight=0.5 if target.dc_id == "cloud" else 0.25)
    source = SimpleNamespace(global_dc_list=dcs, datacenter_graph=graph, job_num=jobs)
    with patch.object(config, "ENABLE_CLOUD_ACTION", cloud):
        env = CloudEdgeEnv(env_source=source, seed=42)

    def workload():
        result = []
        counts = {dc_id: 0 for dc_id in env.edge_dc_ids}
        for index in range(jobs):
            dc_id = env.edge_dc_ids[index % len(env.edge_dc_ids)]
            job = Job(f"job-{index}", 2, 1, 3, dc_id, dc_id)
            job.set_arrive_time(index * 0.5)
            result.append(job)
            counts[dc_id] += 1
        env.episode_exogenous_arrival_counts = counts
        return result

    env._generate_episode_workload = workload
    return env


class ObservationTests(unittest.TestCase):
    def test_partial_deployment_reports_missing_shared_cpu_support(self):
        @dataclass(frozen=True)
        class LegacyRoutingMASACConfig:
            device: str = "cuda:0"

        with patch("schedulers.OPT.train_opt.RoutingMASACConfig", LegacyRoutingMASACConfig):
            with self.assertRaisesRegex(RuntimeError, "h_masac_agent.py") as error:
                validate_opt_device("cpu")
        self.assertIn("allow_cpu", str(error.exception))

    def test_incompatible_cuda_is_caught_before_training_and_auto_uses_cpu(self):
        agents.resolve_training_device.cache_clear()
        self.addCleanup(agents.resolve_training_device.cache_clear)
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "device_count", return_value=1), \
             patch.object(torch.cuda, "get_device_name", return_value="test GPU"), \
             patch.object(torch.cuda, "get_device_capability", return_value=(7, 0)), \
             patch.object(torch, "ones", side_effect=RuntimeError("no kernel image is available")):
            with self.assertRaisesRegex(RuntimeError, "--device cpu"):
                validate_opt_device("cuda:0")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(choose_opt_device("auto"), "cpu")
            self.assertIn("CPU", output.getvalue())
        with self.assertRaisesRegex(ValueError, "GPU"):
            agents.resolve_training_device("cpu")
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "manual_seed_all", side_effect=AssertionError("CUDA was seeded during CPU training")):
            trainer.set_global_random_seeds(11, device="cpu")
            cpu_agent = OPTRoutingMASAC(
                4, 4, 2, 2,
                config=agents.RoutingMASACConfig(
                    actor_hidden_dim=8, critic_hidden_dim=8,
                    device="cpu", allow_cpu=True, seed=11,
                ),
            )
            self.assertEqual(cpu_agent.device.type, "cpu")

    def test_missing_host_data_reports_path_instead_of_empty_host_list(self):
        missing = ROOT / "dataset" / "DC_dataset" / "missing-hosts.csv"
        with self.assertRaisesRegex(FileNotFoundError, "missing-hosts.csv"):
            hosts_generate([], 5, str(missing))
        with self.assertRaisesRegex(ValueError, "NUM_HOST"):
            hosts_generate([], 0, config.HOST_DATASET_PATH)
        with patch.object(config, "HOST_DATASET_PATH", str(missing)):
            with self.assertRaisesRegex(FileNotFoundError, "--host-dataset"):
                train(TrainConfig())
        with patch.object(config, "NUM_HOST", 0):
            with self.assertRaisesRegex(ValueError, "NUM_HOST=0"):
                train(TrainConfig())
        with patch.object(config, "HOST_DATASET_PATH", str(missing)), \
             patch.object(trainer, "train", return_value="uses_saved_environment"):
            self.assertEqual(train(TrainConfig(old_env_path="saved-env")), "uses_saved_environment")

    def test_remote_load_visible_only_to_opt_and_snapshot_is_stable(self):
        env = make_env(jobs=1)
        env.reset(seed=42)
        opt = RoutingObservationBuilder(env)
        local = observations.RoutingObservationBuilder(env)
        before = opt.build("DC-1")
        local_before = local.build("DC-1")
        remote_job = Job("remote-running", 8, 2, 10)
        env.dc_map["DC-2"].host_list[0].add_to_running_queue(remote_job, env.current_time)
        after = opt.build("DC-1")
        self.assertEqual(opt.obs_dim, 127)
        self.assertEqual(local.obs_dim, 62)
        block = opt.dc_slices["DC-2"]
        self.assertGreater(after[block][2], before[block][2])
        np.testing.assert_array_equal(before[:block.start], after[:block.start])
        np.testing.assert_array_equal(before[block.stop:], after[block.stop:])
        np.testing.assert_array_equal(local_before, local.build("DC-1"))
        # Maps may change insertion order; the schema order must stay fixed.
        env.dc_map = dict(reversed(list(env.dc_map.items())))
        np.testing.assert_array_equal(after, opt.build("DC-1"))
        # Future workload contents are not actor inputs.
        env.job_map["future"] = Job("future", 999, 999, 999)
        np.testing.assert_array_equal(after, opt.build("DC-1"))

    def test_cloud_switch_keeps_observation_shape_and_agent_identity_separate(self):
        observations_by_switch = []
        for cloud in (False, True):
            env = make_env(cloud, jobs=1)
            env.reset(seed=42)
            obs = RoutingObservationBuilder(env)
            self.assertEqual(obs.state_dc_ids, env.edge_dc_ids + ["cloud"])
            self.assertEqual(env.action_dim, 6 if cloud else 5)
            all_dc_end = obs.job_feat_dim + obs.global_dc_feat_dim
            np.testing.assert_array_equal(obs.build("DC-1")[:all_dc_end], obs.build("DC-2")[:all_dc_end])
            observations_by_switch.append(obs.build("DC-1"))
        np.testing.assert_array_equal(*observations_by_switch)

    def test_successor_observation_captures_new_global_load_without_rewriting_history(self):
        env = make_env(jobs=1)
        settings = TrainConfig()
        obs, state, _, _ = build_observers(env, settings)
        traces = trainer.PendingJobTraceStore()
        reward = SimpleNamespace(calculate_routing_immediate_reward=lambda facts: 0.0)
        collector = trainer.TransitionCollector(env, obs, state, traces, reward)
        env.reset(seed=42)
        first = collector.capture_decision()
        saved = first.local_obs.copy()
        env.dc_map["DC-3"].host_list[0].add_to_running_queue(Job("load", 8, 2, 10), env.current_time)
        collector.execute_and_record(first, env.routing_dc_id_to_action["DC-2"], action_source="policy")
        self.assertEqual(env.current_job_id, first.job_id)
        second = collector.capture_decision()
        step = traces.get_trace(first.job_id).routing_steps[0]
        self.assertEqual(step.next_agent_id, "DC-2")
        self.assertEqual(step.next_local_obs.shape, (127,))
        np.testing.assert_array_equal(step.local_obs, saved)
        np.testing.assert_array_equal(step.next_local_obs, second.local_obs)
        self.assertGreater(step.next_local_obs[obs.dc_slices["DC-3"]][2], saved[obs.dc_slices["DC-3"]][2])
        second.local_obs[:] = 0
        self.assertTrue(np.any(step.next_local_obs))

    def test_entry_point_uses_project_hyperparameters(self):
        for cls, prefix in ((agents.RoutingMASACConfig, "ROUTING"), (agents.HostSACConfig, "HOST")):
            cfg = model_config(cls, prefix, seed=7, device="cuda:0")
            self.assertEqual(cfg.actor_lr, getattr(config, prefix + "_ACTOR_LR"))
            self.assertEqual(cfg.policy_update_interval, getattr(config, prefix + "_POLICY_UPDATE_INTERVAL"))
            self.assertEqual(cfg.seed, 7)
        self.assertIn("OPT", TrainConfig().checkpoint_dir)


@unittest.skipUnless(torch.cuda.is_available(), "The shared project SAC implementation requires CUDA")
class TrainingTests(unittest.TestCase):
    def run_training(self, cloud):
        torch.set_num_threads(1)
        env = make_env(cloud)
        stages = []
        original_modes = trainer.apply_training_stage_modes
        original_batch_to_tensors = agents.RoutingMASAC._batch_to_tensors
        seen_batches = []

        def modes(stage, routing_masac, host_sac_agents):
            stages.append((stage.value, routing_masac.update_step,
                           {key: value.update_step for key, value in host_sac_agents.items()},
                           {key: copy.deepcopy(value.actor.state_dict()) for key, value in host_sac_agents.items()}))
            original_modes(stage, routing_masac, host_sac_agents)

        def batch_to_tensors(agent, batch):
            seen_batches.append((batch.local_obs.shape[-1], batch.next_local_obs.shape[-1]))
            return original_batch_to_tensors(agent, batch)

        with workspace_temp() as temp:
            cfg = TrainConfig(
                num_episodes=3, host_pretrain_episodes=1, routing_train_episodes=1,
                joint_finetune_episodes=1, checkpoint_dir=str(temp / "checkpoints"),
                episode_log_csv_path=str(temp / "episode.csv"), dc_log_csv_path=str(temp / "dc.csv"),
                old_env_path=None, resume_checkpoint=None, host_init_checkpoint=None,
                routing_batch_size=2, host_batch_size=2, routing_replay_capacity=512,
                host_replay_capacity=512, routing_random_warmup_steps=0, host_random_warmup_steps=0,
                routing_learning_starts=2, host_learning_starts=2, routing_train_every=1,
                host_train_every=1, routing_updates_per_train=1, host_updates_per_train=1,
            )
            route_config = agents.RoutingMASACConfig(actor_hidden_dim=16, critic_hidden_dim=16, device="cuda:0")
            host_config = agents.HostSACConfig(actor_hidden_dim=16, critic_hidden_dim=16, device="cuda:0")
            with patch.object(trainer, "build_environment", return_value=env), \
                 patch.object(trainer, "apply_training_stage_modes", side_effect=modes), \
                 patch.object(agents.RoutingMASAC, "_batch_to_tensors", new=batch_to_tensors), \
                 contextlib.redirect_stdout(io.StringIO()):
                routing, hosts = train(cfg, route_config, host_config)
            self.assertEqual([item[0] for item in stages], ["host_pretrain", "routing_train", "joint_finetune"])
            self.assertEqual(stages[1][1], 0)
            self.assertGreater(sum(stages[1][2].values()), 0)
            self.assertGreater(stages[2][1], 0)
            self.assertEqual(stages[1][2], stages[2][2])
            for dc_id in hosts:
                for key in stages[1][3][dc_id]:
                    self.assertTrue(torch.equal(stages[1][3][dc_id][key], stages[2][3][dc_id][key]))
            self.assertGreater(routing.update_step, stages[2][1])
            self.assertGreater(sum(agent.update_step for agent in hosts.values()), sum(stages[2][2].values()))
            self.assertTrue(any(not torch.equal(stages[2][3][dc_id][key], agent.actor.state_dict()[key])
                                for dc_id, agent in hosts.items() for key in stages[2][3][dc_id]))
            self.assertTrue(seen_batches)
            self.assertEqual(set(seen_batches), {(127, 127)})
            self.assertEqual(routing.actor.fc1.in_features, 132)
            self.assertEqual(routing.action_dim, 6 if cloud else 5)
            final_path = temp / "checkpoints" / "final.pt"
            metadata = json.loads(final_path.with_suffix(".trainer.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["architecture"], OPTRoutingMASAC.checkpoint_architecture)
            self.assertEqual(metadata["structure"]["routing_observation"]["dc_ids"], env.edge_dc_ids + ["cloud"])
            self.assertEqual(metadata["trainer_state"]["next_episode"], 4)
            self.assertEqual(metadata["scheduler_config"]["checkpoint_dir"], cfg.checkpoint_dir)
            with next(temp.glob("episode_*.csv")).open(encoding="utf-8-sig", newline="") as log:
                self.assertEqual(len(list(csv.DictReader(log))), 3)

            loaded, loaded_hosts, loaded_settings = load_models(env, final_path)
            for key, value in routing.actor.state_dict().items():
                self.assertTrue(torch.equal(value, loaded.actor.state_dict()[key]))
            before = {"routing": copy.deepcopy(loaded.actor.state_dict()),
                      **{dc: copy.deepcopy(agent.actor.state_dict()) for dc, agent in loaded_hosts.items()}}
            for deterministic in (True, False):
                with contextlib.redirect_stdout(io.StringIO()):
                    metrics = evaluate_episode(env, loaded, loaded_hosts, loaded_settings, deterministic=deterministic)
                self.assertGreater(metrics["routing_decisions"], 0)
                self.assertEqual(metrics["completed_jobs"] + metrics["dropped_jobs"], 30)
                json.dumps(metrics)
            for name, agent in {"routing": loaded, **loaded_hosts}.items():
                for key, value in agent.actor.state_dict().items():
                    self.assertTrue(torch.equal(value, before[name][key]))
            # Complete resume uses OPT metadata and rejects observation semantic changes.
            with contextlib.redirect_stdout(io.StringIO()):
                resumed = trainer.load_two_layer_checkpoint_if_needed(
                    env, cfg, loaded, loaded_hosts, str(final_path)
                )
            self.assertEqual(resumed[0], 4)
            bad = copy.deepcopy(metadata)
            bad["structure"]["routing_observation"]["dc_ids"].reverse()
            with self.assertRaisesRegex(RuntimeError, "observation schema"):
                trainer.validate_checkpoint_structure_metadata(bad, env, loaded, loaded_hosts)
            bad = copy.deepcopy(metadata)
            bad["architecture"] = trainer.CHECKPOINT_ARCHITECTURE
            with self.assertRaisesRegex(RuntimeError, "architecture"):
                trainer.validate_checkpoint_structure_metadata(bad, env, loaded, loaded_hosts)
            local = agents.RoutingMASAC(127, loaded.global_state_dim, loaded.action_dim, 5, config=route_config)
            with self.assertRaises(RuntimeError):
                local.load(final_path)
            local.save(temp / "local.pt")
            with self.assertRaises(RuntimeError):
                loaded.load(temp / "local.pt")
            # Host-only transfer remains supported independently of Routing weights.
            initialize_hosts(env, loaded_hosts, final_path)
            self.assertTrue(all(agent.update_step == 0 for agent in loaded_hosts.values()))
            if not cloud:
                previous_updates = routing.update_step
                resumed_cfg = replace(cfg, num_episodes=4, joint_finetune_episodes=2,
                                      resume_checkpoint=str(final_path))
                with patch.object(trainer, "build_environment", return_value=env), \
                     contextlib.redirect_stdout(io.StringIO()):
                    continued, _ = train(resumed_cfg, route_config, host_config)
                self.assertGreater(continued.update_step, previous_updates)
                resumed_metadata = json.loads(final_path.with_suffix(".trainer.json").read_text(encoding="utf-8"))
                self.assertEqual(resumed_metadata["trainer_state"]["next_episode"], 5)

    def test_three_stages_checkpoint_and_evaluation_cloud_off(self):
        self.run_training(False)

    def test_three_stages_checkpoint_and_evaluation_cloud_on(self):
        self.run_training(True)

    def test_h_masac_default_training_and_legacy_checkpoint_remain_compatible(self):
        torch.set_num_threads(1)
        env = make_env(jobs=10)
        with workspace_temp() as temp:
            cfg = trainer.TrainConfig(
                num_episodes=1, host_pretrain_episodes=1, routing_train_episodes=0,
                joint_finetune_episodes=0, checkpoint_dir=str(temp / "checkpoints"),
                episode_log_csv_path=str(temp / "episode.csv"), dc_log_csv_path=str(temp / "dc.csv"),
                old_env_path=None, resume_checkpoint=None, host_batch_size=2,
                host_random_warmup_steps=0, host_learning_starts=2, host_train_every=1,
                routing_replay_capacity=64, host_replay_capacity=64,
            )
            route_config = agents.RoutingMASACConfig(actor_hidden_dim=16, critic_hidden_dim=16, device="cuda:0")
            host_config = agents.HostSACConfig(actor_hidden_dim=16, critic_hidden_dim=16, device="cuda:0")
            with patch.object(trainer, "build_environment", return_value=env), \
                 contextlib.redirect_stdout(io.StringIO()):
                routing, hosts = trainer.train(cfg, route_config, host_config)
            self.assertEqual(type(routing), agents.RoutingMASAC)
            self.assertEqual(routing.local_obs_dim, 62)
            self.assertIsNone(routing.observation_metadata)
            path = temp / "checkpoints" / "final.pt"
            metadata = json.loads(path.with_suffix(".trainer.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["architecture"], trainer.CHECKPOINT_ARCHITECTURE)
            self.assertNotIn("routing_observation", metadata["structure"])
            trainer.validate_checkpoint_structure_metadata(metadata, env, routing, hosts)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            checkpoint.pop("observation_metadata", None)
            torch.save(checkpoint, temp / "legacy.pt")
            routing.load(temp / "legacy.pt")
            self.assertEqual(routing.algorithm_role, "routing_masac_ctde")


class CPUTrainingTests(unittest.TestCase):
    def test_opt_cpu_three_stages_save_load_and_evaluate(self):
        torch.set_num_threads(1)
        env = make_env(jobs=30)
        with workspace_temp() as temp:
            cfg = TrainConfig(
                num_episodes=3, host_pretrain_episodes=1, routing_train_episodes=1,
                joint_finetune_episodes=1, checkpoint_dir=str(temp / "checkpoints"),
                episode_log_csv_path=str(temp / "episode.csv"), dc_log_csv_path=str(temp / "dc.csv"),
                routing_batch_size=2, host_batch_size=2, routing_replay_capacity=512,
                host_replay_capacity=512, routing_random_warmup_steps=0,
                host_random_warmup_steps=0, routing_learning_starts=2,
                host_learning_starts=2, routing_train_every=1, host_train_every=1,
                routing_updates_per_train=1, host_updates_per_train=1,
            )
            routing_cfg = agents.RoutingMASACConfig(
                actor_hidden_dim=16, critic_hidden_dim=16, device="cpu"
            )
            host_cfg = agents.HostSACConfig(
                actor_hidden_dim=16, critic_hidden_dim=16, device="cpu"
            )
            with patch.object(trainer, "build_environment", return_value=env), \
                 contextlib.redirect_stdout(io.StringIO()):
                routing, hosts = train(cfg, routing_cfg, host_cfg)
            self.assertEqual(routing.device.type, "cpu")
            self.assertTrue(all(host.device.type == "cpu" for host in hosts.values()))
            self.assertGreater(routing.update_step, 0)
            self.assertGreater(sum(host.update_step for host in hosts.values()), 0)
            self.assertTrue(torch.isfinite(routing.alpha.detach()).item())
            saved = temp / "checkpoints" / "final.pt"
            loaded, loaded_hosts, settings = load_models(env, saved)
            self.assertEqual(loaded.device.type, "cpu")
            with contextlib.redirect_stdout(io.StringIO()):
                result = evaluate_episode(env, loaded, loaded_hosts, settings)
            self.assertEqual(result["completed_jobs"] + result["dropped_jobs"], 30)


if __name__ == "__main__":
    unittest.main()
