"""Independent routing, cross-DC Bellman targets and three-stage integration."""

import contextlib
import copy
import csv
from dataclasses import replace
import importlib
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
from environment.job import Job

agents = importlib.import_module('schedulers.MR-H-MASAC.mr_h_masac_agent')
experience = importlib.import_module('schedulers.MR-H-MASAC.experience')
support = importlib.import_module('schedulers.MR-H-MASAC.training_support')
trainer = importlib.import_module('schedulers.MR-H-MASAC.train_mr_h_masac')
baseline = importlib.import_module('schedulers.H-MASAC.h_masac_agent')


@contextlib.contextmanager
def workspace_temp():
    parent = (ROOT / 'tests').resolve()
    path = parent / f'mr-h-smoke-{uuid.uuid4().hex}'
    path.mkdir()
    try:
        yield path
    finally:
        target = path.resolve()
        if target.parent != parent or not target.name.startswith('mr-h-smoke-'):
            raise RuntimeError('Unsafe test cleanup path')
        shutil.rmtree(target)


def routing_config(device='cpu'):
    return agents.RoutingMASACConfig(actor_hidden_dim=16, critic_hidden_dim=16,
                                    device=device, allow_cpu=device == 'cpu')


def manager(device='cpu', dc_ids=('A', 'B', 'C')):
    return agents.MRRoutingMASAC(4, 6, 3, len(dc_ids), routing_config(device),
                                edge_dc_ids=dc_ids, action_target_dc_ids=('A', 'B', 'C'))


def transition(index=0, next_index=-1, forced=False):
    ids = ('A', 'B', 'C')
    done = next_index < 0
    return experience.RoutingTransition(
        job_id='job', agent_id=ids[index], agent_index=index, env_time=0.0,
        local_obs=np.ones(4, dtype=np.float32), global_state=np.ones(6, dtype=np.float32),
        action=index, action_type='self' if done else 'edge_dc',
        action_source='forced' if forced else 'policy', reward=1.0,
        next_agent_id=None if done else ids[next_index], next_agent_index=next_index,
        next_env_time=1.0, next_local_obs=np.ones(4, dtype=np.float32),
        next_global_state=np.ones(6, dtype=np.float32),
        terminated=done, truncated=False, done=done, terminal_reason='completed' if done else None,
    )


def make_env(cloud=False, jobs=30):
    dcs = []
    graph = nx.Graph()
    for index in range(5):
        dc = DataCenter(f'DC-{index + 1}', 0.5)
        dc.host_list = [Host(f'host-{index}-{host}', 4, 16) for host in range(2)]
        dcs.append(dc)
    cloud_dc = DataCenter('cloud', 0.0)
    cloud_dc.host_list = [Host('cloud-host', 999999, 999999)]
    dcs.append(cloud_dc)
    for dc in dcs:
        graph.add_node(dc.dc_id, dc_instance=dc)
    for index, dc in enumerate(dcs):
        for target in dcs[index + 1:]:
            graph.add_edge(dc.dc_id, target.dc_id, weight=0.5 if target.dc_id == 'cloud' else 0.25)
    source = SimpleNamespace(global_dc_list=dcs, datacenter_graph=graph, job_num=jobs)
    with patch.object(config, 'ENABLE_CLOUD_ACTION', cloud):
        env = CloudEdgeEnv(env_source=source, seed=42)

    def workload():
        result = []
        counts = {dc: 0 for dc in env.edge_dc_ids}
        for index in range(jobs):
            dc = env.edge_dc_ids[index % len(env.edge_dc_ids)]
            job = Job(f'job-{index}', 2, 1, 3, dc, dc)
            job.set_arrive_time(index * 0.5)
            result.append(job)
            counts[dc] += 1
        env.episode_exogenous_arrival_counts = counts
        return result

    env._generate_episode_workload = workload
    return env


from host_test_support import make_host_bundle

class IndependentRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_networks_optimizers_and_seeds_are_independent(self):
        routing = manager()
        repeated = manager()
        pointers = []
        for dc, agent in routing.agents.items():
            self.assertEqual(agent.actor.fc1.in_features, 4)
            self.assertEqual(agent.critic.q1.fc1.in_features, 6)
            for network in (agent.actor, agent.critic, agent.target_critic):
                pointers.extend(p.data_ptr() for p in network.parameters())
            pointers.append(agent.log_alpha.data_ptr())
            for key, value in agent.actor.state_dict().items():
                self.assertTrue(torch.equal(value, repeated.agents[dc].actor.state_dict()[key]))
            self.assertFalse(any(p.requires_grad for p in agent.target_critic.parameters()))
        self.assertEqual(len(pointers), len(set(pointers)))
        self.assertFalse(torch.equal(routing.agents['A'].actor.fc1.weight,
                                     routing.agents['B'].actor.fc1.weight))
        self.assertEqual(len({id(a.actor_optimizer) for a in routing.agents.values()}), 3)
        for index in range(3):
            action = routing.select_action(np.ones(4), index, deterministic=True)
            expected = routing.agent_for_index(index).actor(torch.ones(1, 4)).argmax(-1).item()
            self.assertEqual(action, expected)
        with self.assertRaises(ValueError):
            routing.select_action(np.ones(4), -1)

    def test_targets_use_successor_policy_critic_and_temperature(self):
        routing = manager()
        with torch.no_grad():
            for index, agent in enumerate(routing.agents.values()):
                for parameter in agent.actor.parameters():
                    parameter.zero_()
                for parameter in agent.target_critic.parameters():
                    parameter.zero_()
                agent.actor.output_layer.bias.copy_(torch.tensor([index * 2.0, 0.0, -1.0]))
                agent.target_critic.q1.output_layer.bias.fill_(10 * index + 2)
                agent.target_critic.q2.output_layer.bias.fill_(10 * index + 3)
                agent.log_alpha.fill_(np.log(0.2 + index * 0.3))
        replay = experience.RoutingReplayBuffer(8, 4, 6)
        replay.add(transition(0, 1))
        replay.add(transition(0, 2))
        replay.add(transition(0))
        batch = routing.agents['A']._batch_to_tensors(replay.sample(3))
        values = routing.next_soft_values(batch)
        for row, successor_index in enumerate(batch.next_agent_indices.tolist()):
            if successor_index < 0:
                self.assertEqual(values[row].item(), 0)
            else:
                agent = routing.agent_for_index(successor_index)
                probs = torch.softmax(torch.tensor([successor_index * 2.0, 0.0, -1.0]), -1)
                entropy = -(probs * probs.log()).sum()
                expected = 10 * successor_index + 2 + (0.2 + successor_index * 0.3) * entropy
                self.assertAlmostEqual(values[row].item(), expected.item(), places=5)
        self.assertFalse(values.requires_grad)
        info = routing.agents['A']._update_critic(batch, routing.next_soft_values)
        expected_target = batch.rewards + routing.config.gamma * (1 - batch.done) * values
        self.assertAlmostEqual(info['mean_target_q'].item(), expected_target.mean().item(), places=5)
        with patch.object(routing, 'agent_for_index', side_effect=AssertionError('Terminal queried a network')):
            terminal = replace(batch, done=torch.ones_like(batch.done),
                               next_agent_indices=torch.full_like(batch.next_agent_indices, -1))
            self.assertTrue(torch.equal(routing.next_soft_values(terminal), torch.zeros(3)))
        with self.assertRaises(ValueError):
            routing.next_soft_values(replace(batch, done=torch.zeros_like(batch.done),
                                             next_agent_indices=torch.full_like(batch.next_agent_indices, -1)))

    def test_update_isolated_to_current_dc_and_delayed_replay_is_not_missed(self):
        routing = manager()
        replay = experience.RoutingReplayBuffers(24, 4, 6, ('A', 'B', 'C'), seed=7)
        schedule = SimpleNamespace(routing_batch_size=2, routing_learning_starts=2,
                                   routing_train_every=2, routing_updates_per_train=1)
        routing.action_steps['A'] = 3  # Replay becomes available after interval boundary.
        self.assertEqual(routing.update_ready(replay, schedule), [])
        replay.add(transition(0, 1))
        replay.add(transition(0))
        before = {dc: copy.deepcopy(agent.checkpoint_payload()) for dc, agent in routing.agents.items()}
        infos = routing.update_ready(replay, schedule)
        self.assertEqual([dc for dc, _ in infos], ['A'])
        self.assertEqual(routing.agents['A'].update_step, 1)
        self.assertEqual(routing.last_train_action_steps['A'], 2)
        self.assertTrue(any(not torch.equal(value, routing.agents['A'].actor.state_dict()[name])
                            for name, value in before['A']['actor_state_dict'].items()))
        for dc in ('B', 'C'):
            agent = routing.agents[dc]
            for field, network in (('actor_state_dict', agent.actor), ('critic_state_dict', agent.critic),
                                   ('target_critic_state_dict', agent.target_critic)):
                for name, value in before[dc][field].items():
                    self.assertTrue(torch.equal(value, network.state_dict()[name]))
            self.assertTrue(torch.equal(before[dc]['log_alpha'], agent.log_alpha))
            self.assertEqual(agent.actor_optimizer.state, {})
            self.assertTrue(all(p.grad is None for p in agent.actor.parameters()))
        self.assertEqual(routing.update_ready(replay, schedule), [])

    def test_replay_budget_partition_identity_and_forced_exclusion(self):
        replay = experience.RoutingReplayBuffers(10, 4, 6, ('A', 'B', 'C'))
        self.assertEqual([b.capacity for b in replay.buffers.values()], [4, 3, 3])
        replay.add(transition(0, 1))
        replay.add(transition(1))
        self.assertEqual([len(b) for b in replay.buffers.values()], [1, 1, 0])
        self.assertEqual(replay.buffers['A'].next_agent_indices[0], 1)
        with self.assertRaises(ValueError):
            replay.add(replace(transition(), agent_id='B'))
        with self.assertRaises(ValueError):
            replay.add(replace(transition(0, 1), next_agent_id='C'))
        with self.assertRaises(RuntimeError):
            replay.add(transition(forced=True))
        overridden = experience.RoutingReplayBuffers(10, 4, 6, ('A', 'B', 'C'), capacity_per_dc=5)
        self.assertEqual(overridden.capacity, 15)

    def test_checkpoint_roundtrip_and_failures_leave_live_networks_unchanged(self):
        routing = manager()
        replay = experience.RoutingReplayBuffers(24, 4, 6, ('A', 'B', 'C'))
        for _ in range(2):
            replay.add(transition(0, 1))
        routing.action_steps['A'] = 2
        routing.update_ready(replay, SimpleNamespace(routing_batch_size=2, routing_learning_starts=2,
                             routing_train_every=1, routing_updates_per_train=1))
        with workspace_temp() as temp:
            path = temp / 'model.pt'
            routing.save(path)
            loaded = manager()
            loaded.load(path)
            self.assertEqual(loaded.action_steps, routing.action_steps)
            self.assertEqual(loaded.last_train_action_steps, routing.last_train_action_steps)
            for dc, agent in routing.agents.items():
                restored = loaded.agents[dc]
                self.assertEqual(restored.update_step, agent.update_step)
                for network in ('actor', 'critic', 'target_critic'):
                    for name, value in getattr(agent, network).state_dict().items():
                        self.assertTrue(torch.equal(value, getattr(restored, network).state_dict()[name]))
                self.assertTrue(torch.equal(agent.log_alpha, restored.log_alpha))
                for optimizer_name in ('actor_optimizer', 'critic_optimizer', 'alpha_optimizer'):
                    original_optimizer = getattr(agent, optimizer_name)
                    restored_optimizer = getattr(restored, optimizer_name)
                    for group, restored_group in zip(original_optimizer.param_groups, restored_optimizer.param_groups):
                        for parameter, restored_parameter in zip(group['params'], restored_group['params']):
                            original_state = original_optimizer.state.get(parameter, {})
                            restored_state = restored_optimizer.state.get(restored_parameter, {})
                            self.assertEqual(set(original_state), set(restored_state))
                            for name, value in original_state.items():
                                self.assertTrue(torch.equal(value, restored_state[name]))
            saved = torch.load(path, map_location='cpu', weights_only=False)
            for mutate in (
                lambda c: c['edge_dc_ids'].reverse(),
                lambda c: c['agents'].pop('B'),
                lambda c: c['agents']['C']['actor_state_dict'].pop('fc1.weight'),
                lambda c: c['action_steps'].update(A=-1),
                lambda c: c.update(algorithm_role='routing_masac_ctde'),
            ):
                bad = copy.deepcopy(saved)
                mutate(bad)
                torch.save(bad, temp / 'bad.pt')
                before = {dc: copy.deepcopy(a.actor.state_dict()) for dc, a in loaded.agents.items()}
                with self.assertRaises(RuntimeError):
                    loaded.load(temp / 'bad.pt')
                for dc, agent in loaded.agents.items():
                    for name, value in agent.actor.state_dict().items():
                        self.assertTrue(torch.equal(value, before[dc][name]))
            shared = baseline.RoutingMASAC(4, 6, 3, 3, routing_config())
            shared.save(temp / 'shared.pt')
            with self.assertRaises(RuntimeError):
                loaded.load(temp / 'shared.pt')
            with self.assertRaises(RuntimeError):
                shared.load(path)

    def test_metrics_are_weighted_and_keep_per_dc_values(self):
        stats = support.EpisodeStatistics(1, 42, 'routing_train', {})
        stats.record_routing_update({'critic_loss': 1.0}, 'A', sample_count=2)
        stats.record_routing_update({'critic_loss': 4.0}, 'B', sample_count=4)
        self.assertEqual(stats.mean_routing_metric('critic_loss'), 3.0)
        self.assertEqual(stats.mean_dc_routing_metric('A', 'critic_loss'), 1.0)
        self.assertEqual(stats.mean_dc_routing_metric('B', 'critic_loss'), 4.0)


class TrainingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def run_frozen_host_training(self, cloud, device='cpu'):
        env = make_env(cloud)
        with workspace_temp() as temp:
            cfg = support.TrainConfig(
                host_checkpoint=str(make_host_bundle(env, temp / "host", device)),
                num_episodes=3, host_pretrain_episodes=0, routing_train_episodes=3,
                joint_finetune_episodes=0, checkpoint_dir=str(temp / 'checkpoints'),
                episode_log_csv_path=str(temp / 'episode.csv'), dc_log_csv_path=str(temp / 'dc.csv'),
                routing_batch_size=2, host_batch_size=2, routing_replay_capacity=160,
                host_replay_capacity=160, routing_random_warmup_steps=1,
                host_random_warmup_steps=0, routing_learning_starts=2,
                host_learning_starts=2, routing_train_every=1, host_train_every=1,
                routing_updates_per_train=1, host_updates_per_train=1,
            )
            route_cfg = routing_config(device)
            host_cfg = agents.HostSACConfig(actor_hidden_dim=16, critic_hidden_dim=16,
                                           device=device, allow_cpu=device == 'cpu')
            stages = []
            apply_modes = trainer.apply_training_stage_modes

            def capture(stage, routing_masac, host_sac_agents):
                stages.append((stage, {dc: copy.deepcopy(a.actor.state_dict()) for dc, a in routing_masac.agents.items()},
                               {dc: copy.deepcopy(a.actor.state_dict()) for dc, a in host_sac_agents.items()}))
                apply_modes(stage, routing_masac, host_sac_agents)

            with patch.object(trainer, 'build_environment', return_value=env), \
                 patch.object(trainer, 'apply_training_stage_modes', side_effect=capture), \
                 contextlib.redirect_stdout(io.StringIO()):
                routing, hosts = trainer.train(cfg, route_cfg, host_cfg)
            self.assertEqual([s[0].value for s in stages], ['routing_train'] * 3)
            for stage in stages:
                for dc in env.edge_dc_ids:
                    for name, value in stage[2][dc].items():
                        self.assertTrue(torch.equal(value, hosts[dc].actor.state_dict()[name]))
            self.assertTrue(any(not torch.equal(value, routing.agents[dc].actor.state_dict()[name])
                                for dc in env.edge_dc_ids for name, value in stages[0][1][dc].items()))
            self.assertGreater(routing.update_step, 0)
            self.assertEqual(sum(h.update_step for h in hosts.values()), 0)
            self.assertEqual(routing.action_dim, 6 if cloud else 5)
            self.assertTrue(all(a.device.type == device.split(':')[0] for a in routing.agents.values()))
            final_path = temp / 'checkpoints' / 'final.pt'
            self.assertFalse((temp / 'checkpoints' / 'joint_finetune_start.pt').exists())
            metadata = json.loads(final_path.with_suffix('.trainer.json').read_text(encoding='utf-8'))
            self.assertEqual(metadata['architecture'], routing.checkpoint_architecture)
            self.assertEqual(metadata['trainer_state']['routing_action_steps_by_dc'], routing.action_steps)
            self.assertEqual(metadata['trainer_state']['next_episode'], 4)
            with next(temp.glob('episode_*.csv')).open(encoding='utf-8-sig', newline='') as log:
                episodes = list(csv.DictReader(log))
            self.assertEqual(len(episodes), 3)
            self.assertEqual(int(episodes[-1]['routing_replay_size']), int(episodes[-1]['routing_replay_trainable_size']))
            self.assertEqual(float(episodes[-1]['routing_alpha_min']), min(float(a.alpha.detach()) for a in routing.agents.values()))
            with next(temp.glob('dc_*.csv')).open(encoding='utf-8-sig', newline='') as log:
                rows = list(csv.DictReader(log))
            self.assertEqual(len(rows), 15)
            self.assertEqual(sum(int(r['routing_replay_capacity']) for r in rows[-5:]), 160)
            for row in rows[-5:]:
                self.assertEqual(int(row['routing_update_step']), routing.agents[row['dc_id']].update_step)
            loaded = agents.MRRoutingMASAC(routing.local_obs_dim, routing.global_state_dim,
                routing.action_dim, routing.num_agents, route_cfg,
                edge_dc_ids=env.possible_agents, action_target_dc_ids=env.routing_action_target_dc_ids)
            loaded.observation_metadata = routing.observation_metadata
            with contextlib.redirect_stdout(io.StringIO()):
                restored = support.load_two_layer_checkpoint_if_needed(env, cfg, loaded, hosts, str(final_path))
            self.assertEqual(restored[0], 4)
            self.assertEqual(loaded.action_steps, routing.action_steps)
            if device == 'cpu' and not cloud:
                continued_cfg = replace(cfg, num_episodes=4, routing_train_episodes=4,
                                        resume_checkpoint=str(final_path))
                with patch.object(trainer, 'build_environment', return_value=env), \
                     contextlib.redirect_stdout(io.StringIO()):
                    continued, _ = trainer.train(continued_cfg, route_cfg, host_cfg)
                self.assertGreater(continued.update_step, routing.update_step)
                self.assertTrue(all(continued.action_steps[dc] > routing.action_steps[dc] for dc in env.edge_dc_ids))

    def test_frozen_host_cloud_off_and_resume(self):
        self.run_frozen_host_training(False)

    def test_frozen_host_cloud_on(self):
        self.run_frozen_host_training(True)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_gpu_frozen_host(self):
        self.run_frozen_host_training(True, 'cuda:0')


if __name__ == '__main__':
    unittest.main()
