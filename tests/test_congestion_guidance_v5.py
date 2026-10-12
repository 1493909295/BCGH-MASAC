"""Regression tests for causal congestion, lifecycle and composite SAC policy."""
import importlib
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import pytest
import torch

from schedulers.congestion_guidance import TransferLifecycle, guided_distribution


@pytest.fixture(params=['BCGH-MASAC', 'MR-BCGH-MASAC'])
def modules(request):
    prefix = 'schedulers.' + request.param
    return NS(name=request.param, **{name: importlib.import_module(prefix+'.'+filename)
        for name, filename in [('game', 'bayesian_congestion_game'), ('policy', 'guided_policy'),
                               ('experience', 'experience'), ('agent', 'h_masac_agent')]})


def make_game(m, **kwargs):
    return m.game.BayesianCongestionGame(NS(player_ids=('A', 'B', 'C')),
        capacities={dc: m.game.StaticDCCapacity(100, 10, ((100, 10),)) for dc in ('A', 'B', 'C')},
        window_s=10, **kwargs)


def test_active_pressure_survives_window_and_releases_once(modules):
    game = make_game(modules)
    store = game.pressure_store
    args = dict(event_id='j:0', event_time=0, cpu_request=40, gpu_request=4)
    store.record_selection('B', **args)
    initial = store.get_congestion_cost('B', 10, 1)
    store.advance_time(11)
    assert 0 < store.get_pressure('B')[0] < .4
    row = store.snapshot()['B']
    assert row['active_commitment_count'] == 1
    assert row['recent_selection_count'] == 0
    assert store.release('j:0', 11)
    assert not store.release('j:0', 11)
    store.record_selection('B', **args)  # Never resurrect a released commitment.
    assert store.get_pressure('B') == (0, 0)
    assert store.get_congestion_cost('B', 10, 1) < initial


def test_known_start_uses_remaining_work_and_background(modules):
    store = make_game(modules, pressure_background_cpu=.1).pressure_store
    store.record_selection('B', event_id='j:0', event_time=0, cpu_request=40, gpu_request=0)
    store.set_service_hint('j:0', 30.)
    store.mark_started('j:0', 1., 'B')
    store.advance_time(16.)
    assert store.get_pressure('B')[0] == pytest.approx(.1 + .4*15/300)
    store.advance_time(31.)
    assert store.get_pressure('B')[0] == pytest.approx(.1)
    assert store.get_congestion_cost('B', 10, service_s=600) > store.get_congestion_cost('B', 10, service_s=300)


def test_local_commitments_do_not_reveal_remote_local_arrivals(modules):
    game = make_game(modules, resource_penalty_enabled=True)
    store = game.pressure_store
    store.record_selection('B', event_id='local:0', event_time=0, cpu_request=80, gpu_request=0)
    store.mark_local('local:0')
    actions = {dc: modules.game.RoutingActionFeatures(dc, cpu_request=10) for dc in ('A', 'B', 'C')}
    assert game.evaluate_actions('A', actions)[1].cpu_pressure == 0
    assert game.evaluate_actions('B', actions)[1].cpu_pressure == pytest.approx(.8)
    assert store.snapshot()['B']['recent_selection_count'] == 0
    assert store.snapshot()['B']['historical_cpu_pressure'] == 0


@pytest.mark.parametrize('started,success,sla,terminal', [
    (None, False, False, 'dropped_before_start'),
    (4., True, True, 'completed_on_time'),
    (4., True, False, 'completed_late'),
    (4., False, False, 'dropped_after_start'),
])
def test_queue_lifecycle_pending_conservation_and_local_wait(started, success, sla, terminal):
    store = TransferLifecycle()
    store.register(NS(event_id='j:0', job_id='j', source_dc_id='A', target_dc_id='B', routed_at=0))
    assert store.successor('j:0', 'queued', 2)
    assert not store.successor('j:0', 'started', 3)
    assert store.metrics()['local_pending_count'] == 1
    assert store.metrics()['effective_local_completion_rate'] is None
    if started is not None:
        store.start('j', 'B', started)
        assert store.metrics()['queue_started_count'] == 1
        assert store.metrics()['local_queue_wait_mean_s'] == 2  # Excludes transfer [0, 2].
    trace = NS(job_id='j', host_step=NS(dc_id='B'), terminal_time=8,
               terminal_reason='completed' if success else 'waiting_timeout')
    store.close(trace, NS(start_time=started), sla)
    store.close(trace, NS(start_time=started), sla)
    assert store.records['j:0']['terminal'] == terminal
    metrics = store.metrics()
    assert metrics['local_accepted_count'] == metrics['local_terminal_count'] + metrics['local_pending_count']
    assert metrics['successor_started_count'] == 0
    assert metrics['successor_queued_count'] == 1
    assert metrics['effective_local_completion_rate'] == float(sla and success)


def test_reforward_releases_previous_target_and_preserves_success(modules):
    game = make_game(modules, transit_penalty_enabled=True)
    runtime = modules.game.ShortWindowRuntime(game)
    job = NS(job_id='j', cpu_request=20, gpu_request=0, arrive_time=0, duration=10,
             start_time=3., get_turnaround_time=lambda: 5.)
    for source, target, time in [('A', 'B', 0.), ('B', 'C', 1.)]:
        runtime.record_routing(NS(job_id='j', agent_id=source, env_time=time), 'edge_dc', target, 'policy', job)
    assert game.get_pressure('B') == (0, 0)
    assert game.get_pressure('C')[0] > 0
    runtime.record_routing(NS(job_id='j', agent_id='C', env_time=2.), 'self', 'C', 'policy', job)
    # Host API may advance the simulation before returning its initial result.
    runtime.record_host_result('j', 'C', 'queued', 4.)
    runtime.lifecycle.start('j', 'C', 3.)
    assert runtime.lifecycle.metrics('C')['local_queue_wait_mean_s'] == 1.
    trace = NS(job_id='j', terminal_reason='completed', terminal_time=5.,
        host_step=NS(dc_id='C', execution_result='queued'), routing_steps=tuple(
            NS(sequence_index=i, source_dc_id=s, target_dc_id=t, env_time=float(i), action_type=a)
            for i, (s, t, a) in enumerate([('A', 'B', 'edge_dc'), ('B', 'C', 'edge_dc'), ('C', 'C', 'self')])))
    env = NS(job_map={'j': job}, current_time=5., sla_deadline_ratio=2, max_job_duration=10, drop_deadline_ratio=3)
    outcomes = game.update_guidance_outcomes(trace, env)
    assert outcomes[0].success and not outcomes[0].completed_at_target
    assert outcomes[1].completed_at_target
    assert outcomes[1].event_id in runtime.lifecycle.records
    assert outcomes[1].local_queue_wait_s == 1.
    assert runtime.lifecycle.metrics('B')['local_accepted_count'] == 0
    assert runtime.lifecycle.metrics('C')['local_completed_on_time_count'] == 1
    assert game.get_pressure('C') == (0, 0)
    context = {dc: modules.game.RoutingActionFeatures(dc) for dc in ('A', 'B', 'C')}
    rows = game.evaluate_actions('A', context)
    assert rows[1].applied_transit_penalty == 0
    assert game.evaluate_actions('A', {**context, 'B': replace(context['B'], cycle_risk=1)})[1].applied_transit_penalty > 0


def test_resource_class_fallback_and_admission_quality(modules):
    store = modules.policy.TerminalOutcomeStore(100, 2)
    for i in range(8):
        item = modules.policy.GuidanceOutcome(str(i), str(i), 'A', 'B', 0, 1,
            i < 4, i < 4, 1. if i < 4 else None,
            'cpu:low' if i < 4 else 'gpu:high', i < 4, i < 4)
        store.record(item, now=1)
    cpu = store.feedback('A', 'B', completion_time_scale=10, resource_class='cpu:low')
    gpu = store.feedback('A', 'B', completion_time_scale=10, resource_class='gpu:high')
    fallback = store.feedback('A', 'B', completion_time_scale=10, resource_class='cpu:medium')
    admitted = store.feedback('A', 'B', completion_time_scale=10, admitted_only=True)
    assert cpu['success_score'] > fallback['success_score'] > gpu['success_score']
    assert fallback['resource_class_mix'] == 0
    assert admitted['outcome_samples'] == 4 and admitted['success_score'] == 1


def make_transition(m, *, done=False):
    return m.experience.RoutingTransition(job_id='j', agent_id='A', agent_index=0, env_time=0,
        local_obs=np.ones(3, np.float32), global_state=np.ones(4, np.float32),
        action=1, action_type='edge_dc', action_source='policy', reward=1,
        next_agent_id=None if done else 'B', next_agent_index=-1 if done else 1,
        next_env_time=1, next_local_obs=np.ones(3, np.float32), next_global_state=np.ones(4, np.float32),
        terminated=done, truncated=False, done=done, terminal_reason='completed' if done else None,
        guidance_offsets=np.array([.2, -.2], np.float32), guidance_mask=np.array([True, True]),
        next_guidance_offsets=None if done else np.array([-.3, .3], np.float32),
        next_guidance_mask=None if done else np.array([False, True]))


def test_execution_replay_actor_alpha_and_target_share_distribution(modules, monkeypatch):
    torch.set_num_threads(1)
    agentmod = modules.agent
    monkeypatch.setattr(agentmod, 'resolve_training_device', lambda _: torch.device('cpu'))
    config = agentmod.RoutingMASACConfig(device='cpu', actor_hidden_dim=8, critic_hidden_dim=8)
    if modules.name == 'BCGH-MASAC':
        system = agentmod.RoutingMASAC(3, 4, 2, 2, config)
        agent = system
    else:
        system = agentmod.RoutingMASAC(3, 4, 2, 2, ['A', 'B'], ['A', 'B'], config)
        agent = system.agent_for_index(0)
    policy = modules.policy.GuidedRoutingPolicy(system, ['A', 'B'], enabled=True)
    policy.select_action(np.ones(3, np.float32), 0, 'A', action_bias={0: .4, 1: -.4}, guidance_lambda=.5)
    offsets, mask = policy.last_guidance_snapshot
    transition = replace(make_transition(modules), guidance_offsets=offsets, guidance_mask=mask)
    replay = modules.experience.RoutingReplayBuffer(4, 3, 4, seed=0)
    replay.add(transition)
    offsets[:] = 99  # Replay must have captured a copy.
    batch = agent._batch_to_tensors(replay.sample(1))
    assert np.allclose(batch.guidance_offsets.numpy(), [[.2, -.2]])
    probs, _, _ = agent.actor.get_policy(batch.local_obs, batch.agent_indices,
        guidance_offsets=batch.guidance_offsets, guidance_mask=batch.guidance_mask)
    assert np.allclose(probs.detach().numpy()[0], policy.last_probabilities['guided_probabilities'])
    with patch.object(agent.actor, 'get_policy', wraps=agent.actor.get_policy) as call:
        agent._update_actor(batch)
        agent._update_alpha(batch)
        assert len(call.call_args_list) == 2
        assert all(c.kwargs['guidance_offsets'] is batch.guidance_offsets for c in call.call_args_list)
    target = agent if modules.name == 'BCGH-MASAC' else system.agent_for_index(1)
    with patch.object(target.actor, 'get_policy', wraps=target.actor.get_policy) as call:
        if modules.name == 'BCGH-MASAC':
            agent._update_critic(batch)
        else:
            system.next_soft_values(batch)
        assert call.call_args.kwargs['guidance_offsets'].shape == (1, 2)
        assert not call.call_args.kwargs['guidance_mask'][0, 0]


def test_masked_distribution_has_finite_gradients():
    logits = torch.zeros((1, 3), requires_grad=True)
    probs, logs, _ = guided_distribution(logits, torch.tensor([[.2, .3, .4]]), torch.tensor([[True, False, True]]))
    assert probs[0, 1] == 0
    (probs * logs).sum().backward()
    assert torch.isfinite(logits.grad).all()
    with pytest.raises(ValueError):
        guided_distribution(logits, mask=torch.zeros((1, 3), dtype=torch.bool))
