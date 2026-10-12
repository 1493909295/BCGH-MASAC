"""Semantic checks for permanently guided, context-conditioned routing SAC."""
import copy
import importlib
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import pytest
import torch

from schedulers.congestion_guidance import GuidanceConditioner


@pytest.fixture(params=['BCGH-MASAC', 'MR-BCGH-MASAC'])
def setup(request, monkeypatch):
    prefix = 'schedulers.' + request.param
    agent = importlib.import_module(prefix + '.h_masac_agent')
    experience = importlib.import_module(prefix + '.experience')
    policy = importlib.import_module(prefix + '.guided_policy')
    monkeypatch.setattr(agent, 'resolve_training_device', lambda _: torch.device('cpu'))
    torch.set_num_threads(1)
    config = agent.RoutingMASACConfig(device='cpu', actor_hidden_dim=8, critic_hidden_dim=8, seed=7)
    args = {} if request.param == 'BCGH-MASAC' else dict(edge_dc_ids=['A', 'B'], action_target_dc_ids=['A', 'B'])
    system = agent.RoutingMASAC(3, 4, 2, 2, config=config, **args)
    local = system if request.param == 'BCGH-MASAC' else system.agent_for_index(0)
    return NS(name=request.param, system=system, local=local, experience=experience, policy=policy)


def transition(s, *, done=False):
    return s.experience.RoutingTransition(job_id='j', agent_id='A', agent_index=0, env_time=0,
        local_obs=np.ones(3, np.float32), global_state=np.ones(4, np.float32),
        action=1, action_type='edge_dc', action_source='policy', reward=2,
        next_agent_id=None if done else 'B', next_agent_index=-1 if done else 1,
        next_env_time=1, next_local_obs=np.ones(3, np.float32), next_global_state=np.ones(4, np.float32),
        terminated=done, truncated=False, done=done, terminal_reason='completed' if done else None,
        guidance_offsets=np.array([.2, -.2], np.float32), guidance_mask=np.array([True, True]),
        next_guidance_offsets=None if done else np.array([-.3, .3], np.float32),
        next_guidance_mask=None if done else np.array([False, True]))


def tensors(s, item):
    replay = s.experience.RoutingReplayBuffer(4, 3, 4, require_guidance=True)
    replay.add(item)
    return s.local._batch_to_tensors(replay.sample(1))


def test_actor_and_critic_learn_context_without_backpropagating_into_guidance(setup):
    s = setup
    batch = tensors(s, transition(s, done=True))
    batch.guidance_offsets.requires_grad_(True)
    before = s.local.actor.guidance_context.projection.weight.detach().clone()
    s.local._update_actor(batch)
    assert batch.guidance_offsets.grad is None
    assert s.local.actor.guidance_context.projection.weight.grad.abs().sum() > 0
    assert not torch.equal(before, s.local.actor.guidance_context.projection.weight)
    if s.name == 'BCGH-MASAC':
        s.local._update_critic(batch)
    else:
        s.local._update_critic(batch, s.system.next_soft_values)
    assert batch.guidance_offsets.grad is None
    assert s.local.critic.q1.guidance_context.projection.weight.grad.abs().sum() > 0
    with torch.no_grad():
        kw = dict(agent_indices=batch.agent_indices, guidance_mask=batch.guidance_mask)
        actor_a = s.local.actor(batch.local_obs, guidance_offsets=batch.guidance_offsets, **kw)
        actor_b = s.local.actor(batch.local_obs, guidance_offsets=-batch.guidance_offsets, **kw)
        q_a = s.local.critic(batch.global_states, guidance_offsets=batch.guidance_offsets, **kw)[0]
        q_b = s.local.critic(batch.global_states, guidance_offsets=-batch.guidance_offsets, **kw)[0]
    assert not torch.equal(actor_a, actor_b)
    assert not torch.equal(q_a, q_b)


def test_execution_matches_replay_after_context_weights_learn(setup):
    s = setup
    s.local._update_actor(tensors(s, transition(s, done=True)))
    policy = s.policy.GuidedRoutingPolicy(s.system, ['A', 'B'], enabled=True)
    policy.select_action(np.ones(3, np.float32), 0, 'A', action_bias={0: .4, 1: -.4}, guidance_lambda=.5)
    offsets, mask = policy.last_guidance_snapshot
    batch = tensors(s, replace(transition(s), guidance_offsets=offsets, guidance_mask=mask))
    probabilities = s.local.actor.get_policy(batch.local_obs, batch.agent_indices,
        guidance_offsets=batch.guidance_offsets, guidance_mask=batch.guidance_mask)[0]
    np.testing.assert_allclose(probabilities.detach().numpy()[0], policy.last_probabilities['guided_probabilities'])
    audits = policy.diagnostics()
    assert audits['guidance_mean_snapshot_complete'] == 1
    assert audits['guidance_mean_effective_bias_span'] == pytest.approx(.4)
    assert audits['guidance_mean_logit_span'] >= 0


def test_all_critic_paths_receive_the_matching_snapshot(setup):
    s = setup
    batch = tensors(s, transition(s))
    successor = s.local if s.name == 'BCGH-MASAC' else s.system.agent_for_index(1)
    with patch.object(s.local.critic, 'forward', wraps=s.local.critic.forward) as current:
        s.local._update_actor(batch)
        assert current.call_args.kwargs['guidance_offsets'] is batch.guidance_offsets
    with patch.object(successor.target_critic, 'forward', wraps=successor.target_critic.forward) as target:
        if s.name == 'BCGH-MASAC':
            s.local._update_critic(batch)
        else:
            s.local._update_critic(batch, s.system.next_soft_values)
        torch.testing.assert_close(target.call_args.kwargs['guidance_offsets'], batch.next_guidance_offsets)
        torch.testing.assert_close(target.call_args.kwargs['guidance_mask'], batch.next_guidance_mask)


def test_replay_rejects_missing_mixed_or_old_context(setup):
    s = setup
    item = transition(s)
    plain = replace(item, guidance_offsets=None, guidance_mask=None,
                    next_guidance_offsets=None, next_guidance_mask=None)
    required = s.experience.RoutingReplayBuffer(4, 3, 4, require_guidance=True)
    for bad in (plain, replace(item, guidance_mask=None), replace(item, next_guidance_offsets=None),
                replace(item, guidance_context_version=0)):
        with pytest.raises(ValueError):
            required.add(bad)
        assert len(required) == 0
    for first, second in ((item, plain), (plain, item)):
        replay = s.experience.RoutingReplayBuffer(4, 3, 4)
        replay.add(first)
        with pytest.raises(ValueError, match='mix'):
            replay.add(second)
        assert len(replay) == 1
        replay.clear()
        replay.add(second)
        assert (replay.sample(1).guidance_offsets is None) == (second is plain)


def test_zero_lambda_has_explicit_neutral_replay_and_same_action(setup):
    s = setup
    policy = s.policy.GuidedRoutingPolicy(s.system, ['A', 'B'], enabled=True)
    torch.manual_seed(19)
    expected = s.system.select_action(np.ones(3, np.float32), 0)
    torch.manual_seed(19)
    actual = policy.select_action(np.ones(3, np.float32), 0, 'A', guidance_lambda=0)
    assert actual == expected
    offsets, mask = policy.last_guidance_snapshot
    np.testing.assert_array_equal(offsets, [0, 0])
    assert mask.all()
    warmup_offsets, warmup_mask = policy.snapshot_context('A', None, 0)
    np.testing.assert_array_equal(warmup_offsets, offsets)
    np.testing.assert_array_equal(warmup_mask, mask)
    tensors(s, replace(transition(s, done=True), guidance_offsets=offsets, guidance_mask=mask))


def test_finalize_preserves_actual_same_job_successor_context(setup):
    s = setup
    store = s.experience.PendingJobTraceStore()
    obs, state = np.ones(3, np.float32), np.ones(4, np.float32)
    store.record_routing_step(job_id='j', agent_id='A', agent_index=0, env_time=0,
        local_obs=obs, global_state=state, action=1, action_type='edge_dc',
        action_source='random', immediate_reward=0, target_dc_id='B')
    pending = store.get_trace('j')
    pending.routing_steps[0].guidance_offsets = np.array([.2, -.2], np.float32)
    pending.routing_steps[0].guidance_mask = np.ones(2, dtype=bool)
    store.resolve_routing_successor(job_id='j', next_agent_id='B', next_agent_index=1,
        next_env_time=1, next_local_obs=obs * 2, next_global_state=state * 2)
    store.record_routing_step(job_id='j', agent_id='B', agent_index=1, env_time=1,
        local_obs=obs * 2, global_state=state * 2, action=1, action_type='cloud',
        action_source='policy', immediate_reward=0, target_dc_id='cloud')
    pending.routing_steps[1].guidance_offsets = np.array([-.4, .4], np.float32)
    pending.routing_steps[1].guidance_mask = np.array([False, True])
    store.record_reward_event(job_id='j', env_time=100, reward_delta=1,
        reason='completed', terminal=True)
    finalized = store.finalize_terminal_trace('j')
    pending.routing_steps[0].guidance_offsets[:] = 99
    pending.routing_steps[1].guidance_offsets[:] = 99
    pending.routing_steps[1].guidance_mask[:] = True
    first, last = finalized.routing_transitions
    assert first.next_agent_index == 1 and not first.done
    assert first.next_env_time == 1  # The successor decision, not completion at 100.
    np.testing.assert_allclose(first.next_guidance_offsets, [-.4, .4])
    np.testing.assert_array_equal(first.next_guidance_mask, [False, True])
    np.testing.assert_allclose(first.guidance_offsets, [.2, -.2])
    np.testing.assert_allclose(last.guidance_offsets, [-.4, .4])
    assert last.done and last.next_guidance_offsets is None
    tensors(s, first)  # Random warmup predecessors remain valid off-policy data.


def test_checkpoint_rejects_legacy_context_before_mutation(setup, tmp_path):
    s = setup
    path = tmp_path / 'routing.pt'
    s.system.save(path)
    s.system.load(path)  # New context versions still round-trip normally.
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if s.name == 'BCGH-MASAC':
        del saved['guidance_context_version']
    else:
        del saved['structure']['guidance_context_version']
    torch.save(saved, path)
    before = copy.deepcopy(s.local.actor.state_dict())
    with pytest.raises(RuntimeError, match='version|structure'):
        s.system.load(path)
    for key, value in before.items():
        assert torch.equal(value, s.local.actor.state_dict()[key])


def test_neutral_context_preserves_initialization_rng_and_values():
    torch.manual_seed(123)
    rng = torch.get_rng_state().clone()
    branch = GuidanceConditioner(2, 4)
    assert torch.equal(rng, torch.get_rng_state())
    with torch.no_grad():
        branch.projection.weight.fill_(1)
    reference = torch.ones(3, 5)
    assert torch.equal(branch(reference), branch(reference, torch.zeros(3, 2), torch.ones(3, 2, dtype=torch.bool)))
    assert not branch(reference).any()
