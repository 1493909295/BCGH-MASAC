"""Behavioral checks for independent routing and same-job DC successors."""

import copy
import csv
import importlib
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


PACKAGE = "schedulers.MR-BCGH-MASAC"
networks = importlib.import_module(PACKAGE + ".h_masac_agent")
experience = importlib.import_module(PACKAGE + ".experience")
support = importlib.import_module(PACKAGE + ".training_support")
guided = importlib.import_module(PACKAGE + ".guided_policy")
training = importlib.import_module(PACKAGE + ".train_mr_bcgh_masac")


@pytest.fixture
def system(monkeypatch):
    # Unit tests are device-independent; production retains its GPU preflight.
    monkeypatch.setattr(networks, "resolve_training_device", lambda _: torch.device("cpu"))
    config = networks.RoutingMASACConfig(
        device="cpu", actor_hidden_dim=8, critic_hidden_dim=8,
        policy_update_interval=1, target_update_interval=1,
    )
    return networks.RoutingMASAC(3, 4, 2, 2, ["A", "B"], ["A", "B"], config)


def transition(source="A", successor="B", *, done=False, action_source="policy"):
    return experience.RoutingTransition(
        job_id="job", agent_id=source, agent_index=["A", "B"].index(source),
        env_time=0.0, local_obs=np.array([0.2, 0.4, 0.6], np.float32),
        global_state=np.ones(4, np.float32), action=1, action_type="edge_dc",
        action_source=action_source, reward=1.0,
        next_agent_id=None if done else successor,
        next_agent_index=-1 if done else ["A", "B"].index(successor),
        next_env_time=1.0, next_local_obs=np.zeros(3, np.float32),
        next_global_state=np.zeros(4, np.float32),
        terminated=done, truncated=False, done=done,
        terminal_reason="completed" if done else None,
    )


def replays():
    return experience.RoutingReplayBuffers(12, 3, 4, ["A", "B"], seed=42)


def snapshot(agent):
    return [p.detach().clone() for net in (agent.actor, agent.critic, agent.target_critic)
            for p in net.parameters()] + [agent.log_alpha.detach().clone()]


def assert_unchanged(agent, before):
    assert all(torch.equal(x, y) for x, y in zip(snapshot(agent), before))


def test_independent_parameters_and_updates(system):
    a, b = system.agents.values()
    for name in ("actor", "critic", "target_critic"):
        left, right = getattr(a, name), getattr(b, name)
        assert {p.data_ptr() for p in left.parameters()}.isdisjoint(
            {p.data_ptr() for p in right.parameters()}
        )
    assert a.log_alpha.data_ptr() != b.log_alpha.data_ptr()
    assert a.actor.fc1.in_features == 3
    assert a.critic.q1.fc1.in_features == 4
    assert not torch.equal(a.actor.fc1.weight, b.actor.fc1.weight)
    before_a, before_b = snapshot(a), snapshot(b)
    replay = replays()
    replay.add(transition())
    a.update(replay.buffers["A"], 1, system.next_soft_values)
    assert any(not torch.equal(x, y) for x, y in zip(snapshot(a), before_a))
    assert_unchanged(b, before_b)
    assert b.update_step == 0
    assert all(p.grad is None for p in b.actor.parameters())
    assert all(p.grad is None for p in b.target_critic.parameters())


def constant_networks(agent, q, alpha):
    with torch.no_grad():
        for parameter in agent.actor.parameters():
            parameter.zero_()
        for parameter in agent.target_critic.parameters():
            parameter.zero_()
        agent.target_critic.q1.output_layer.bias.fill_(q)
        agent.target_critic.q2.output_layer.bias.fill_(q + 1)
        agent.log_alpha.fill_(math.log(alpha))


def test_cross_dc_bootstrap_and_terminal(system):
    constant_networks(system.agents["A"], q=2, alpha=0.1)
    constant_networks(system.agents["B"], q=7, alpha=0.6)
    replay = replays()
    for item in (transition(successor="B"), transition(successor="A"), transition(done=True)):
        replay.add(item)
    batch = system.agents["A"]._batch_to_tensors(replay.buffers["A"].sample(3))
    values = system.next_soft_values(batch)
    for index, value in enumerate(values):
        expected = (0 if batch.done[index] else
                    7 + 0.6 * math.log(2) if batch.next_agent_indices[index] == 1 else
                    2 + 0.1 * math.log(2))
        assert float(value) == pytest.approx(expected)
    info = system.agents["A"]._update_critic(batch, system.next_soft_values)
    expected_target = batch.rewards + system.config.gamma * values
    assert float(info["mean_target_q"]) == pytest.approx(float(expected_target.mean()))


def test_terminal_rows_never_call_successor_and_invalid_successor_fails(system, monkeypatch):
    replay = replays()
    replay.add(transition(done=True))
    batch = system.agents["A"]._batch_to_tensors(replay.buffers["A"].sample(1))
    with monkeypatch.context() as patch:
        patch.setattr(system, "agent_for_index", lambda _: pytest.fail("Terminal accessed a DC"))
        assert system.next_soft_values(batch).item() == 0
    bad = replace(batch, done=torch.zeros_like(batch.done), next_agent_indices=torch.tensor([-1]))
    with pytest.raises(ValueError, match="successor"):
        system.next_soft_values(bad)


def test_replay_source_ownership_and_delayed_updates(system):
    replay = replays()
    item = transition()
    with pytest.raises(ValueError, match="belong"):
        replay.buffers["B"].add(item)
    with pytest.raises(ValueError, match="mismatched"):
        replay.add(replace(item, agent_index=1))
    with pytest.raises(ValueError, match="successor"):
        replay.add(replace(item, next_agent_id="A"))
    for source in ("forced", "orchestrator"):
        with pytest.raises(ValueError, match="trainable"):
            replay.add(replace(item, action_source=source))
    settings = SimpleNamespace(routing_learning_starts=2, routing_train_every=2,
                               routing_batch_size=1, routing_updates_per_train=1)
    for _ in range(3):
        system.record_action("A")
    assert system.update_ready(replay, settings) == []
    replay.add(item)  # Outcome arrives after the interval boundary.
    assert len(replay.buffers["A"]) == 1
    assert len(replay.buffers["B"]) == 0
    assert [dc for dc, _ in system.update_ready(replay, settings)] == ["A"]
    assert system.update_ready(replay, settings) == []  # No duplicate block.
    assert system.agents["B"].update_step == 0


def test_guided_and_plain_selection_use_current_dc(system):
    constant_networks(system.agents["A"], 0, 0.1)
    constant_networks(system.agents["B"], 0, 0.1)
    with torch.no_grad():
        system.agents["A"].actor.output_layer.bias.copy_(torch.tensor([2.0, 0.0]))
        system.agents["B"].actor.output_layer.bias.copy_(torch.tensor([0.0, 2.0]))
    policy = guided.GuidedRoutingPolicy(system, ["A", "B"])
    assert policy.select_action(np.zeros(3), 0, "A", deterministic=True) == 0
    assert policy.select_action(np.zeros(3), 1, "B", deterministic=True) == 1
    assert policy.select_action(np.zeros(3), 0, "A", action_bias={1: 5},
                                guidance_lambda=1, deterministic=True) == 1
    with pytest.raises(ValueError, match="mismatch"):
        policy.select_action(np.zeros(3), 0, "B")


def test_routing_checkpoint_roundtrip_and_atomic_failure(system, tmp_path):
    replay = replays()
    replay.add(transition())
    system.record_action("A")
    system.agents["A"].update(replay.buffers["A"], 1, system.next_soft_values)
    system.last_train_action_steps["A"] = 1
    checkpoint = tmp_path / "final.pt"
    system.save(checkpoint)
    restored = copy.deepcopy(system)
    with torch.no_grad():
        restored.agents["B"].actor.fc1.weight.add_(3)
    restored.load(checkpoint)
    for dc in system.edge_dc_ids:
        assert_unchanged(restored.agents[dc], snapshot(system.agents[dc]))
    assert restored.trainer_counters() == system.trainer_counters()
    assert restored.agents["A"].actor_optimizer.state
    before = {dc: snapshot(agent) for dc, agent in restored.agents.items()}
    (tmp_path / "final_routing" / "dc_1.pt").write_bytes(b"corrupted")
    with pytest.raises(Exception):
        restored.load(checkpoint)
    for dc in system.edge_dc_ids:
        assert_unchanged(restored.agents[dc], before[dc])
    system.save(checkpoint)
    (tmp_path / "final_routing" / "dc_1.pt").unlink()
    with pytest.raises(FileNotFoundError):
        restored.load(checkpoint)
    for dc in system.edge_dc_ids:
        assert_unchanged(restored.agents[dc], before[dc])
    torch.save({"algorithm_role": "routing_masac_ctde"}, checkpoint)
    with pytest.raises(RuntimeError, match="not MR"):
        restored.load(checkpoint)


def test_modes_and_per_dc_metrics(system):
    host = {"A": SimpleNamespace(train_mode=lambda: None, eval_mode=lambda: None)}
    for stage in support.TrainingStage:
        support.apply_training_stage_modes(stage, system, host)
        expected = stage != support.TrainingStage.HOST_PRETRAIN
        assert all(agent.actor.training == expected for agent in system.agents.values())
        assert all(not agent.target_critic.training for agent in system.agents.values())
    stats = support.EpisodeStatistics(1, 42, "routing_train", {"A": 0, "B": 0})
    metrics = {name: torch.tensor(1.) for name in support.UPDATE_TENSOR_METRIC_NAMES}
    support.record_routing_update_block(stats, "A", [{**metrics, "actor_loss": torch.tensor(2.)}])
    support.record_routing_update_block(stats, "B", [{**metrics, "actor_loss": torch.tensor(6.)}])
    support.flush_pending_update_metrics(stats)
    assert stats.mean_routing_metric("actor_loss") == 4
    assert stats.mean_routing_metric("actor_loss", "A") == 2
    assert stats.mean_routing_metric("actor_loss", "B") == 6
    assert stats.routing_update_count == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Production training requires CUDA")
@pytest.mark.parametrize("cloud_enabled", [False, True])
def test_three_stage_training_restore_and_evaluation(tmp_path, monkeypatch, cloud_enabled):
    from environment.cloud_edge_env import CloudEdgeEnv
    from environment.env_generate import UseOldEnv
    import config

    monkeypatch.setattr(config, "ENABLE_CLOUD_ACTION", cloud_enabled)

    project = Path(__file__).resolve().parents[1]
    fixture = project / "environment/env_keep/2026_10_07_10_18"
    if not fixture.is_dir():
        pytest.skip("Saved environment fixture is unavailable")

    def small_environment(seed, old_env_path=None):
        source = UseOldEnv(str(fixture))
        source.job_num = 25
        return CloudEdgeEnv(env_source=source, seed=seed)

    monkeypatch.setattr(training, "build_environment", small_environment)
    settings = support.TrainConfig(
        num_episodes=3, host_pretrain_episodes=1, routing_train_episodes=1,
        joint_finetune_episodes=1, routing_replay_capacity=128, host_replay_capacity=128,
        routing_batch_size=2, host_batch_size=2, routing_random_warmup_steps=2,
        routing_learning_starts=1, host_random_warmup_steps=0, host_learning_starts=1,
        routing_train_every=1, host_train_every=1, routing_updates_per_train=1,
        host_updates_per_train=1, checkpoint_dir=str(tmp_path / "checkpoints"),
        episode_log_csv_path=str(tmp_path / "episode.csv"), dc_log_csv_path=str(tmp_path / "dc.csv"),
        log_interval=1, seed=42,
    )
    route_config = networks.RoutingMASACConfig(
        device="cuda:0", actor_hidden_dim=8, critic_hidden_dim=8, policy_update_interval=1
    )
    host_config = networks.HostSACConfig(
        device="cuda:0", actor_hidden_dim=8, critic_hidden_dim=8, policy_update_interval=1
    )
    system, hosts = training.train(settings, route_config, host_config)
    episode_path = next(tmp_path.glob("episode_*.csv"))
    dc_path = next(tmp_path.glob("dc_*.csv"))
    with episode_path.open(encoding="utf-8-sig", newline="") as stream:
        episodes = list(csv.DictReader(stream))
    with dc_path.open(encoding="utf-8-sig", newline="") as stream:
        dc_rows = list(csv.DictReader(stream))
    assert len(episodes) == 3
    assert int(episodes[0]["routing_episode_updates"]) == 0
    assert int(episodes[1]["host_episode_updates"]) == 0
    assert int(episodes[1]["routing_episode_updates"]) > 0
    assert int(episodes[2]["routing_episode_updates"]) > 0
    assert int(episodes[2]["host_episode_updates"]) > 0
    assert len(dc_rows) == 3 * system.num_agents
    assert all(row["routing_parameter_sharing"] == "False" for row in dc_rows)
    assert sum(system.action_steps.values()) == int(episodes[-1]["routing_training_action_steps"])
    before = {dc: snapshot(agent) for dc, agent in system.agents.items()}
    env = small_environment(42)
    result = support.evaluate_episode(env, system, hosts, settings)
    assert result["routing_decisions"] > 0
    unguided = replace(settings, enable_bayesian_game=False, enable_heuristic_guidance=False)
    result = support.evaluate_episode(env, system, hosts, unguided)
    assert result["routing_decisions"] > 0
    for dc in system.edge_dc_ids:
        assert_unchanged(system.agents[dc], before[dc])
    checkpoint = tmp_path / "checkpoints/final.pt"
    restored, restored_hosts = training.train(replace(settings, resume_checkpoint=str(checkpoint)),
                                              route_config, host_config)
    assert restored.trainer_counters() == system.trainer_counters()
    for dc in system.edge_dc_ids:
        assert_unchanged(restored.agents[dc], before[dc])
    # A broken final Host file must not mutate any routing or Host network.
    host_before = {dc: snapshot(agent) for dc, agent in restored_hosts.items()}
    last_dc = system.edge_dc_ids[-1]
    (support.host_checkpoint_dir(checkpoint) / f"{last_dc}.pt").write_bytes(b"corrupted")
    with pytest.raises(Exception):
        support.load_two_layer_checkpoint_if_needed(
            env, settings, restored, restored_hosts, str(checkpoint)
        )
    for dc in system.edge_dc_ids:
        assert_unchanged(restored.agents[dc], before[dc])
        assert_unchanged(restored_hosts[dc], host_before[dc])
