"""Causal, symmetric terminal scoring in both scheduler policy modules."""
from dataclasses import replace
import importlib
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch


def outcome(modules, key="job:0", target="A", *, success=True, sla=True, time=1., delay=1.):
    return modules.policy.GuidanceOutcome(key, key.split(":")[0], "A", target, 0., time, success, sla, delay)


@pytest.fixture(params=["BCGH-MASAC", "MR-BCGH-MASAC"])
def modules(request):
    root = "schedulers." + request.param
    return NS(game=importlib.import_module(root + ".bayesian_congestion_game"),
              policy=importlib.import_module(root + ".guided_policy"),
              support=importlib.import_module(root + ".training_support"), name=request.param)


def make_game(modules, **kwargs):
    return modules.game.BayesianCongestionGame(
        NS(player_ids=("A", "B")),
        capacities={dc: modules.game.StaticDCCapacity(100, 10, ((100, 10),)) for dc in ("A", "B")},
        window_s=10, confidence_scale=2, **kwargs,
    )


def context(modules, **kwargs):
    return {dc: modules.game.RoutingActionFeatures(target_dc_id=dc, **kwargs) for dc in ("A", "B")}


def test_terminal_window_delay_unknown_and_dedup(modules):
    store = modules.policy.TerminalOutcomeStore(10, 2)
    item = outcome(modules, time=9, success=False, sla=False, delay=None)
    assert store.record(item, now=9)
    feedback = store.feedback("A", "A", completion_time_scale=5)
    assert feedback["outcome_samples"] == 1
    assert feedback["delay_known"] is False
    assert feedback["delay_score"] == .5
    assert feedback["success_score"] == 0
    assert feedback["outcome_confidence"] == pytest.approx(1 / 3)
    store.advance_time(18.99)
    assert store.feedback("A", "A", completion_time_scale=5)["outcome_samples"] == 1
    store.advance_time(19)
    assert store.feedback("A", "A", completion_time_scale=5)["outcome_samples"] == 0
    assert not store.record(item, now=20)  # Expiry must not allow duplicate evidence.
    store.reset()
    assert store.record(item, now=9)


@pytest.mark.parametrize("changes", [dict(observed_at=3), dict(decision_time=2),
    dict(observed_at=float("nan")), dict(success=False), dict(completion_time_s=-1)])
def test_noncausal_or_invalid_outcome_rejected(changes, modules):
    store = modules.policy.TerminalOutcomeStore(10, 2)
    with pytest.raises(ValueError):
        store.record(replace(outcome(modules), **changes), now=1)
    assert not store._records


def test_quality_is_identical_for_local_and_remote(modules):
    game = make_game(modules)
    for dc in ("A", "B"):
        game.outcome_store.record(outcome(modules, dc + ":0", dc), now=1)
    rows = game.evaluate_actions("A", context(modules, cpu_request=30, gpu_request=2))
    assert rows[0].guidance_utility == pytest.approx(rows[1].guidance_utility)
    assert rows[0].confidence == rows[1].confidence
    assert rows[0].bias == pytest.approx(0)
    assert rows[1].bias == pytest.approx(0)
    assert rows[1].congestion_cost > 0  # Still measured, but inactive by default.
    assert rows[1].applied_resource_cost == 0


def test_no_terminal_samples_are_neutral_even_with_source_pressure(modules):
    game = make_game(modules)
    actions = context(modules, cpu_request=30, gpu_request=2)
    actions["B"] = replace(actions["B"], source_signal=modules.game.LocalSourceSignal(
        waiting_volume=20, running_volume=10, effective_events=10))
    rows = game.evaluate_actions("A", actions)
    assert rows[1].source_signal > 0
    for row in rows:
        assert row.outcome_samples == 0
        assert row.confidence == 0
        assert row.guidance_utility == .5
        assert row.bias == 0
        assert row.delay_known is False


def test_bad_local_outcomes_reduce_local_score(modules):
    game = make_game(modules)
    for i in range(10):
        game.outcome_store.record(outcome(modules, f"fail{i}:0", "A", success=False, sla=False, delay=None), now=1)
        game.outcome_store.record(outcome(modules, f"good{i}:0", "B"), now=1)
    local, remote = game.evaluate_actions("A", context(modules))
    assert local.bias < 0 < remote.bias
    assert local.success_score == local.sla_score == 0
    assert remote.success_score == remote.sla_score == 1
    assert max(abs(local.bias), abs(remote.bias)) <= game.max_logit_bias
    game.advance_time(11)
    assert all(r.bias == 0 for r in game.evaluate_actions("A", context(modules)))
    game.reset()
    assert game.outcome_store.now == 0


def test_penalty_switches_and_legacy_formula(modules):
    game = make_game(modules)
    actions = context(modules, cpu_request=30, gpu_request=2)
    neutral = game.evaluate_actions("A", actions)
    game.resource_penalty_enabled = True
    penalty = game.evaluate_actions("A", actions)
    assert penalty[1].applied_resource_cost > 0
    assert penalty[1].guidance_utility < neutral[1].guidance_utility
    game.record_absorption(modules.game.AbsorptionEvidence(
        event_id="forward", job_id="forward", source_dc_id="A", target_dc_id="B",
        routed_at=0, observed_at=1, outcome="edge", cpu_request=30, gpu_request=2,
        resource_class=game.absorption_store.resource_class("B", 30, 2),
        predecessor_action_source="policy", successor_action_source="policy"))
    game.advance_time(1)
    game.transit_penalty_enabled = True
    transit = game.evaluate_actions("A", actions)
    assert transit[1].applied_transit_penalty == 0  # Reforwarding is not itself a failure.
    assert transit[1].guidance_utility == penalty[1].guidance_utility
    actions['B'] = replace(actions['B'], cycle_risk=1.0)
    assert game.evaluate_actions('A', actions)[1].applied_transit_penalty > 0
    game.scoring_mode = "legacy"
    old, reproduced = game._evaluate_legacy_actions("A", actions), game.evaluate_actions("A", actions)
    assert [r.bias for r in old] == [r.bias for r in reproduced]
    assert [r.guidance_utility for r in old] == [r.guidance_utility for r in reproduced]


def test_only_terminal_trace_updates_both_route_and_host_actions(modules):
    game = make_game(modules)
    job = NS(arrive_time=0, duration=2, get_turnaround_time=lambda: 5)
    env = NS(job_map={"job": job}, current_time=5, sla_deadline_ratio=2,
             max_job_duration=10, drop_deadline_ratio=3)
    trace = NS(job_id="job", terminal_reason="completed", terminal_time=5, host_step=object(),
               routing_steps=(NS(sequence_index=0, action_type="edge_dc", source_dc_id="A", target_dc_id="B", env_time=0),
                              NS(sequence_index=1, action_type="self", source_dc_id="B", target_dc_id="B", env_time=1)))
    # Host admission never creates completion evidence.
    runtime = modules.game.ShortWindowRuntime(game)
    runtime.record_host_result(job_id="job", dc_id="B", execution_result="success", now=1)
    assert not game.outcome_store._records
    audited = []
    game.terminal_audit_callback = lambda trace, rows: audited.append(rows)
    rows = game.update_guidance_outcomes(trace, env)
    assert len(rows) == 2
    assert all(r.success and not r.sla_satisfied and r.completion_time_s == 5 for r in rows)
    assert rows[0].observed_at == rows[1].observed_at == 5
    assert game.outcome_store.feedback("A", "B", completion_time_scale=30) == game.outcome_store.feedback("B", "B", completion_time_scale=30)
    assert game.update_guidance_outcomes(trace, env) == []
    assert len(audited) == 1
    game.reset()
    assert len(game.update_guidance_outcomes(trace, env)) == 2


def test_failed_admission_is_not_success_and_forced_self_has_no_host_evidence(modules):
    game = make_game(modules)
    env = NS(job_map={"job": NS(arrive_time=0, duration=1)}, current_time=1,
             sla_deadline_ratio=2, max_job_duration=10, drop_deadline_ratio=3)
    trace = NS(job_id="job", terminal_reason="local_host_resource_failure", terminal_time=1,
               host_step=object(), routing_steps=(NS(sequence_index=0, action_type="self", source_dc_id="A", target_dc_id="A", env_time=0),))
    result = game.update_guidance_outcomes(trace, env)
    assert len(result) == 1 and not result[0].success
    assert result[0].completion_time_s is None
    game.reset()
    trace.host_step = None
    assert game.update_guidance_outcomes(trace, env) == []


def test_probability_audit_and_zero_lambda_bypass(modules):
    class BasePolicy:
        action_dim, device = 2, "cpu"
        actor = NS(forward=lambda **kw: torch.zeros(1, 2))
        def get_logits(self, *args, **kwargs):
            return torch.zeros(1, 2)
        def agent_for_index(self, index):
            return NS(dc_id="A")
        def select_action(self, **kwargs):
            return 1
    policy = modules.policy.GuidedRoutingPolicy(BasePolicy(), ("A", "B"), enabled=True)
    assert policy.select_action(np.zeros(2), 0, "A", action_bias={"A": -.3, "B": .3}, guidance_lambda=1, deterministic=True) == 1
    detail = policy.last_probabilities
    assert detail["base_probabilities"] == [.5, .5]
    assert detail["self_probability_after"] < .5
    assert sum(detail["guided_probabilities"]) == pytest.approx(1)
    assert policy.diagnostics()["guidance_self_probability_shift"] < 0
    # No game/context is needed at zero lambda, and base sampling is retained.
    assert policy.select_action(np.zeros(2), 0, "A", guidance_lambda=0) == 1
    assert policy.last_probabilities is None
    policy.reset_diagnostics()
    assert policy.diagnostics()["guidance_decision_count"] == 0


def test_decision_and_terminal_journal_linkage(tmp_path, modules):
    path = tmp_path / "audit.jsonl"
    log = modules.policy.GuidanceDecisionLog(path, sample_every=2)
    decision = NS(env_time=0, job_id="job", agent_id="A")
    policy = NS(last_probabilities={"base_probabilities": [.5, .5]}, last_breakdowns=(), bayesian_game=NS(scoring_mode="symmetric"), action_target_dc_ids=("A", "B"))
    for _ in range(3):
        log.record(episode=1, decision=decision, policy=policy, action=0, guidance_lambda=1, sequence_index=0)
    log.record_terminal(episode=1, trace=NS(job_id="job", terminal_time=1, terminal_reason="completed"), outcomes=[outcome(modules)])
    log.close()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [r["event"] for r in rows] == ["decision", "decision", "terminal"]
    assert all(r["job_id"] == "job" for r in rows)
    assert rows[0]["event_id"] == rows[-1]["outcomes"][0]["event_id"]
    with pytest.raises(FileExistsError):
        modules.policy.GuidanceDecisionLog(path)


def test_checkpoint_semantics_identify_new_scoring(modules):
    settings = modules.support.TrainConfig()
    meta = modules.support.bcgh_guidance_metadata(settings)
    assert meta["version"] == 5
    assert meta["parameters"]["scoring_mode"] == "symmetric"
    assert meta != modules.support.bcgh_guidance_metadata(replace(settings, scoring_mode="legacy"))
    assert settings.resource_penalty_enabled and settings.transit_penalty_enabled
    assert 0 < settings.risk_congestion_cost_weight < 1
    assert settings.guidance_lambda_stage2_start == settings.guidance_lambda_stage2_peak == settings.guidance_lambda_stage3_end == .3
    assert meta["guidance_context_version"] == 1
