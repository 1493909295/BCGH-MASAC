"""BGH environment adapters: static capacity, local arrivals and local queues only.

The shared pressure observer sees executed routing history, as in the original
trainer. Queue queries only inspect the explicitly requested source DC.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import json

from bayesian_congestion_game import (AbsorptionEvidence, BayesianCongestionGame,
    LocalSourceSignal, StaticDCCapacity)
from bayesian_game import build_bayesian_static_routing_context, build_bayesian_routing_game_definition


def build_short_window_game(env, settings, definition=None):
    if definition is None:
        definition = build_bayesian_routing_game_definition(build_bayesian_static_routing_context(env))
    return BayesianCongestionGame(definition, capacities=build_static_capacities(env),
        prior_alpha=settings.bayesian_prior_alpha, prior_beta=settings.bayesian_prior_beta,
        confidence_scale=settings.bayesian_confidence_scale, window_s=settings.short_window_s,
        source_evidence_weight=settings.source_evidence_weight,
        source_volume_scale=settings.source_volume_scale,
        source_confidence_scale=settings.source_confidence_scale,
        cpu_weight=settings.resource_cpu_weight, gpu_weight=settings.resource_gpu_weight,
        resource_cost_scale=settings.resource_cost_scale, max_logit_bias=settings.max_logit_bias,
        linear_cost_weight=settings.pressure_linear_cost_weight,
        quadratic_cost_weight=settings.pressure_quadratic_cost_weight,
        benefit_success_weight=settings.benefit_success_weight,
        benefit_sla_weight=settings.benefit_sla_weight, benefit_delay_weight=settings.benefit_delay_weight,
        risk_congestion_cost_weight=settings.risk_congestion_cost_weight,
        guidance_scale=settings.guidance_scale,
        absorption_enabled=settings.absorption_enabled,
        absorption_priors={
            "self": settings.absorption_prior_self,
            "edge": settings.absorption_prior_edge,
            "cloud": settings.absorption_prior_cloud,
            "drop": settings.absorption_prior_drop,
        },
        absorption_confidence_scale=settings.absorption_confidence_scale,
        absorption_edge_weight=settings.absorption_edge_weight,
        absorption_cloud_weight=settings.absorption_cloud_weight,
        absorption_drop_weight=settings.absorption_drop_weight,
        absorption_demand_thresholds=settings.absorption_demand_thresholds,
        absorption_class_confidence_scale=settings.absorption_class_confidence_scale,
        absorption_policy_actions_only=settings.absorption_policy_actions_only)


def build_static_capacities(env):
    """One-time whitelist: installed CPU/GPU only, never used/available resources."""
    result = {}
    base_dc_map = {str(dc.dc_id): dc for dc in env.base_datacenters}
    for dc_id in env.edge_dc_ids:
        hosts = tuple((float(h.cpu_num), float(h.gpu_capacity_num))
                      for h in base_dc_map[dc_id].host_list)
        result[str(dc_id)] = StaticDCCapacity(
            sum(c for c, _ in hosts), sum(g for _, g in hosts), hosts)
    return result


@dataclass(frozen=True)
class LocalReceipt:
    previous_dc_id: str
    received_at: float
    event_id: str
    # Only policy-generated offloads are evidence of deliberate load shedding.
    informative: bool


@dataclass(frozen=True)
class PendingAbsorption:
    job_id: str
    source_dc_id: str
    target_dc_id: str
    event_id: str
    routed_at: float
    cpu_request: float
    gpu_request: float
    predecessor_action_source: str


class ShortWindowRuntime:
    def __init__(self, game):
        self.game = game
        self.policy_actions_only = bool(game.absorption_policy_actions_only)
        self.reset()

    def reset(self):
        self.game.reset()
        self._pending_arrivals = {}
        self._pending_absorption = {}
        self._pending_self = {}
        self._receipts = {i: {} for i in self.game.game_definition.player_ids}
        self._sequence = 0

    def record_routing(self, decision, action_type, target_dc_id, action_source, job):
        """Called after env.step succeeded, before building the next observation."""
        action_type = str(action_type)
        action_source = str(action_source)
        job_id = str(decision.job_id)
        self.game.advance_time(decision.env_time)

        predecessor = self._pending_absorption.pop(job_id, None)
        if predecessor is not None:
            if predecessor.target_dc_id != str(decision.agent_id):
                raise RuntimeError("Absorption successor DC does not match edge target")
            if action_type == "self":
                self._pending_self[job_id] = (
                    predecessor, float(decision.env_time), action_source)
            else:
                outcome = {
                    "edge_dc": "edge", "cloud": "cloud", "drop": "drop"
                }.get(action_type)
                if outcome is None:
                    raise RuntimeError(f"Unknown absorption successor action: {action_type}")
                self._record_absorption(predecessor, outcome,
                    observed_at=decision.env_time, successor_action_source=action_source)

        if action_type != "edge_dc":
            return
        self._sequence += 1
        event_id = f"{job_id}:{self._sequence}"
        self.game.update_pressure(decision.agent_id, target_dc_id,
            event_id=event_id, event_time=decision.env_time,
            cpu_request=job.cpu_request, gpu_request=job.gpu_request)
        pending = PendingAbsorption(job_id=job_id,
            source_dc_id=str(decision.agent_id), target_dc_id=str(target_dc_id),
            event_id=event_id, routed_at=float(decision.env_time),
            cpu_request=float(job.cpu_request), gpu_request=float(job.gpu_request),
            predecessor_action_source=action_source)
        self._pending_arrivals[job_id] = pending
        self._pending_absorption[job_id] = pending

    def _record_absorption(self, pending, outcome, *, observed_at, successor_action_source):
        informative = (not self.policy_actions_only or (
            pending.predecessor_action_source == "policy"
            and str(successor_action_source) == "policy"))
        resource_class = self.game.absorption_store.resource_class(
            pending.target_dc_id, pending.cpu_request, pending.gpu_request)
        return self.game.record_absorption(AbsorptionEvidence(
            event_id=pending.event_id, job_id=pending.job_id,
            source_dc_id=pending.source_dc_id, target_dc_id=pending.target_dc_id,
            routed_at=pending.routed_at, observed_at=float(observed_at), outcome=str(outcome),
            cpu_request=pending.cpu_request, gpu_request=pending.gpu_request,
            resource_class=resource_class,
            predecessor_action_source=pending.predecessor_action_source,
            successor_action_source=str(successor_action_source), informative=informative))

    def record_host_result(self, job_id, dc_id, execution_result, now):
        """Finalize a provisional SELF successor after the real host admission result."""
        self.game.advance_time(now)
        item = self._pending_self.pop(str(job_id), None)
        if item is None:
            return False
        pending, observed_at, successor_action_source = item
        if pending.target_dc_id != str(dc_id):
            raise RuntimeError("Host result DC does not match absorption target")
        execution_result = str(execution_result)
        if execution_result not in {"started", "queued", "dropped"}:
            raise ValueError(f"Unknown host execution result: {execution_result}")
        outcome = "self" if execution_result in {"started", "queued"} else "drop"
        return self._record_absorption(pending, outcome,
            observed_at=max(float(observed_at), float(now)),
            successor_action_source=successor_action_source)

    def record_arrival(self, dc_id, job_id, now):
        """Only a real current routing decision makes the receipt visible locally."""
        self.game.advance_time(now)
        pending = self._pending_arrivals.get(str(job_id))
        if pending is not None:
            if pending.target_dc_id != dc_id:
                raise RuntimeError("Incoming task target does not match current DC")
            self._receipts[dc_id][str(job_id)] = LocalReceipt(
                pending.source_dc_id, now, pending.event_id,
                pending.predecessor_action_source == "policy")
            del self._pending_arrivals[str(job_id)]
        cutoff = now - self.game.belief_store.window_s
        for receipts in self._receipts.values():
            for key in tuple(receipts):
                if receipts[key].received_at <= cutoff:
                    del receipts[key]

    def local_source_signals(self, source_dc_id, local_dc, now):
        """Caller supplies exactly one local DC; no access to an environment here."""
        self.game.advance_time(now)
        capacity = self.game.pressure_store.capacities[source_dc_id]
        volumes = {}
        seen = set()
        for host in local_dc.host_list:
            for queue_name, index in (("waiting_queue", 0), ("running_queue", 1)):
                for job in getattr(host, queue_name)._queue:
                    job_id = str(job.job_id)
                    if job_id in seen:
                        continue
                    seen.add(job_id)
                    receipt = self._receipts[source_dc_id].get(job_id)
                    if receipt is None or not receipt.informative:
                        continue
                    age = now - receipt.received_at
                    if not 0 <= age < self.game.belief_store.window_s:
                        continue
                    freshness = 1.0 - age / self.game.belief_store.window_s
                    # Fractions of the receiver's installed capacity, not raw
                    # CPU+GPU units and not guessed remote free capacity.
                    volume = max(float(job.cpu_request) / capacity.cpu if capacity.cpu else 0.0,
                                 float(job.gpu_request) / capacity.gpu if capacity.gpu else 0.0)
                    entry = volumes.setdefault(receipt.previous_dc_id, [0.0, 0.0, 0.0])
                    entry[index] += freshness * volume
                    entry[2] += freshness
        return {j: LocalSourceSignal(*v) for j, v in volumes.items()}

    def build_action_context(self, policy, env, decision, feedback_provider):
        self.game.advance_time(decision.env_time)
        signals = self.local_source_signals(decision.agent_id,
            env.dc_map[decision.agent_id], decision.env_time)
        job = env.job_map[decision.job_id]
        return policy.build_action_context(decision.agent_id, feedback_provider,
            cpu_request=job.cpu_request, gpu_request=job.gpu_request, source_signals=signals)

    def absorption_summary(self):
        return self.game.absorption_store.episode_summary()


class GuidanceDecisionLog:
    """One JSONL record per guided decision; allows auditing individual actions."""
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", encoding="utf-8")

    def record(self, *, episode, decision, policy, action, guidance_lambda):
        self.file.write(json.dumps(dict(episode=episode, time=decision.env_time,
            job_id=decision.job_id, source_dc_id=decision.agent_id, action=action,
            guidance_lambda=guidance_lambda,
            candidates=[asdict(item) for item in policy.last_breakdowns]),
            ensure_ascii=False, allow_nan=False) + "\n")

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()
