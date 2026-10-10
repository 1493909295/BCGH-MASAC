"""Congestion beliefs and short-window runtime."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from .bayesian_game import (
    BayesianHistoricalEvidence,
    build_bayesian_routing_game_definition,
    build_bayesian_static_routing_context,
)


# Congestion game
def positive(value, name, *, zero=False):
    value = float(value)
    if not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return value


@dataclass(frozen=True)
class BayesianCongestionPairKey:
    source_dc_id: str
    target_dc_id: str


@dataclass(frozen=True)
class StaticDCCapacity:
    cpu: float
    gpu: float
    # Static host capacities, needed to reject impossible single-host placements.
    hosts: tuple[tuple[float, float], ...]

    def __post_init__(self):
        positive(self.cpu, "cpu capacity", zero=True)
        positive(self.gpu, "gpu capacity", zero=True)
        for cpu, gpu in self.hosts:
            positive(cpu, "host cpu", zero=True)
            positive(gpu, "host gpu", zero=True)

    def fits(self, cpu, gpu):
        return any(cpu <= c + 1e-9 and gpu <= g + 1e-9 for c, g in self.hosts)


@dataclass(frozen=True)
class LocalSourceSignal:
    waiting_volume: float = 0.0
    running_volume: float = 0.0
    effective_events: float = 0.0


@dataclass(frozen=True)
class RoutingActionFeatures:
    target_dc_id: str
    success_score: float = 0.5
    sla_score: float = 0.5
    delay_score: float = 0.5
    cpu_request: float = 0.0
    gpu_request: float = 0.0
    source_signal: LocalSourceSignal = LocalSourceSignal()


@dataclass(frozen=True)
class ActionUtilityBreakdown:
    target_dc_id: str
    benefit: float
    congestion_probability: float
    congestion_cost: float
    risk: float
    utility: float
    confidence: float
    bias: float
    outcome_probability: float = 0.5
    source_signal: float = 0.0
    source_pseudo_count: float = 0.0
    source_confidence: float = 0.0
    outcome_confidence: float = 0.0
    resource_cost_raw: float = 0.0
    feasible: bool = True
    guidance_utility: float = 0.0
    outcome_samples: float = 0.0
    cpu_pressure: float = 0.0
    gpu_pressure: float = 0.0
    cpu_request: float = 0.0
    gpu_request: float = 0.0
    absorption_probability: float = 0.0
    edge_reforward_probability: float = 0.0
    cloud_escape_probability: float = 0.0
    absorption_drop_probability: float = 0.0
    absorption_confidence: float = 0.0
    absorption_pair_samples: float = 0.0
    absorption_class_samples: float = 0.0
    absorption_class_mix: float = 0.0
    absorption_resource_class: str = ""
    transit_penalty_raw: float = 0.0
    transit_penalty: float = 0.0


@dataclass
class BetaBeliefState:
    alpha_congested: float = 1.0
    beta_non_congested: float = 1.0
    sample_weight: float = 0.0
    last_update_clock: float | None = None

    @property
    def congestion_probability(self):
        return self.alpha_congested / (self.alpha_congested + self.beta_non_congested)

    def confidence(self, confidence_scale):
        return self.sample_weight / (
            self.sample_weight + positive(confidence_scale, "confidence_scale")
        )


ABSORPTION_OUTCOMES = ("self", "edge", "cloud", "drop")


@dataclass(frozen=True)
class AbsorptionEvidence:
    """Observed successor of one executed source->target edge transfer."""

    event_id: str
    job_id: str
    source_dc_id: str
    target_dc_id: str
    routed_at: float
    observed_at: float
    outcome: str
    cpu_request: float
    gpu_request: float
    resource_class: str
    predecessor_action_source: str
    successor_action_source: str
    informative: bool = True

    def __post_init__(self):
        if not self.event_id or not self.job_id:
            raise ValueError("Absorption evidence requires event_id and job_id")
        if not self.source_dc_id or not self.target_dc_id or self.source_dc_id == self.target_dc_id:
            raise ValueError("Absorption evidence requires a directed non-self DC pair")
        if self.outcome not in ABSORPTION_OUTCOMES:
            raise ValueError(f"Unknown absorption outcome: {self.outcome}")
        positive(self.routed_at, "routed_at", zero=True)
        positive(self.observed_at, "observed_at", zero=True)
        if self.observed_at + 1e-9 < self.routed_at:
            raise ValueError("Absorption outcome precedes its edge transfer")
        positive(self.cpu_request, "cpu request", zero=True)
        positive(self.gpu_request, "gpu request", zero=True)


@dataclass(frozen=True)
class AbsorptionBeliefState:
    p_absorb: float
    p_edge: float
    p_cloud: float
    p_drop: float
    confidence: float
    pair_samples: float
    class_samples: float
    class_mix: float
    resource_class: str
    counts: Mapping[str, float]


class ShortWindowAbsorptionStore:
    """Dirichlet successor-action beliefs for each directed Edge pair."""

    def __init__(
        self,
        edge_dc_ids,
        capacities,
        *,
        window_s=1200.0,
        priors=None,
        confidence_scale=8.0,
        demand_thresholds=(0.25, 0.60),
        class_confidence_scale=4.0,
    ):
        self.edge_dc_ids = tuple(map(str, edge_dc_ids))
        self.capacities = dict(capacities)
        self.window_s = positive(window_s, "absorption window")
        priors = priors or {name: 1.0 for name in ABSORPTION_OUTCOMES}
        if set(priors) != set(ABSORPTION_OUTCOMES):
            raise ValueError("Absorption priors must cover self/edge/cloud/drop")
        self.priors = {
            name: positive(priors[name], f"{name} prior") for name in ABSORPTION_OUTCOMES
        }
        self.confidence_scale = positive(confidence_scale, "absorption confidence scale")
        thresholds = tuple(float(x) for x in demand_thresholds)
        if len(thresholds) != 2 or not 0 < thresholds[0] < thresholds[1]:
            raise ValueError("Demand thresholds must be two increasing positive values")
        self.demand_thresholds = thresholds
        self.class_confidence_scale = positive(
            class_confidence_scale, "absorption class confidence scale"
        )
        self.reset()

    def reset(self):
        self.now = 0.0
        self._records = {}
        self._seen_event_ids = set()
        self._episode_counts = {name: 0 for name in ABSORPTION_OUTCOMES}
        self._episode_informative_counts = {name: 0 for name in ABSORPTION_OUTCOMES}
        self._episode_pair_counts = {}

    def advance_time(self, now):
        now = positive(now, "simulation time", zero=True)
        if now < self.now - 1e-9:
            raise ValueError("Simulation clock moved backwards")
        self.now = now
        self._records = {
            key: record
            for key, record in self._records.items()
            if record.observed_at > now - self.window_s
        }

    def resource_class(self, target_dc_id, cpu_request, gpu_request):
        capacity = self.capacities[str(target_dc_id)]
        max_cpu = max((cpu for cpu, _ in capacity.hosts), default=0.0)
        max_gpu = max((gpu for _, gpu in capacity.hosts), default=0.0)
        cpu_ratio = float(cpu_request) / max_cpu if max_cpu > 0 else 0.0
        gpu_ratio = float(gpu_request) / max_gpu if max_gpu > 0 else 0.0
        dominant = max(cpu_ratio, gpu_ratio)
        if dominant < self.demand_thresholds[0]:
            size = "low"
        elif dominant < self.demand_thresholds[1]:
            size = "medium"
        else:
            size = "high"
        return f"{'gpu' if float(gpu_request) > 0 else 'cpu'}:{size}"

    def record(self, evidence):
        if not isinstance(evidence, AbsorptionEvidence):
            raise TypeError("Expected AbsorptionEvidence")
        if (
            evidence.source_dc_id not in self.capacities
            or evidence.target_dc_id not in self.capacities
        ):
            raise KeyError(
                f"Unknown absorption pair: {evidence.source_dc_id}->{evidence.target_dc_id}"
            )
        if evidence.event_id in self._seen_event_ids:
            return False
        self.advance_time(max(self.now, evidence.observed_at))
        self._seen_event_ids.add(evidence.event_id)
        if evidence.observed_at > self.now - self.window_s:
            self._records[evidence.event_id] = evidence
        self._episode_counts[evidence.outcome] += 1
        if evidence.informative:
            self._episode_informative_counts[evidence.outcome] += 1
        pair = f"{evidence.source_dc_id}->{evidence.target_dc_id}"
        pair_counts = self._episode_pair_counts.setdefault(
            pair, {name: 0 for name in ABSORPTION_OUTCOMES}
        )
        pair_counts[evidence.outcome] += 1
        return True

    def records(self, source_dc_id, target_dc_id, *, informative_only=True):
        source_dc_id, target_dc_id = str(source_dc_id), str(target_dc_id)
        return tuple(
            r
            for r in self._records.values()
            if r.source_dc_id == source_dc_id
            and r.target_dc_id == target_dc_id
            and (r.informative or not informative_only)
        )

    def belief(self, source_dc_id, target_dc_id, cpu_request=0.0, gpu_request=0.0):
        target_dc_id = str(target_dc_id)
        resource_class = self.resource_class(target_dc_id, cpu_request, gpu_request)
        pair_records = self.records(source_dc_id, target_dc_id)
        class_records = tuple(r for r in pair_records if r.resource_class == resource_class)

        def posterior(records):
            counts = {name: sum(r.outcome == name for r in records) for name in ABSORPTION_OUTCOMES}
            total = sum(counts.values())
            denominator = sum(self.priors.values()) + total
            probabilities = {
                name: (self.priors[name] + counts[name]) / denominator
                for name in ABSORPTION_OUTCOMES
            }
            return counts, float(total), probabilities

        pair_counts, pair_n, pair_p = posterior(pair_records)
        _, class_n, class_p = posterior(class_records)
        class_mix = class_n / (class_n + self.class_confidence_scale)
        probabilities = {
            name: class_mix * class_p[name] + (1.0 - class_mix) * pair_p[name]
            for name in ABSORPTION_OUTCOMES
        }
        confidence = pair_n / (pair_n + self.confidence_scale)
        return AbsorptionBeliefState(
            p_absorb=probabilities["self"],
            p_edge=probabilities["edge"],
            p_cloud=probabilities["cloud"],
            p_drop=probabilities["drop"],
            confidence=confidence,
            pair_samples=pair_n,
            class_samples=class_n,
            class_mix=class_mix,
            resource_class=resource_class,
            counts=pair_counts,
        )

    def snapshot(self):
        result = {}
        pairs = {(r.source_dc_id, r.target_dc_id) for r in self._records.values()}
        for source, target in sorted(pairs):
            records = self.records(source, target)
            result[f"{source}->{target}"] = {
                name: sum(r.outcome == name for r in records) for name in ABSORPTION_OUTCOMES
            }
        return result

    def episode_summary(self):
        total = sum(self._episode_counts.values())
        informative = sum(self._episode_informative_counts.values())
        return {
            "total": total,
            "informative_total": informative,
            "counts": dict(self._episode_counts),
            "informative_counts": dict(self._episode_informative_counts),
            "pair_counts": {key: dict(value) for key, value in self._episode_pair_counts.items()},
            "edge_to_cloud_transit_rate": (self._episode_counts["cloud"] / total if total else 0.0),
            "edge_to_edge_reforward_rate": (self._episode_counts["edge"] / total if total else 0.0),
            "edge_target_absorption_rate": (self._episode_counts["self"] / total if total else 0.0),
        }


class BayesianBeliefStore:
    def __init__(
        self,
        edge_dc_ids,
        *,
        prior_alpha=1.0,
        prior_beta=1.0,
        confidence_scale=20.0,
        window_s=1200.0,
    ):
        self.edge_dc_ids = tuple(map(str, edge_dc_ids))
        self.prior_alpha = positive(prior_alpha, "prior_alpha")
        self.prior_beta = positive(prior_beta, "prior_beta")
        self.confidence_scale = positive(confidence_scale, "confidence_scale")
        self.window_s = positive(window_s, "window_s")
        self.reset()

    def reset(self):
        self.now = 0.0
        self._records = {(i, j): {} for i in self.edge_dc_ids for j in self.edge_dc_ids if i != j}

    def advance_time(self, now):
        now = positive(now, "simulation time", zero=True)
        if now < self.now - 1e-9:
            raise ValueError("Simulation clock moved backwards; reset the episode first")
        self.now = now
        for records in self._records.values():
            for key in tuple(records):
                if records[key].routing_time <= now - self.window_s:
                    del records[key]

    def records(self, source_dc_id, target_dc_id):
        return tuple(self._records[(str(source_dc_id), str(target_dc_id))].values())

    def update(self, evidence: BayesianHistoricalEvidence, *, update_clock=None):
        evidence.validate_information_boundary()
        if evidence.timestamp is None or evidence.routing_time is None or not evidence.evidence_id:
            raise ValueError("Window evidence requires terminal time, routing time and a unique ID")
        terminal = positive(evidence.timestamp, "terminal time", zero=True)
        routed = positive(evidence.routing_time, "routing time", zero=True)
        if routed > terminal:
            raise ValueError("Outcome precedes routing")
        self.advance_time(max(self.now, terminal) if update_clock is None else update_clock)
        if terminal > self.now + 1e-9:
            raise ValueError("Future outcome cannot be used")
        records = self._records[(evidence.source_dc_id, evidence.target_dc_id)]
        if routed > self.now - self.window_s:
            records.setdefault(evidence.evidence_id, evidence)
        return self.get_state(evidence.source_dc_id, evidence.target_dc_id)

    def get_state(self, source_dc_id, target_dc_id):
        records = self.records(source_dc_id, target_dc_id)
        bad = sum(e.evidence_weight for e in records if e.congestion_observed)
        good = sum(e.evidence_weight for e in records if not e.congestion_observed)
        return BetaBeliefState(
            self.prior_alpha + bad,
            self.prior_beta + good,
            bad + good,
            max((e.timestamp for e in records), default=None),
        )

    def get_congestion_probability(self, source_dc_id, target_dc_id):
        return self.get_state(source_dc_id, target_dc_id).congestion_probability

    def get_confidence(self, source_dc_id, target_dc_id):
        return self.get_state(source_dc_id, target_dc_id).confidence(self.confidence_scale)

    def snapshot(self):
        result = {}
        for i, j in self._records:
            state = self.get_state(i, j)
            result[f"{i}->{j}"] = dict(
                alpha_congested=state.alpha_congested,
                beta_non_congested=state.beta_non_congested,
                sample_weight=state.sample_weight,
                congestion_probability=state.congestion_probability,
                confidence=state.confidence(self.confidence_scale),
            )
        return result


class HistoricalPressureStore:
    """Executed edge transfers within (now-T, now], weighted by CPU/GPU demand."""

    def __init__(
        self,
        capacities: Mapping[str, StaticDCCapacity],
        *,
        window_s=1200.0,
        linear_cost_weight=1.0,
        quadratic_cost_weight=1.0,
        cpu_weight=0.5,
        gpu_weight=0.5,
    ):
        self.capacities = dict(capacities)
        self.edge_dc_ids = tuple(self.capacities)
        self.window_s = positive(window_s, "window_s")
        self.linear_cost_weight = positive(linear_cost_weight, "linear cost", zero=True)
        self.quadratic_cost_weight = positive(quadratic_cost_weight, "quadratic cost", zero=True)
        self.weights = (
            positive(cpu_weight, "cpu weight", zero=True),
            positive(gpu_weight, "gpu weight", zero=True),
        )
        positive(sum(self.weights), "resource weight sum")
        self.reset()

    def reset(self):
        self.now = 0.0
        self._records = {}

    def advance_time(self, now):
        now = positive(now, "simulation time", zero=True)
        if now < self.now - 1e-9:
            raise ValueError("Simulation clock moved backwards")
        self.now = now
        self._records = {k: v for k, v in self._records.items() if v[0] > now - self.window_s}

    def record_selection(self, target_dc_id, *, event_id, event_time, cpu_request, gpu_request):
        if target_dc_id not in self.capacities:
            raise KeyError(target_dc_id)
        cpu = positive(cpu_request, "cpu request", zero=True)
        gpu = positive(gpu_request, "gpu request", zero=True)
        event_time = positive(event_time, "event time", zero=True)
        if event_time > self.now + 1e-9:
            raise ValueError("Transfer has not occurred")
        if event_time > self.now - self.window_s:
            self._records.setdefault(event_id, (event_time, target_dc_id, cpu, gpu))

    def get_pressure(self, target_dc_id):
        capacity = self.capacities[target_dc_id]
        selected = [r for r in self._records.values() if r[1] == target_dc_id]
        return tuple(
            sum(r[index + 2] for r in selected) / cap if cap > 0 else 0.0
            for index, cap in enumerate((capacity.cpu, capacity.gpu))
        )

    def get_congestion_cost(self, target_dc_id, cpu_request=0.0, gpu_request=0.0):
        cpu_request = positive(cpu_request, "cpu request", zero=True)
        gpu_request = positive(gpu_request, "gpu request", zero=True)
        capacity = self.capacities[target_dc_id]
        if not capacity.fits(cpu_request, gpu_request):
            return math.inf
        cost = 0.0
        for x, demand, cap, weight in zip(
            self.get_pressure(target_dc_id),
            (cpu_request, gpu_request),
            (capacity.cpu, capacity.gpu),
            self.weights,
        ):
            delta = demand / cap if cap > 0 else 0.0
            cost += weight * (
                self.linear_cost_weight * delta
                + self.quadratic_cost_weight * (2 * x * delta + delta * delta)
            )
        return cost

    def snapshot(self):
        return {
            j: dict(
                cpu_pressure=self.get_pressure(j)[0],
                gpu_pressure=self.get_pressure(j)[1],
                recent_selection_count=sum(r[1] == j for r in self._records.values()),
            )
            for j in self.edge_dc_ids
        }


class BayesianCongestionGame:
    def __init__(
        self,
        game_definition,
        *,
        capacities,
        prior_alpha=1.0,
        prior_beta=1.0,
        confidence_scale=20.0,
        window_s=1200.0,
        linear_cost_weight=1.0,
        quadratic_cost_weight=1.0,
        benefit_success_weight=0.4,
        benefit_sla_weight=0.4,
        benefit_delay_weight=0.2,
        risk_congestion_cost_weight=1.0,
        guidance_scale=0.3,
        source_evidence_weight=2.0,
        source_volume_scale=1.0,
        source_confidence_scale=5.0,
        cpu_weight=0.5,
        gpu_weight=0.5,
        resource_cost_scale=1.0,
        max_logit_bias=0.3,
        absorption_enabled=True,
        absorption_priors=None,
        absorption_confidence_scale=8.0,
        absorption_edge_weight=0.7,
        absorption_cloud_weight=0.5,
        absorption_drop_weight=1.0,
        absorption_demand_thresholds=(0.25, 0.60),
        absorption_class_confidence_scale=4.0,
        absorption_policy_actions_only=True,
    ):
        self.game_definition = game_definition
        if set(capacities) != set(game_definition.player_ids):
            raise ValueError("Static capacities must cover exactly the edge players")
        self.belief_store = BayesianBeliefStore(
            game_definition.player_ids,
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
            confidence_scale=confidence_scale,
            window_s=window_s,
        )
        self.pressure_store = HistoricalPressureStore(
            capacities,
            window_s=window_s,
            linear_cost_weight=linear_cost_weight,
            quadratic_cost_weight=quadratic_cost_weight,
            cpu_weight=cpu_weight,
            gpu_weight=gpu_weight,
        )
        self.absorption_enabled = bool(absorption_enabled)
        self.absorption_policy_actions_only = bool(absorption_policy_actions_only)
        self.absorption_store = ShortWindowAbsorptionStore(
            game_definition.player_ids,
            capacities,
            window_s=window_s,
            priors=absorption_priors,
            confidence_scale=absorption_confidence_scale,
            demand_thresholds=absorption_demand_thresholds,
            class_confidence_scale=absorption_class_confidence_scale,
        )
        self.absorption_weights = {
            "edge": positive(absorption_edge_weight, "absorption edge weight", zero=True),
            "cloud": positive(absorption_cloud_weight, "absorption cloud weight", zero=True),
            "drop": positive(absorption_drop_weight, "absorption drop weight", zero=True),
        }
        self.benefit_weights = tuple(
            positive(w, "benefit weight", zero=True)
            for w in (benefit_success_weight, benefit_sla_weight, benefit_delay_weight)
        )
        positive(sum(self.benefit_weights), "benefit weight sum")
        self.risk_congestion_cost_weight = positive(
            risk_congestion_cost_weight, "risk weight", zero=True
        )
        self.guidance_scale = positive(guidance_scale, "guidance_scale", zero=True)
        self.source_evidence_weight = positive(source_evidence_weight, "source weight", zero=True)
        self.source_volume_scale = positive(source_volume_scale, "source volume scale")
        self.source_confidence_scale = positive(source_confidence_scale, "source confidence scale")
        self.resource_cost_scale = positive(resource_cost_scale, "resource cost scale")
        self.max_logit_bias = positive(max_logit_bias, "max logit bias")

    def reset(self):
        self.belief_store.reset()
        self.pressure_store.reset()
        self.absorption_store.reset()

    def advance_time(self, now):
        self.belief_store.advance_time(now)
        self.pressure_store.advance_time(now)
        self.absorption_store.advance_time(now)

    def update_belief(self, evidence, *, update_clock=None):
        return self.belief_store.update(evidence, update_clock=update_clock)

    def get_congestion_probability(self, source_dc_id, target_dc_id):
        return self.belief_store.get_congestion_probability(source_dc_id, target_dc_id)

    def update_pressure(self, source_dc_id, target_dc_id, **kwargs):
        self.belief_store.get_state(source_dc_id, target_dc_id)
        return self.pressure_store.record_selection(target_dc_id, **kwargs)

    def record_absorption(self, evidence):
        return self.absorption_store.record(evidence)

    def get_absorption_belief(self, source_dc_id, target_dc_id, cpu_request=0.0, gpu_request=0.0):
        return self.absorption_store.belief(source_dc_id, target_dc_id, cpu_request, gpu_request)

    def get_pressure(self, target_dc_id):
        return self.pressure_store.get_pressure(target_dc_id)

    def get_congestion_cost(self, target_dc_id, cpu_request=0.0, gpu_request=0.0):
        return self.pressure_store.get_congestion_cost(target_dc_id, cpu_request, gpu_request)

    def calculate_benefit(self, action):
        values = (action.success_score, action.sla_score, action.delay_score)
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError("Quality scores must lie in [0,1]")
        return sum(w * v for w, v in zip(self.benefit_weights, values))

    def evaluate_actions(self, source_dc_id, job_context):
        if not job_context:
            raise ValueError("No candidate actions")
        rows = []
        for target, raw in job_context.items():
            action = raw if isinstance(raw, RoutingActionFeatures) else RoutingActionFeatures(**raw)
            if target != action.target_dc_id:
                raise ValueError("Candidate target mismatch")
            benefit = self.calculate_benefit(action)
            remote = target in self.pressure_store.capacities and target != source_dc_id
            row = dict(
                target_dc_id=target,
                benefit=benefit,
                congestion_probability=0.0,
                congestion_cost=0.0,
                risk=0.0,
                utility=benefit,
                confidence=1.0,
                bias=0.0,
                guidance_utility=benefit,
                cpu_request=action.cpu_request,
                gpu_request=action.gpu_request,
            )
            if remote:
                state = self.belief_store.get_state(source_dc_id, target)
                local = action.source_signal
                volume = positive(local.waiting_volume, "waiting volume", zero=True) + positive(
                    local.running_volume, "running volume", zero=True
                )
                events = positive(local.effective_events, "source events", zero=True)
                signal = volume / (volume + self.source_volume_scale)
                pseudo = self.source_evidence_weight * signal
                source_conf = (
                    events / (events + self.source_confidence_scale) if pseudo > 0 else 0.0
                )
                out_conf = state.confidence(self.belief_store.confidence_scale)
                fused = (state.alpha_congested + pseudo) / (
                    state.alpha_congested + state.beta_non_congested + pseudo
                )
                cost = self.get_congestion_cost(target, action.cpu_request, action.gpu_request)
                feasible = math.isfinite(cost)
                normalized_cost = cost / (cost + self.resource_cost_scale) if feasible else 1.0
                # Confidence gates uncertain outcome risk and inbound evidence, not
                # known executed resource demand. Empty windows are neutral.
                confidence = 1 - (1 - out_conf) * (1 - source_conf)
                effective_probability = (
                    0.5
                    + out_conf * (state.congestion_probability - 0.5)
                    + source_conf * (fused - state.congestion_probability)
                )
                absorption = self.get_absorption_belief(
                    source_dc_id, target, action.cpu_request, action.gpu_request
                )
                transit_raw = (
                    self.absorption_weights["edge"] * absorption.p_edge
                    + self.absorption_weights["cloud"] * absorption.p_cloud
                    + self.absorption_weights["drop"] * absorption.p_drop
                )
                transit_penalty = (
                    absorption.confidence * transit_raw if self.absorption_enabled else 0.0
                )
                risk = fused + self.risk_congestion_cost_weight * normalized_cost + transit_penalty
                utility = 0.5 + out_conf * (benefit - 0.5) - (effective_probability - 0.5)
                utility -= self.risk_congestion_cost_weight * normalized_cost
                utility -= transit_penalty
                row.update(
                    congestion_probability=fused,
                    congestion_cost=normalized_cost,
                    risk=risk,
                    utility=benefit - risk,
                    guidance_utility=utility,
                    confidence=confidence,
                    outcome_probability=state.congestion_probability,
                    source_signal=signal,
                    source_pseudo_count=pseudo,
                    source_confidence=source_conf,
                    outcome_confidence=out_conf,
                    resource_cost_raw=cost if feasible else 0.0,
                    feasible=feasible,
                    outcome_samples=state.sample_weight,
                    cpu_pressure=self.get_pressure(target)[0],
                    gpu_pressure=self.get_pressure(target)[1],
                    absorption_probability=absorption.p_absorb,
                    edge_reforward_probability=absorption.p_edge,
                    cloud_escape_probability=absorption.p_cloud,
                    absorption_drop_probability=absorption.p_drop,
                    absorption_confidence=absorption.confidence,
                    absorption_pair_samples=absorption.pair_samples,
                    absorption_class_samples=absorption.class_samples,
                    absorption_class_mix=absorption.class_mix,
                    absorption_resource_class=absorption.resource_class,
                    transit_penalty_raw=transit_raw,
                    transit_penalty=transit_penalty,
                )
            rows.append(row)
        feasible_rows = [r for r in rows if r.get("feasible", True)]
        mean = sum(r["guidance_utility"] for r in feasible_rows) / max(1, len(feasible_rows))
        biases = [
            self.guidance_scale * (r["guidance_utility"] - mean) if r.get("feasible", True) else 0.0
            for r in rows
        ]
        # Scale all biases together to retain centering and their ordering.
        scale = min(1.0, self.max_logit_bias / max(max(map(abs, biases)), 1e-12))
        return tuple(
            ActionUtilityBreakdown(**{**r, "bias": b * scale}) for r, b in zip(rows, biases)
        )

    def get_action_bias(self, source_dc_id, job_context):
        return {r.target_dc_id: r.bias for r in self.evaluate_actions(source_dc_id, job_context)}


# Short-window runtime
def build_short_window_game(env, settings, definition=None):
    if definition is None:
        definition = build_bayesian_routing_game_definition(
            build_bayesian_static_routing_context(env)
        )
    return BayesianCongestionGame(
        definition,
        capacities=build_static_capacities(env),
        prior_alpha=settings.bayesian_prior_alpha,
        prior_beta=settings.bayesian_prior_beta,
        confidence_scale=settings.bayesian_confidence_scale,
        window_s=settings.short_window_s,
        source_evidence_weight=settings.source_evidence_weight,
        source_volume_scale=settings.source_volume_scale,
        source_confidence_scale=settings.source_confidence_scale,
        cpu_weight=settings.resource_cpu_weight,
        gpu_weight=settings.resource_gpu_weight,
        resource_cost_scale=settings.resource_cost_scale,
        max_logit_bias=settings.max_logit_bias,
        linear_cost_weight=settings.pressure_linear_cost_weight,
        quadratic_cost_weight=settings.pressure_quadratic_cost_weight,
        benefit_success_weight=settings.benefit_success_weight,
        benefit_sla_weight=settings.benefit_sla_weight,
        benefit_delay_weight=settings.benefit_delay_weight,
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
        absorption_policy_actions_only=settings.absorption_policy_actions_only,
    )


def build_static_capacities(env):
    """One-time whitelist: installed CPU/GPU only, never used/available resources."""
    result = {}
    base_dc_map = {str(dc.dc_id): dc for dc in env.base_datacenters}
    for dc_id in env.edge_dc_ids:
        hosts = tuple(
            (float(h.cpu_num), float(h.gpu_capacity_num)) for h in base_dc_map[dc_id].host_list
        )
        result[str(dc_id)] = StaticDCCapacity(
            sum(c for c, _ in hosts), sum(g for _, g in hosts), hosts
        )
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
                self._pending_self[job_id] = (predecessor, float(decision.env_time), action_source)
            else:
                outcome = {"edge_dc": "edge", "cloud": "cloud", "drop": "drop"}.get(action_type)
                if outcome is None:
                    raise RuntimeError(f"Unknown absorption successor action: {action_type}")
                self._record_absorption(
                    predecessor,
                    outcome,
                    observed_at=decision.env_time,
                    successor_action_source=action_source,
                )

        if action_type != "edge_dc":
            return
        self._sequence += 1
        event_id = f"{job_id}:{self._sequence}"
        self.game.update_pressure(
            decision.agent_id,
            target_dc_id,
            event_id=event_id,
            event_time=decision.env_time,
            cpu_request=job.cpu_request,
            gpu_request=job.gpu_request,
        )
        pending = PendingAbsorption(
            job_id=job_id,
            source_dc_id=str(decision.agent_id),
            target_dc_id=str(target_dc_id),
            event_id=event_id,
            routed_at=float(decision.env_time),
            cpu_request=float(job.cpu_request),
            gpu_request=float(job.gpu_request),
            predecessor_action_source=action_source,
        )
        self._pending_arrivals[job_id] = pending
        self._pending_absorption[job_id] = pending

    def _record_absorption(self, pending, outcome, *, observed_at, successor_action_source):
        informative = not self.policy_actions_only or (
            pending.predecessor_action_source == "policy"
            and str(successor_action_source) == "policy"
        )
        resource_class = self.game.absorption_store.resource_class(
            pending.target_dc_id, pending.cpu_request, pending.gpu_request
        )
        return self.game.record_absorption(
            AbsorptionEvidence(
                event_id=pending.event_id,
                job_id=pending.job_id,
                source_dc_id=pending.source_dc_id,
                target_dc_id=pending.target_dc_id,
                routed_at=pending.routed_at,
                observed_at=float(observed_at),
                outcome=str(outcome),
                cpu_request=pending.cpu_request,
                gpu_request=pending.gpu_request,
                resource_class=resource_class,
                predecessor_action_source=pending.predecessor_action_source,
                successor_action_source=str(successor_action_source),
                informative=informative,
            )
        )

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
        return self._record_absorption(
            pending,
            outcome,
            observed_at=max(float(observed_at), float(now)),
            successor_action_source=successor_action_source,
        )

    def record_arrival(self, dc_id, job_id, now):
        """Only a real current routing decision makes the receipt visible locally."""
        self.game.advance_time(now)
        pending = self._pending_arrivals.get(str(job_id))
        if pending is not None:
            if pending.target_dc_id != dc_id:
                raise RuntimeError("Incoming task target does not match current DC")
            self._receipts[dc_id][str(job_id)] = LocalReceipt(
                pending.source_dc_id,
                now,
                pending.event_id,
                pending.predecessor_action_source == "policy",
            )
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
                    volume = max(
                        float(job.cpu_request) / capacity.cpu if capacity.cpu else 0.0,
                        float(job.gpu_request) / capacity.gpu if capacity.gpu else 0.0,
                    )
                    entry = volumes.setdefault(receipt.previous_dc_id, [0.0, 0.0, 0.0])
                    entry[index] += freshness * volume
                    entry[2] += freshness
        return {j: LocalSourceSignal(*v) for j, v in volumes.items()}

    def build_action_context(self, policy, env, decision, feedback_provider):
        self.game.advance_time(decision.env_time)
        signals = self.local_source_signals(
            decision.agent_id, env.dc_map[decision.agent_id], decision.env_time
        )
        job = env.job_map[decision.job_id]
        return policy.build_action_context(
            decision.agent_id,
            feedback_provider,
            cpu_request=job.cpu_request,
            gpu_request=job.gpu_request,
            source_signals=signals,
        )

    def absorption_summary(self):
        return self.game.absorption_store.episode_summary()
