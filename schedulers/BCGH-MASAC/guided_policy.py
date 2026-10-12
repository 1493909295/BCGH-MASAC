"""Guidance schedule, terminal outcome scoring, routing policy and audits."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from schedulers.congestion_guidance import GUIDANCE_CONTEXT_VERSION, guided_distribution

if TYPE_CHECKING:
    from .bayesian_congestion_game import LocalSourceSignal, RoutingActionFeatures


# Terminal-only scoring: both local and forwarded actions use terminal observation
# windows; no remote live state is read. Reset outcome stores each episode.
@dataclass(frozen=True)
class GuidanceOutcome:
    event_id: str
    job_id: str
    source_dc_id: str
    target_dc_id: str
    decision_time: float
    observed_at: float
    success: bool
    sla_satisfied: bool
    completion_time_s: float | None
    resource_class: str = ""
    admitted: bool = True
    completed_at_target: bool = True
    admitted_at: float | None = None
    started_at: float | None = None
    local_queue_wait_s: float | None = None
    local_terminal_status: str | None = None


class TerminalOutcomeStore:
    def __init__(self, window_s, confidence_scale, prior_alpha=1.0, prior_beta=1.0):
        values = (window_s, confidence_scale, prior_alpha, prior_beta)
        if any(not math.isfinite(x) or x <= 0 for x in values):
            raise ValueError("Outcome window, confidence scale and priors must be positive")
        self.window_s, self.confidence_scale = window_s, confidence_scale
        self.prior_alpha, self.prior_beta = prior_alpha, prior_beta
        self.reset()

    def reset(self):
        self.now = 0.0
        self._records = {}
        self._seen = set()

    def advance_time(self, now):
        now = float(now)
        if not math.isfinite(now) or now < self.now - 1e-9:
            raise ValueError("Outcome clock must be finite and monotonic; reset each episode")
        self.now = now
        for pair, records in tuple(self._records.items()):
            for key, item in tuple(records.items()):
                if item.observed_at <= now - self.window_s:
                    del records[key]
            if not records:
                del self._records[pair]

    def record(self, item, *, now):
        if not isinstance(item, GuidanceOutcome):
            raise TypeError("Expected terminal GuidanceOutcome")
        if not item.event_id or not item.job_id or not item.source_dc_id or not item.target_dc_id:
            raise ValueError("Outcome needs stable event, task and DC identifiers")
        times = (item.decision_time, item.observed_at, now)
        if any(not math.isfinite(x) or x < 0 for x in times):
            raise ValueError("Outcome times must be finite and nonnegative")
        if item.decision_time > item.observed_at or item.observed_at > now + 1e-9:
            raise ValueError("Future or noncausal outcome")
        if item.sla_satisfied and not item.success:
            raise ValueError("An unsuccessful task cannot satisfy SLA")
        if item.completion_time_s is not None and (
            not item.success or not math.isfinite(item.completion_time_s) or item.completion_time_s < 0
        ):
            raise ValueError("Completion time must represent a completed task")
        self.advance_time(now)
        if item.event_id in self._seen:
            return False
        self._seen.add(item.event_id)
        if item.observed_at > now - self.window_s:
            self._records.setdefault((item.source_dc_id, item.target_dc_id), {})[item.event_id] = item
        return True

    def feedback(self, source, target, *, completion_time_scale, resource_class=None, admitted_only=False):
        if not math.isfinite(completion_time_scale) or completion_time_scale <= 0:
            raise ValueError("Completion time scale must be positive")
        records = tuple(self._records.get((source, target), {}).values())
        if admitted_only:
            records = tuple(r for r in records if r.admitted)
        pair_records = records
        class_records = tuple(r for r in records if r.resource_class == resource_class) if resource_class else ()
        # Shrink sparse resource classes toward the directed-pair estimate.
        mix = len(class_records) / (len(class_records) + 4.0)
        n = len(records)
        completed = [r.completion_time_s for r in records if r.completion_time_s is not None]
        delay = sum(completed) / len(completed) if completed else None
        bad = sum(not r.success or not r.sla_satisfied for r in records)
        result = dict(
            success_score=sum(r.success for r in records) / n if n else 0.5,
            sla_score=sum(r.sla_satisfied for r in records) / n if n else 0.5,
            delay_score=completion_time_scale / (delay + completion_time_scale) if delay is not None else 0.5,
            delay_known=delay is not None,
            outcome_samples=n,
            outcome_confidence=n / (n + self.confidence_scale),
            outcome_probability=(self.prior_alpha + bad) / (self.prior_alpha + self.prior_beta + n),
            feedback_age_s=self.now - max(r.observed_at for r in records) if n else None,
        )
        if class_records:
            m = len(class_records)
            completed_class = [r.completion_time_s for r in class_records if r.completion_time_s is not None]
            delay_class = sum(completed_class)/len(completed_class) if completed_class else None
            estimates = dict(success_score=sum(r.success for r in class_records)/m,
                sla_score=sum(r.sla_satisfied for r in class_records)/m,
                delay_score=completion_time_scale/(delay_class+completion_time_scale) if delay_class is not None else .5,
                outcome_probability=(self.prior_alpha+sum(not r.success or not r.sla_satisfied for r in class_records))/(self.prior_alpha+self.prior_beta+m))
            for key, value in estimates.items():
                result[key] = mix*value+(1-mix)*result[key]
        admitted = [r for r in pair_records if r.admitted]
        result.update(resource_class_samples=len(class_records), resource_class_mix=mix,
            admitted_terminal_samples=len(admitted),
            effective_local_completion_rate=sum(r.success and r.sla_satisfied and r.completed_at_target for r in admitted)/len(admitted) if admitted else None)
        return result


class SymmetricGuidanceMixin:
    def update_guidance_outcomes(self, trace, env):
        """Only a closed trace reveals quality; host admission is not success."""
        self.outcome_store.advance_time(float(env.current_time))
        job = env.job_map[str(trace.job_id)]
        success = str(trace.terminal_reason) == "completed"
        delay = None
        if success:
            delay = job.get_turnaround_time()
            if delay is None:
                delay = max(float(trace.terminal_time) - float(job.arrive_time), 0.0)
            delay = float(delay)
        sla = success and delay <= float(env.sla_deadline_ratio) * float(job.duration) + 1e-9
        runtime = getattr(self, "lifecycle_runtime", None)
        if runtime is not None:
            runtime.record_terminal(trace, job, sla)
        outcomes = []
        for index, step in enumerate(trace.routing_steps):
            if step.action_type not in {"self", "edge_dc", "cloud"}:
                continue
            if step.action_type == "self" and getattr(trace, "host_step", None) is None:
                continue
            target = str(step.target_dc_id)
            host = getattr(trace, "host_step", None)
            next_step = trace.routing_steps[index+1] if index+1 < len(trace.routing_steps) else None
            local_visit = step.action_type == 'self' or (next_step is not None and next_step.action_type == 'self' and str(next_step.source_dc_id) == target)
            admitted = local_visit and host is not None and str(getattr(host, "dc_id", "")) == target and getattr(host, "execution_result", None) in {"started", "queued"}
            task_class = self.absorption_store.resource_class(target, getattr(job, "cpu_request", 0.0), getattr(job, "gpu_request", 0.0)) if target in self.pressure_store.capacities else "cloud"
            event_id = f"{trace.job_id}:{step.sequence_index}"
            lifecycle = runtime.lifecycle.records.get(event_id, {}) if runtime is not None else {}
            admitted_at = lifecycle.get('admitted_at')
            if admitted and admitted_at is None:
                admitted_at = float(getattr(host, 'env_time', step.env_time))
            started_at = float(job.start_time) if admitted and getattr(job, 'start_time', None) is not None else None
            local_wait = max(started_at-admitted_at, 0.0) if started_at is not None and admitted_at is not None else None
            status = lifecycle.get('terminal')
            if admitted and status is None:
                status = ('completed_on_time' if sla else 'completed_late') if success else ('dropped_after_start' if started_at is not None else 'dropped_before_start')
            outcomes.append(GuidanceOutcome(
                f"{trace.job_id}:{step.sequence_index}", str(trace.job_id), str(step.source_dc_id),
                target, float(step.env_time), float(trace.terminal_time), success, bool(sla), delay,
                task_class, admitted, bool(admitted and success),
                admitted_at, started_at, local_wait, status,
            ))
        self.completion_time_scale = max(float(env.max_job_duration) * float(env.drop_deadline_ratio), 1e-8)
        recorded = [item for item in outcomes if self.outcome_store.record(item, now=float(env.current_time))]
        callback = getattr(self, "terminal_audit_callback", None)
        if callback is not None and str(trace.job_id) not in self._audited_terminal_jobs:
            self._audited_terminal_jobs.add(str(trace.job_id))
            callback(trace, recorded)
        return recorded

    def evaluate_actions(self, source_dc_id, job_context):
        from .bayesian_congestion_game import RoutingActionFeatures
        legacy = self._evaluate_legacy_actions(source_dc_id, job_context)
        if self.scoring_mode == "legacy":
            return tuple(replace(r, applied_resource_cost=self.risk_congestion_cost_weight * r.congestion_cost,
                                 applied_transit_penalty=r.transit_penalty) for r in legacy)
        rows = []
        for row in legacy:
            target = row.target_dc_id
            features = job_context[target]
            if isinstance(features, Mapping):
                features = RoutingActionFeatures(**features)
            task_class = self.absorption_store.resource_class(target, features.cpu_request, features.gpu_request) if target in self.pressure_store.capacities else "cloud"
            quality = self.outcome_store.feedback(source_dc_id, target, completion_time_scale=self.completion_time_scale, resource_class=task_class)
            benefit = sum(w * quality[k] for w, k in zip(self.benefit_weights, ("success_score", "sla_score", "delay_score")))
            conf, probability = quality['outcome_confidence'], quality['outcome_probability']
            # No source-only queue pseudo-evidence in symmetric terminal scoring.
            # It remains available in the legacy row for diagnostics/ablations.
            effective = 0.5 + conf * (probability - 0.5)
            # Apply the same potential to SELF and remote candidates.
            raw_cost = self.get_congestion_cost(target, features.cpu_request, features.gpu_request,
                local=target == source_dc_id, service_s=features.expected_service_s or None) if target in self.pressure_store.capacities else 0.0
            feasible = math.isfinite(raw_cost)
            cost = raw_cost/(raw_cost+self.resource_cost_scale) if feasible else 1.0
            resource = self.risk_congestion_cost_weight * cost if self.resource_penalty_enabled else 0.0
            # Forwarding itself is not a failure: only known deadline overrun
            # and returning to a visited DC add residual risk. Latency/energy
            # already represented by terminal returns are not charged again.
            transit = (.05 * min(max(getattr(features, 'path_cost', 0.0)-1.0, 0.0), 1.0)
                       + .2 * getattr(features, 'cycle_risk', 0.0)) if self.transit_penalty_enabled else 0.0
            utility = 0.5 + conf * (benefit - 0.5) - resource - transit
            rows.append(replace(
                row, benefit=benefit, guidance_utility=utility, utility=utility,
                feasible=feasible, congestion_cost=cost, resource_cost_raw=raw_cost if feasible else 0.0,
                cpu_pressure=self.get_pressure(target, local=target == source_dc_id)[0] if target in self.pressure_store.capacities else 0.0,
                gpu_pressure=self.get_pressure(target, local=target == source_dc_id)[1] if target in self.pressure_store.capacities else 0.0,
                risk=resource + transit, confidence=conf,
                outcome_probability=probability, outcome_confidence=conf,
                outcome_samples=quality['outcome_samples'], applied_resource_cost=resource,
                applied_transit_penalty=transit, effective_probability=effective,
                success_score=quality['success_score'], sla_score=quality['sla_score'],
                delay_score=quality['delay_score'], delay_known=quality['delay_known'],
                feedback_age_s=quality['feedback_age_s'],
                terminal_class_samples=quality['resource_class_samples'],
                terminal_class_mix=quality['resource_class_mix'],
                admitted_terminal_samples=quality['admitted_terminal_samples'],
                effective_local_completion_rate=quality['effective_local_completion_rate'],
            ))
        feasible = [r for r in rows if r.feasible]
        mean = sum(r.guidance_utility for r in feasible) / max(1, len(feasible))
        biases = [self.guidance_scale * (r.guidance_utility - mean) if r.feasible else 0.0 for r in rows]
        scale = min(1.0, self.max_logit_bias / max(max(map(abs, biases)), 1e-12))
        return tuple(replace(r, bias=b * scale) for r, b in zip(rows, biases))


def reset_policy_diagnostics(policy):
    policy.last_guidance_snapshot = None
    policy.last_probabilities = None
    policy._decision_totals = {}


def record_policy_diagnostics(policy, source, base_probs, guided_probs, *, logits=None, offsets=None, mask=None):
    base, guided = base_probs.detach().cpu().reshape(-1).tolist(), guided_probs.detach().cpu().reshape(-1).tolist()
    self_index = policy.action_target_dc_ids.index(source)
    def entropy(probs):
        return -sum(p * math.log(p) for p in probs if p > 0)
    detail = dict(base_probabilities=base, guided_probabilities=guided,
                  base_argmax=max(range(len(base)), key=base.__getitem__),
                  guided_argmax=max(range(len(guided)), key=guided.__getitem__),
                  self_probability_before=base[self_index], self_probability_after=guided[self_index],
                  base_entropy=entropy(base), guided_entropy=entropy(guided))
    if logits is not None:
        logit_span = float((logits.max(-1).values - logits.min(-1).values).mean())
        offset_span = float((offsets.max(-1).values - offsets.min(-1).values).mean())
        detail.update(logit_span=logit_span, effective_bias_span=offset_span,
                      bias_logit_span_ratio=offset_span / max(logit_span, 1e-8),
                      masked_action_count=int((~mask).sum()) if mask is not None else 0,
                      snapshot_complete=True)
        confidences = [float(row.outcome_confidence) for row in policy.last_breakdowns]
        detail["outcome_confidence"] = sum(confidences) / len(confidences) if confidences else None
    policy.last_probabilities = detail
    totals = policy._decision_totals.setdefault(source, dict(n=0, self_before=0., self_after=0., changed=0, base_entropy=0., guided_entropy=0.))
    totals['n'] += 1
    totals['self_before'] += detail['self_probability_before']
    totals['self_after'] += detail['self_probability_after']
    totals['changed'] += detail['base_argmax'] != detail['guided_argmax']
    totals['base_entropy'] += detail['base_entropy']
    totals['guided_entropy'] += detail['guided_entropy']
    for key in ('logit_span', 'effective_bias_span', 'bias_logit_span_ratio',
                'masked_action_count', 'snapshot_complete', 'outcome_confidence'):
        if detail.get(key) is not None:
            totals[key] = totals.get(key, 0.) + detail[key]
            totals[key + '_n'] = totals.get(key + '_n', 0) + 1


def policy_diagnostics(policy, source=None):
    totals = list(policy._decision_totals.values()) if source is None else [policy._decision_totals.get(source, {})]
    sums = {k: sum(t.get(k, 0) for t in totals) for k in ('n', 'self_before', 'self_after', 'changed', 'base_entropy', 'guided_entropy')}
    n = sums['n']
    result = dict(guidance_context_version=GUIDANCE_CONTEXT_VERSION, guidance_decision_count=n,
                guidance_self_probability_before=sums['self_before'] / n if n else None,
                guidance_self_probability_after=sums['self_after'] / n if n else None,
                guidance_self_probability_shift=(sums['self_after'] - sums['self_before']) / n if n else None,
                guidance_argmax_changed_count=sums['changed'],
                guidance_base_entropy=sums['base_entropy'] / n if n else None,
                guidance_executed_entropy=sums['guided_entropy'] / n if n else None)
    for key in ('logit_span', 'effective_bias_span', 'bias_logit_span_ratio',
                'masked_action_count', 'snapshot_complete', 'outcome_confidence'):
        count = sum(t.get(key + '_n', 0) for t in totals)
        result['guidance_mean_' + key] = sum(t.get(key, 0.) for t in totals) / count if count else None
    return result


class GuidanceDecisionLog:
    """Sample decision details, record every terminal result for linkage by job_id."""
    def __init__(self, path, sample_every=100):
        if isinstance(sample_every, bool) or not isinstance(sample_every, int) or sample_every < 1:
            raise ValueError("Guidance log sampling interval must be a positive integer")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open('x', encoding='utf-8')
        self.sample_every, self.count = sample_every, 0

    def _write(self, row):
        self.file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')

    def record(self, *, episode, decision, policy, action, guidance_lambda, sequence_index=None):
        self.count += 1
        if (self.count - 1) % self.sample_every:
            return
        self._write(dict(event='decision', guidance_context_version=GUIDANCE_CONTEXT_VERSION,
                         episode=episode, time=decision.env_time,
                         job_id=decision.job_id, source_dc_id=decision.agent_id, action=action,
                         sequence_index=sequence_index,
                         event_id=f'{decision.job_id}:{sequence_index}' if sequence_index is not None else None,
                         action_target_dc_ids=list(policy.action_target_dc_ids),
                         scoring_mode=getattr(policy.bayesian_game, 'scoring_mode', None),
                         guidance_lambda=guidance_lambda, probabilities=policy.last_probabilities,
                         candidates=[asdict(item) for item in policy.last_breakdowns]))

    def record_terminal(self, *, episode, trace, outcomes):
        self._write(dict(event='terminal', episode=episode, job_id=trace.job_id,
                         time=trace.terminal_time, terminal_reason=trace.terminal_reason,
                         outcomes=[asdict(item) for item in outcomes]))

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()


# Guidance schedule
@dataclass(frozen=True)
class GuidanceLambdaSchedule:
    """按现有三阶段 Episode 边界计算 [0, 1] 内的 λ。"""

    stage2_start: float = 0.0
    stage2_peak: float = 1.0
    stage3_end: float = 0.3

    def __post_init__(self) -> None:
        values = (
            self.stage2_start,
            self.stage2_peak,
            self.stage3_end,
        )
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("Guidance λ 参数必须位于 [0, 1]。")
        if self.stage2_start > self.stage2_peak:
            raise ValueError("stage2_start 不能大于 stage2_peak。")
        if self.stage3_end > self.stage2_peak:
            raise ValueError("stage3_end 不能大于 stage2_peak。")

    @staticmethod
    def _interpolate(
        start: float,
        end: float,
        progress: float,
    ) -> float:
        progress = max(0.0, min(1.0, float(progress)))
        if progress == 0.0:
            return float(start)
        if progress == 1.0:
            return float(end)
        return float(start + (end - start) * progress)

    def lambda_for_episode(
        self,
        *,
        stage: str,
        episode: int,
        stage_start_episode: int,
        stage_end_episode: int,
    ) -> float:
        """返回指定 Episode 的 Guidance λ。"""

        stage = str(stage)
        episode = int(episode)
        stage_start_episode = int(stage_start_episode)
        stage_end_episode = int(stage_end_episode)
        if stage_end_episode < stage_start_episode:
            raise ValueError("训练阶段结束 Episode 不能早于开始 Episode。")

        if stage == "host_pretrain":
            return 0.0
        if stage not in {"routing_train", "joint_finetune"}:
            raise ValueError(f"未知训练阶段：{stage}")

        if stage_start_episode == stage_end_episode:
            return self.stage2_peak if stage == "routing_train" else self.stage3_end

        denominator = max(
            1,
            stage_end_episode - stage_start_episode,
        )
        progress = (episode - stage_start_episode) / denominator

        if stage == "routing_train":
            return self._interpolate(
                self.stage2_start,
                self.stage2_peak,
                progress,
            )

        if stage == "joint_finetune":
            return self._interpolate(
                self.stage2_peak,
                self.stage3_end,
                progress,
            )

        raise ValueError(f"未知训练阶段：{stage}")


# Guided policy
class GuidedRoutingPolicy:
    """将基础 Routing MASAC 与可选启发式 Bias 统一为一个动作接口。"""

    def __init__(
        self,
        base_policy: Any,
        action_target_dc_ids: Sequence[str],
        *,
        enabled: bool = False,
        bayesian_game: Optional[Any] = None,
    ) -> None:
        self.base_policy = base_policy
        self.action_target_dc_ids = tuple(
            str(target_dc_id) for target_dc_id in action_target_dc_ids
        )
        if len(self.action_target_dc_ids) != int(base_policy.action_dim):
            raise ValueError("action_target_dc_ids 数量必须等于 Routing Actor action_dim。")
        if len(set(self.action_target_dc_ids)) != len(self.action_target_dc_ids):
            raise ValueError("action_target_dc_ids 不能包含重复动作目标。")

        reset_policy_diagnostics(self)
        self.enabled = bool(enabled)
        self.bayesian_game = bayesian_game
        self.last_breakdowns = ()
        self._feasible_mask = None

    def reset_diagnostics(self):
        reset_policy_diagnostics(self)

    def diagnostics(self, source_dc_id=None):
        return policy_diagnostics(self, source_dc_id)

    def build_action_context(
        self,
        source_dc_id: str,
        historical_feedback_provider: Optional[Any] = None,
        *,
        cpu_request: float = 0.0,
        gpu_request: float = 0.0,
        source_signals: Optional[Mapping[str, LocalSourceSignal]] = None,
    ) -> Mapping[str, RoutingActionFeatures]:
        """从历史反馈构造动作特征，不读取远端实时状态。

        缺少某个 Pair 历史样本时使用 0.5 中性先验；这不会伪造成功记录，
        只表示当前 Evidence 不足，真正的远端风险仍由 Bayesian confidence 抑制。
        """

        # Imported when building context so the game can import scoring classes
        # from this module without a circular import during initialization.
        from .bayesian_congestion_game import LocalSourceSignal, RoutingActionFeatures

        source_dc_id = str(source_dc_id)
        action_context: Dict[str, RoutingActionFeatures] = {}
        for target_dc_id in self.action_target_dc_ids:
            success_score = 0.5
            sla_score = 0.5
            delay_score = 0.5

            # 只有 target 是其他 Edge DC 时才查询历史 Neighbor Pair。
            if target_dc_id != source_dc_id and historical_feedback_provider is not None:
                feedback = historical_feedback_provider.get_feedback(
                    source_dc_id,
                    target_dc_id,
                )
                if feedback is not None:
                    success_score = float(feedback.get("success_ewma", 0.5))
                    sla_score = float(feedback.get("sla_success_ewma", 0.5))
                    # Provider 中 completion_time_ewma 越大表示耗时越长，
                    # 而 Guided Policy 的 delay_score 定义为越大越好。
                    delay_score = 1.0 - float(feedback.get("completion_time_ewma", 0.5))

            action_context[target_dc_id] = RoutingActionFeatures(
                target_dc_id=target_dc_id,
                success_score=max(0.0, min(1.0, success_score)),
                sla_score=max(0.0, min(1.0, sla_score)),
                delay_score=max(0.0, min(1.0, delay_score)),
                cpu_request=float(cpu_request),
                gpu_request=float(gpu_request),
                source_signal=(source_signals or {}).get(target_dc_id, LocalSourceSignal()),
            )

        return action_context

    def snapshot_context(self, source_dc_id, action_context, guidance_lambda):
        """Capture the target policy context even during uniform warmup."""
        if not math.isfinite(guidance_lambda) or not 0.0 <= guidance_lambda <= 1.0:
            raise ValueError("guidance_lambda must lie in [0, 1]")
        if not self.enabled:
            return None
        if guidance_lambda == 0.0:
            return (np.zeros(self.base_policy.action_dim, dtype=np.float32),
                    np.ones(self.base_policy.action_dim, dtype=bool))
        rows = self.bayesian_game.evaluate_actions(str(source_dc_id), action_context)
        by_target = {row.target_dc_id: row for row in rows}
        return (np.asarray([guidance_lambda * by_target[t].bias for t in self.action_target_dc_ids], dtype=np.float32),
                np.asarray([by_target[t].feasible for t in self.action_target_dc_ids], dtype=bool))

    def _bias_vector(
        self,
        action_bias: Mapping[Any, float],
    ) -> torch.Tensor:
        """将 action index 或 target DC key 的 Bias 转成 Actor 向量。"""

        if not isinstance(action_bias, Mapping):
            raise TypeError("action_bias 必须是动作索引/目标到 Bias 的映射。")

        bias_values = np.zeros(
            int(self.base_policy.action_dim),
            dtype=np.float32,
        )
        target_to_index = {
            target_dc_id: index for index, target_dc_id in enumerate(self.action_target_dc_ids)
        }

        for raw_key, raw_value in action_bias.items():
            if isinstance(raw_key, (int, np.integer)):
                action_index = int(raw_key)
            else:
                target_dc_id = str(raw_key)
                if target_dc_id not in target_to_index:
                    raise KeyError(f"Bias 包含未知 Routing action target：{target_dc_id}")
                action_index = target_to_index[target_dc_id]

            if not 0 <= action_index < len(bias_values):
                raise IndexError(f"Bias action index 越界：{action_index}")
            bias_value = float(raw_value)
            if not math.isfinite(bias_value):
                raise ValueError("action_bias 必须全部是有限数。")
            bias_values[action_index] = bias_value

        return torch.as_tensor(
            bias_values,
            dtype=torch.float32,
            device=self.base_policy.device,
        )

    def _select_with_bias(
        self,
        local_obs: np.ndarray,
        agent_index: int,
        action_bias: Mapping[Any, float],
        guidance_lambda: float,
        deterministic: bool,
        source_dc_id: str,
    ) -> int:
        """仅在需要指导时执行 ``logits + λ × bias`` 的动作采样。

        ``guidance_lambda`` 是调度器给出的归一化强度；具体 Bias 尺度仍由
        Bayesian Game 的 ``BCGH_GUIDANCE_SCALE`` 控制，避免把训练阶段调度
        与效用量纲耦合在一起。
        """

        local_obs_array = np.asarray(
            local_obs,
            dtype=np.float32,
        ).copy()
        local_obs_tensor = torch.as_tensor(
            local_obs_array,
            dtype=torch.float32,
            device=self.base_policy.device,
        ).unsqueeze(0)
        agent_index_tensor = torch.tensor(
            [int(agent_index)],
            dtype=torch.long,
            device=self.base_policy.device,
        )

        guidance_lambda = float(guidance_lambda)
        if not math.isfinite(guidance_lambda) or not 0.0 <= guidance_lambda <= 1.0:
            raise ValueError("guidance_lambda 必须位于 [0, 1]。")

        with torch.no_grad():
            # 采样与 SAC 更新共用同一复合分布；启发式快照不参与梯度。
            offsets = guidance_lambda * self._bias_vector(action_bias).unsqueeze(0)
            mask = None
            if self._feasible_mask is not None:
                mask = torch.as_tensor(
                    self._feasible_mask, dtype=torch.bool, device=self.base_policy.device
                ).unsqueeze(0)
                if not bool(mask.any()):
                    raise RuntimeError("No statically feasible routing action")
            logits = self.base_policy.actor.forward(
                local_obs=local_obs_tensor, agent_indices=agent_index_tensor,
                guidance_offsets=offsets, guidance_mask=mask)
            action_probs, _, _ = guided_distribution(logits, offsets, mask)
            self.last_guidance_snapshot = (offsets.cpu().numpy()[0].copy(),
                np.ones(self.base_policy.action_dim, dtype=bool) if mask is None else mask.cpu().numpy()[0].copy())

            record_policy_diagnostics(self, str(source_dc_id),
                guided_distribution(logits, mask=mask)[0], action_probs,
                logits=logits, offsets=offsets, mask=mask)

            if deterministic:
                action_tensor = torch.argmax(
                    action_probs,
                    dim=-1,
                )
            else:
                action_tensor = torch.distributions.Categorical(probs=action_probs).sample()

        return int(action_tensor.item())

    def select_action(
        self,
        local_obs: np.ndarray,
        agent_index: int,
        source_dc_id: str,
        *,
        action_context: Optional[Mapping[str, Any]] = None,
        action_bias: Optional[Mapping[Any, float]] = None,
        guidance_lambda: float = 0.3,
        deterministic: bool = False,
    ) -> int:
        """统一选择 Routing Action。

        ``action_bias`` 优先级高于 ``bayesian_game`` 自动计算结果，便于
        测试和未来接入其他启发式提供者。默认关闭时直接调用基础策略，
        不改变原有采样实现。``guidance_lambda`` 只影响启用指导时的
        action-level Bias，并将因果快照写入 Replay，训练同一复合策略。
        """

        self.last_guidance_snapshot = None
        self.last_breakdowns = ()
        self.last_probabilities = None
        self._feasible_mask = None
        if not math.isfinite(guidance_lambda) or not 0.0 <= guidance_lambda <= 1.0:
            raise ValueError("guidance_lambda must lie in [0, 1]")
        if (not self.enabled and action_bias is None) or guidance_lambda == 0.0:
            if self.enabled:
                self.last_guidance_snapshot = (np.zeros(self.base_policy.action_dim, dtype=np.float32),
                                               np.ones(self.base_policy.action_dim, dtype=bool))
            return self.base_policy.select_action(
                local_obs=local_obs,
                agent_index=agent_index,
                deterministic=deterministic,
            )

        if action_bias is None:
            if self.bayesian_game is None:
                raise RuntimeError("Guided Policy 已启用，但未提供 Bayesian Game 或 action_bias。")
            if action_context is None:
                raise RuntimeError("Guided Policy 已启用，但缺少候选动作 action_context。")
            self.last_breakdowns = self.bayesian_game.evaluate_actions(
                str(source_dc_id),
                action_context,
            )
            by_target = {item.target_dc_id: item for item in self.last_breakdowns}
            action_bias = {target: item.bias for target, item in by_target.items()}
            self._feasible_mask = [
                by_target[target].feasible for target in self.action_target_dc_ids
            ]

        return self._select_with_bias(
            local_obs=local_obs,
            agent_index=agent_index,
            action_bias=action_bias,
            guidance_lambda=guidance_lambda,
            deterministic=deterministic,
            source_dc_id=str(source_dc_id),
        )
