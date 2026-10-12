"""Shared causal guidance primitives for BCGH and MR-BCGH (schema 5)."""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import torch


GUIDANCE_CONTEXT_VERSION = 1


def copy_guidance_array(value, *, mask=False):
    """Every causal storage boundary owns its arrays, including finalization."""
    return None if value is None else np.array(value, dtype=bool if mask else np.float32, copy=True)


def validate_replay_guidance(transition, *, required=False):
    """Reject incomplete causal contexts rather than fabricating neutral ones."""
    current = (transition.guidance_offsets, transition.guidance_mask)
    successor = (transition.next_guidance_offsets, transition.next_guidance_mask)
    guided = current[0] is not None
    if (current[0] is None) != (current[1] is None):
        raise ValueError('Incomplete current guidance snapshot')
    if required and not guided:
        raise ValueError('Missing required current guidance snapshot')
    if (successor[0] is None) != (successor[1] is None):
        raise ValueError('Incomplete successor guidance snapshot')
    if not transition.done and guided != (successor[0] is not None):
        raise ValueError('Nonterminal successor guidance snapshot is missing or inconsistent')
    if not guided and successor[0] is not None:
        raise ValueError('Successor guidance without a current snapshot')
    if guided and getattr(transition, 'guidance_context_version', None) != GUIDANCE_CONTEXT_VERSION:
        raise ValueError('Replay guidance context version mismatch')
    if guided:
        shape = np.asarray(current[0]).shape
        if len(shape) != 1 or shape[0] < 1:
            raise ValueError('Guidance snapshot must have one action dimension')
        for offsets, mask in (current, successor):
            if offsets is None:
                continue
            if (np.asarray(offsets).shape != shape or np.asarray(mask).shape != shape
                    or not np.isfinite(offsets).all() or not np.isfinite(mask).all()
                    or not np.asarray(mask, dtype=bool).any()):
                raise ValueError('Invalid guidance snapshot shape, values or action mask')
    return guided


class GuidanceConditioner(torch.nn.Module):
    """Encode the detached execution snapshot in actor/critic hidden features.

    The effective lambda*bias and static forbidden-action flags are sufficient
    to describe the direct policy intervention. No live beliefs are queried.
    A zero-initialized, bias-free branch preserves the original neutral policy
    and RNG stream while allowing a learned response to historical guidance.
    """
    def __init__(self, action_dim, hidden_dim):
        super().__init__()
        self.action_dim = int(action_dim)
        with torch.random.fork_rng(devices=[]):
            self.projection = torch.nn.Linear(2 * self.action_dim, int(hidden_dim), bias=False)
        torch.nn.init.zeros_(self.projection.weight)

    def forward(self, reference, offsets=None, mask=None):
        shape = (*reference.shape[:-1], self.action_dim)
        features = reference.new_zeros((*reference.shape[:-1], 2 * self.action_dim))
        if offsets is not None:
            if tuple(offsets.shape) != shape or not bool(torch.isfinite(offsets).all()):
                raise ValueError('Invalid guidance offsets shape or values')
            features[..., :self.action_dim] = offsets.detach().to(reference)
        if mask is not None:
            if tuple(mask.shape) != shape or not bool(mask.bool().any(-1).all()):
                raise ValueError('Invalid guidance mask shape or empty action set')
            features[..., self.action_dim:] = (~mask.detach().bool()).to(reference)
        return self.projection(features)


def guided_distribution(logits, offsets=None, mask=None, eps=1e-8):
    """Same composite policy for execution, actor, target value and temperature."""
    if offsets is not None and offsets.shape[-1]:
        logits = logits + offsets.detach().to(logits)
    if mask is not None and mask.shape[-1]:
        mask = mask.to(device=logits.device, dtype=torch.bool)
        if not bool(mask.any(dim=-1).all()):
            raise ValueError("Every guidance mask must allow at least one action")
        logits = logits.masked_fill(~mask, -torch.inf)
    probabilities = torch.softmax(logits, dim=-1)
    # Finite logs keep masked 0 * log(0) out of SAC expectations.
    return probabilities, probabilities.clamp_min(eps).log(), logits


def guidance_tensor(value, device, *, mask=False):
    if value is None:
        return None
    return torch.as_tensor(value, device=device, dtype=torch.bool if mask else torch.float32)


class ActivePressureMixin:
    """Estimated outstanding commitments, never remote live resource readings.

    A pending commitment survives the observation window. Its unknown residence
    has exponential survival, with a target-specific EWMA learned on release.
    Historical transfer counts are retained separately by the underlying store.
    """
    def reset(self):
        super().reset()
        self._active = {}
        self._seen_selections = set()
        self._residence = {}
        self._service_hints = {}
        self._started = {}
        self._local_events = set()
        self.initial_residence_s = getattr(self, 'initial_residence_s', 300.0)
        self.background_cpu = getattr(self, 'background_cpu', 0.0)
        self.background_gpu = getattr(self, 'background_gpu', 0.0)

    def record_selection(self, target_dc_id, **kwargs):
        key = kwargs['event_id']
        if key in self._seen_selections:
            return
        super().record_selection(target_dc_id, **kwargs)
        if key not in self._seen_selections:
            self._seen_selections.add(key)
            self._active[key] = (kwargs['event_time'], target_dc_id,
                                 kwargs['cpu_request'], kwargs['gpu_request'])

    def release(self, event_id, now):
        self.advance_time(now)
        item = self._active.pop(event_id, None)
        if item is None:
            return False
        age = max(float(now) - item[0], 1.0)
        self._service_hints.pop(event_id, None)
        self._started.pop(event_id, None)
        old = self._residence.get(item[1], self.initial_residence_s)
        self._residence[item[1]] = .9 * old + .1 * age
        return True

    def set_service_hint(self, event_id, duration_s):
        if not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError('Expected service duration must be finite and positive')
        if event_id in self._active:
            self._service_hints[event_id] = float(duration_s)

    def mark_started(self, event_id, now, target_dc_id=None):
        if event_id in self._active and (target_dc_id is None or self._active[event_id][1] == target_dc_id):
            self._started.setdefault(event_id, float(now))

    def mark_local(self, event_id):
        self._local_events.add(event_id)

    def remaining_service(self, event_id, record, mean):
        duration = self._service_hints.get(event_id, self.initial_residence_s)
        if event_id in self._started:
            return max(duration - max(self.now - self._started[event_id], 0.0), 0.0)
        return duration * math.exp(-max(self.now - record[0], 0.0) / mean)

    def get_pressure(self, target_dc_id, *, include_local=False):
        capacity = self.capacities[target_dc_id]
        mean = self._residence.get(target_dc_id, self.initial_residence_s)
        selected = [(key, r) for key, r in self._active.items() if r[1] == target_dc_id
                    and (include_local or key not in self._local_events)]
        return tuple(background + sum(
            r[index + 2] * self.remaining_service(key, r, mean) / self.initial_residence_s
            for key, r in selected) / cap if cap > 0 else 0.0
            for index, cap, background in ((0, capacity.cpu, self.background_cpu),
                                            (1, capacity.gpu, self.background_gpu)))

    def get_congestion_cost(self, target_dc_id, cpu_request=0.0, gpu_request=0.0, *, include_local=False, service_s=None):
        if any(not math.isfinite(value) or value < 0 for value in (cpu_request, gpu_request)):
            raise ValueError('Resource demand must be finite and nonnegative')
        capacity = self.capacities[target_dc_id]
        if not capacity.fits(cpu_request, gpu_request):
            return math.inf
        if service_s is not None and (not math.isfinite(service_s) or service_s <= 0):
            raise ValueError('Candidate service duration must be finite and positive')
        work_scale = service_s / self.initial_residence_s if service_s is not None else 1.0
        return sum(weight * (self.linear_cost_weight * delta + self.quadratic_cost_weight * (2*x*delta+delta*delta))
                   for x, delta, weight in zip(self.get_pressure(target_dc_id, include_local=include_local),
                       (work_scale*cpu_request/capacity.cpu if capacity.cpu else 0., work_scale*gpu_request/capacity.gpu if capacity.gpu else 0.), self.weights))

    def snapshot(self):
        rows = super().snapshot()
        for dc, row in rows.items():
            row['active_commitment_count'] = sum(r[1] == dc for r in self._active.values())
            row['estimated_residence_s'] = self._residence.get(dc, self.initial_residence_s)
            flow = [record for key, record in self._records.items() if record[1] == dc and key not in self._local_events]
            row['recent_selection_count'] = len(flow)
            capacity = self.capacities[dc]
            row['historical_cpu_pressure'], row['historical_gpu_pressure'] = (
                sum(r[index+2] for r in flow)/cap if cap else 0.
                for index, cap in enumerate((capacity.cpu, capacity.gpu)))
        return rows


class TransferLifecycle:
    """One successor and one terminal result per executed edge transfer."""
    def __init__(self):
        self.records = {}
        self._jobs = {}

    def register(self, pending):
        if pending.event_id in self.records:
            return
        self._jobs.setdefault(pending.job_id, []).append(pending.event_id)
        self.records.setdefault(pending.event_id, dict(
            job_id=pending.job_id, source=pending.source_dc_id, target=pending.target_dc_id,
            routed_at=pending.routed_at, successor=None, admitted_at=None,
            started_at=None, terminal=None, terminal_at=None, local_queue_wait_s=None))

    def successor(self, event_id, result, now):
        row = self.records[event_id]
        if row['successor'] is not None:
            return False
        row['successor'] = result
        if result in {'started', 'queued'}:
            row['admitted_at'] = float(now)
        if result == 'started':
            row['started_at'] = float(now)
            row['local_queue_wait_s'] = 0.0
        return True

    def start(self, job_id, dc_id, now):
        for key in self._jobs.get(str(job_id), ()):
            row = self.records[key]
            if row['job_id'] == str(job_id) and row['target'] == str(dc_id) and row['successor'] in {'queued', 'started'} and row['started_at'] is None:
                if float(now) < row['admitted_at']:
                    raise ValueError('Service start precedes local admission')
                row['started_at'] = float(now)
                row['local_queue_wait_s'] = float(now) - row['admitted_at']

    def close(self, trace, job, sla):
        host = getattr(trace, 'host_step', None)
        start = getattr(job, 'start_time', None)
        for key in self._jobs.get(str(trace.job_id), ()):
            row = self.records[key]
            if row['job_id'] != str(trace.job_id) or row['terminal'] is not None:
                continue
            local = host is not None and str(host.dc_id) == row['target'] and row['successor'] in {'queued', 'started'}
            started = local and start is not None and float(start) >= row['admitted_at'] - 1e-9
            if started:
                row['started_at'] = float(start)
                row['local_queue_wait_s'] = max(float(start) - row['admitted_at'], 0.0)
            completed = str(trace.terminal_reason) == 'completed'
            row['terminal'] = ('completed_on_time' if sla else 'completed_late') if local and completed else (
                'dropped_after_start' if started else 'dropped_before_start') if local else 'forwarded_terminal'
            row['chain_success'] = completed
            row['terminal_at'] = float(trace.terminal_time)

    def metrics(self, dc=None):
        rows = [r for r in self.records.values() if dc is None or r['target'] == dc]
        counts = Counter(r['successor'] for r in rows)
        admitted = [r for r in rows if r['successor'] in {'started', 'queued'}]
        queued = [r for r in admitted if r['successor'] == 'queued']
        closed = [r for r in admitted if r['terminal'] is not None]
        effective = sum(r['terminal'] == 'completed_on_time' for r in closed)
        waits = [r['local_queue_wait_s'] for r in admitted if r['local_queue_wait_s'] is not None]
        return dict(
            guidance_schema_version=5, transfer_count=len(rows),
            successor_started_count=counts['started'], successor_queued_count=counts['queued'],
            successor_edge_count=counts['edge'], successor_cloud_count=counts['cloud'],
            successor_drop_count=counts['drop'], successor_pending_count=counts[None],
            local_accepted_count=len(admitted), local_terminal_count=len(closed),
            local_pending_count=len(admitted)-len(closed),
            legacy_absorption_semantics='host_admitted_including_queued',
            incoming_edge_accepted_count=len(admitted),
            incoming_edge_effective_local_completion_count=effective,
            incoming_edge_effective_local_completion_rate=effective/len(closed) if closed else None,
            queue_started_count=sum(r['started_at'] is not None for r in queued),
            queue_completed_on_time_count=sum(r['terminal'] == 'completed_on_time' for r in queued),
            queue_terminal_count=sum(r['terminal'] is not None for r in queued),
            queue_pending_count=sum(r['terminal'] is None for r in queued),
            local_completed_on_time_count=effective,
            local_completed_late_count=sum(r['terminal'] == 'completed_late' for r in closed),
            local_dropped_before_start_count=sum(r['terminal'] == 'dropped_before_start' for r in closed),
            local_dropped_after_start_count=sum(r['terminal'] == 'dropped_after_start' for r in closed),
            effective_local_completion_rate=effective/len(closed) if closed else None,
            queue_sla_rate=sum(r['terminal'] == 'completed_on_time' for r in queued)/sum(r['terminal'] is not None for r in queued)
                if any(r['terminal'] is not None for r in queued) else None,
            local_queue_wait_samples=len(waits), local_queue_wait_mean_s=sum(waits)/len(waits) if waits else None)
