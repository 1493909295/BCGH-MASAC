"""Evaluate OPT with the same global observations and online historical feedback."""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "schedulers.OPT"

import torch
import numpy as np
import config as conf
from ._base import (
    trainer, RoutingMASACConfig, HostSACConfig, LocalHostSAC,
    HostObservationBuilder, RoutingCentralizedStateBuilder,
    NeighborHistoricalFeedbackStore, PendingJobTraceStore, TransitionCollector,
    TrainingRewardConfig, TrainingRewardModel,
)
from .opt_agent import OPTRoutingMASAC
from .routing_observation import RoutingObservationBuilder
from .train_opt import TrainConfig, project_path, validate_opt_device


def build_observers(env, settings):
    if settings.use_neighbor_historical_feedback and not settings.collect_neighbor_historical_feedback:
        raise ValueError("Using neighbor feedback requires collecting it")
    feedback = NeighborHistoricalFeedbackStore(
        env, ewma_alpha=settings.neighbor_feedback_ewma_alpha,
        age_scale_samples=settings.neighbor_feedback_age_scale_samples,
        confidence_scale_samples=settings.neighbor_feedback_confidence_scale_samples,
    )
    observation = RoutingObservationBuilder(
        env, settings.use_neighbor_historical_feedback, feedback
    )
    state = RoutingCentralizedStateBuilder(env, observation)
    return observation, state, HostObservationBuilder(env), feedback


def load_models(env, checkpoint, *, device=None):
    """Load a complete OPT checkpoint against a caller-supplied matching topology."""
    path = project_path(checkpoint)
    metadata = json.loads(trainer.checkpoint_state_path(path).read_text(encoding="utf-8"))
    if metadata.get("architecture") != OPTRoutingMASAC.checkpoint_architecture:
        raise RuntimeError("Evaluation requires an OPT checkpoint")
    if "scheduler_config" not in metadata:
        raise RuntimeError("OPT checkpoint is missing its evaluation settings")
    settings = TrainConfig(**metadata["scheduler_config"])
    obs, state, host_obs, _ = build_observers(env, settings)
    routing_data = torch.load(path, map_location="cpu", weights_only=False)
    routing_config = dict(routing_data["config"])
    selected_device = validate_opt_device(device or routing_config.get("device") or conf.DEVICE)
    routing_config["device"] = selected_device
    routing_config["allow_cpu"] = selected_device == "cpu"
    routing = OPTRoutingMASAC(
        obs.obs_dim, state.state_dim, env.action_dim, len(env.edge_dc_ids),
        config=RoutingMASACConfig(**routing_config),
    )
    routing.observation_metadata = obs.checkpoint_metadata()
    host_dir = trainer.host_checkpoint_dir(path)
    hosts = {}
    for dc_id in env.edge_dc_ids:
        host_data = torch.load(host_dir / f"{dc_id}.pt", map_location="cpu", weights_only=False)
        host_config = dict(host_data["config"])
        host_config["device"] = selected_device
        host_config["allow_cpu"] = selected_device == "cpu"
        hosts[dc_id] = LocalHostSAC(
            host_obs.get_obs_dim(dc_id), host_obs.get_action_dim(dc_id),
            config=HostSACConfig(**host_config),
        )
    trainer.validate_checkpoint_structure_metadata(metadata, env, routing, hosts)
    routing.load(path, load_optimizers=False)
    for dc_id, agent in hosts.items():
        agent.load(host_dir / f"{dc_id}.pt", load_optimizers=False)
    return routing, hosts, settings


def evaluate_episode(env, routing_agent, host_agents, settings, *, seed=42, deterministic=True):
    """Fresh episode and feedback store; no replay, optimizer updates or checkpoint writes."""
    obs, state, host_obs, feedback = build_observers(env, settings)
    if not isinstance(routing_agent, OPTRoutingMASAC):
        raise ValueError("Expected an OPT routing agent")
    if (routing_agent.local_obs_dim != obs.obs_dim
            or routing_agent.global_state_dim != state.state_dim
            or routing_agent.action_dim != env.action_dim
            or routing_agent.num_agents != len(env.edge_dc_ids)
            or routing_agent.observation_metadata != obs.checkpoint_metadata()):
        raise ValueError("OPT model does not match the evaluation observation schema")
    if set(host_agents) != set(env.edge_dc_ids):
        raise ValueError("Host agents do not match the evaluation DCs")
    for dc_id, agent in host_agents.items():
        if (agent.obs_dim != host_obs.get_obs_dim(dc_id)
                or agent.action_dim != host_obs.get_action_dim(dc_id)):
            raise ValueError(f"Host model dimensions mismatch: {dc_id}")
        agent.eval_mode()
    routing_agent.eval_mode()
    trainer.set_global_random_seeds(seed, device=str(routing_agent.device))
    reward_values = {f.name: getattr(conf, f.name.upper()) for f in fields(TrainingRewardConfig)
                     if hasattr(conf, f.name.upper())}
    reward_values.update(max_latency_s=env.max_latency, max_job_duration_s=env.max_job_duration,
                         sla_deadline_ratio=env.sla_deadline_ratio,
                         drop_deadline_ratio=env.drop_deadline_ratio, norm_eps=env.norm_eps)
    reward = TrainingRewardModel(TrainingRewardConfig(**reward_values))
    traces = PendingJobTraceStore()
    collector = TransitionCollector(env, obs, state, traces, reward)
    env.reset(seed=seed)
    routing_count = host_count = 0
    total_reward = 0.0

    def finalize(trace):
        if settings.collect_neighbor_historical_feedback:
            feedback.update_from_finalized_trace(trace)
        traces.pop_finalized_trace(trace.job_id)

    def consume_outcomes():
        nonlocal total_reward
        for event in env.pop_job_outcome_events():
            delta = reward.calculate_outcome_reward(event)
            total_reward += delta
            traces.record_reward_event(
                job_id=str(event["job_id"]), env_time=float(event["env_time"]),
                reward_delta=delta, reason=str(event["reason"]),
                terminal=bool(event.get("terminal", False)),
            )
            if event.get("terminal", False):
                finalize(traces.finalize_terminal_trace(str(event["job_id"])))

    with torch.no_grad():
        while env.agents:
            if env.has_pending_host_decision():
                context = env.get_pending_host_decision()
                dc_id, job_id = str(context["dc_id"]), str(context["job_id"])
                now = float(env.current_time)
                local_obs = host_obs.build(dc_id=dc_id, job_id=job_id)
                action = host_agents[dc_id].select_action(local_obs, deterministic=deterministic)
                result = env.execute_pending_host_action(action)
                traces.record_host_step(
                    job_id=job_id, dc_id=dc_id, env_time=now, host_obs=local_obs,
                    action=action, host_id=str(result["host_id"]), action_source="policy",
                )
                traces.record_host_result(job_id=job_id, result=result["execution_result"],
                                          env_time=float(result["env_time"]))
                host_count += 1
            elif collector.drain_one_dead_agent():
                continue
            else:
                decision = collector.capture_decision()
                if decision.forced_action is not None:
                    action, source = decision.forced_action, "forced"
                else:
                    action = routing_agent.select_action(
                        decision.local_obs, decision.agent_index, deterministic=deterministic
                    )
                    source = "policy"
                result, _ = collector.execute_and_record(decision, action, action_source=source)
                total_reward += result.immediate_reward
                routing_count += 1
                if result.job_finalized:
                    finalize(traces.get_finalized_trace(result.job_id))
            consume_outcomes()
        consume_outcomes()
    traces.assert_no_open_trace()
    traces.assert_no_unflushed_finalized_trace()
    completion_times = [float(job.get_turnaround_time()) for job in env.jobs
                        if job.finish_time is not None]
    edge_hops = [int(job.routing_hop_count) for job in env.jobs]
    return {
        "scheduler": "OPT", "seed": seed, "deterministic": deterministic,
        "routing_decisions": routing_count, "host_decisions": host_count,
        "episode_return": float(total_reward),
        "total_jobs": len(env.jobs), "completed_jobs": len(completion_times),
        "dropped_jobs": len(env.dropped_jobs_info),
        "completion_rate": len(completion_times) / max(len(env.jobs), 1),
        "p95_completion_time": float(np.percentile(completion_times, 95)) if completion_times else 0.0,
        "edge_transfer_count": sum(edge_hops),
        "p95_routing_edge_hops": float(np.percentile(edge_hops, 95)) if edge_hops else 0.0,
        **trainer.calculate_service_metrics(env),
        **trainer.calculate_episode_energy_metrics(env),
        "load_metrics": trainer.calculate_episode_load_metrics(env),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a trained OPT scheduler")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--old-env", required=True, help="Matching saved environment directory")
    parser.add_argument("--device", default=None, help="Override checkpoint device, e.g. cpu or cuda:0")
    parser.add_argument("--seed", type=int, default=conf.Seed)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--output", default="result/OPT/evaluation.json")
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    # This topology seed only needs CPU RNGs; the checkpoint decides the model device.
    trainer.set_global_random_seeds(args.seed, device="cpu")
    env = trainer.build_environment(args.seed, str(project_path(args.old_env)))
    try:
        routing, hosts, settings = load_models(env, args.checkpoint, device=args.device)
        results = [evaluate_episode(env, routing, hosts, settings,
                                    seed=args.seed + index, deterministic=not args.stochastic)
                   for index in range(args.episodes)]
        output = project_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        trainer.atomic_checkpoint_text_write(
            json.dumps(results, ensure_ascii=False, indent=2), output
        )
        print(f"OPT evaluation saved: {output}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
