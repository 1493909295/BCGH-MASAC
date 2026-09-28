"""Evaluate loaded BGH agents with live short-window guidance (lambda=0.3 by default).

Call evaluate_episode with the matching environment topology, loaded RoutingMASAC,
host agents and TrainConfig restored from checkpoint bgh_guidance.parameters.
No replay, gradient update, checkpoint write or training warmup is performed.
"""
from dataclasses import fields

import config as conf
from bayesian_evidence import build_bayesian_evidence_from_finalized_trace
from guided_policy import GuidedRoutingPolicy
from host_observation import HostObservationBuilder
from neighbor_feedback import ShortWindowFeedbackStore
from pending_job_trace import PendingJobTraceStore
from routing_observation import RoutingObservationBuilder
from routing_centralized_state import RoutingCentralizedStateBuilder
from short_window_runtime import ShortWindowRuntime, build_short_window_game
from training_reward import HMasacTrainingRewardModel, TrainingRewardConfig
from transition_collector import TransitionCollector


def evaluate_episode(env, routing_agent, host_agents, settings, *, seed=42, deterministic=True):
    if not settings.enable_bayesian_game or not settings.enable_heuristic_guidance:
        raise ValueError("Full BGH evaluation requires Bayesian game and guidance enabled")
    runtime = ShortWindowRuntime(build_short_window_game(env, settings))
    feedback = ShortWindowFeedbackStore(env, window_s=settings.short_window_s,
                                      confidence_scale_samples=settings.bayesian_confidence_scale)
    routing_obs = RoutingObservationBuilder(env,
        use_neighbor_historical_feedback=settings.use_neighbor_historical_feedback,
        neighbor_feedback_provider=feedback)
    routing_state = RoutingCentralizedStateBuilder(env, routing_observation_builder=routing_obs)
    host_obs = HostObservationBuilder(env)
    if routing_agent.local_obs_dim != routing_obs.obs_dim or routing_agent.action_dim != env.action_dim:
        raise ValueError("Loaded actor does not match this environment")
    if set(host_agents) != set(env.edge_dc_ids):
        raise ValueError("Loaded host agents do not match edge DCs")
    for dc_id, agent in host_agents.items():
        if agent.obs_dim != host_obs.get_obs_dim(dc_id) or agent.action_dim != host_obs.get_action_dim(dc_id):
            raise ValueError(f"Host agent shape mismatch: {dc_id}")
    reward_values = {f.name: getattr(conf, f.name.upper()) for f in fields(TrainingRewardConfig)
                     if hasattr(conf, f.name.upper())}
    reward_values.update(max_latency_s=env.max_latency, max_job_duration_s=env.max_job_duration,
        sla_deadline_ratio=env.sla_deadline_ratio, drop_deadline_ratio=env.drop_deadline_ratio)
    reward = HMasacTrainingRewardModel(TrainingRewardConfig(**reward_values))
    traces = PendingJobTraceStore()
    collector = TransitionCollector(env, routing_obs, routing_state, traces, reward,
                                    short_window_runtime=runtime)
    policy = GuidedRoutingPolicy(routing_agent, env.routing_action_target_dc_ids,
                                 enabled=True, bayesian_game=runtime.game)
    env.reset(seed=seed)
    routing_count = 0
    source_evidence_decisions = 0

    def finalize(trace):
        if settings.collect_neighbor_historical_feedback:
            feedback.update_from_finalized_trace(trace)
        for item in build_bayesian_evidence_from_finalized_trace(trace, env):
            runtime.game.update_belief(item, update_clock=env.current_time)
        traces.pop_finalized_trace(trace.job_id)

    def consume_outcomes():
        for event in env.pop_job_outcome_events():
            traces.record_reward_event(job_id=str(event["job_id"]), env_time=float(event["env_time"]),
                reward_delta=reward.calculate_outcome_reward(event), reason=str(event["reason"]),
                terminal=bool(event.get("terminal", False)))
            if event.get("terminal", False):
                finalize(traces.finalize_terminal_trace(str(event["job_id"])))

    while env.agents:
        if env.has_pending_host_decision():
            ctx = env.get_pending_host_decision()
            dc, job_id, now = str(ctx["dc_id"]), str(ctx["job_id"]), float(env.current_time)
            obs = host_obs.build(dc_id=dc, job_id=job_id)
            action = host_agents[dc].select_action(obs, deterministic=deterministic)
            result = env.execute_pending_host_action(host_action=action)
            traces.record_host_step(job_id=job_id, dc_id=dc, env_time=now, host_obs=obs,
                action=action, host_id=str(result["host_id"]), action_source="policy")
            traces.record_host_result(job_id=job_id, result=result["execution_result"],
                                      env_time=float(result["env_time"]))
            runtime.record_host_result(job_id=job_id, dc_id=dc,
                execution_result=str(result["execution_result"]), now=float(result["env_time"]))
        elif collector.drain_one_dead_agent():
            continue
        else:
            decision = collector.capture_decision()
            if decision.forced_action is not None:
                action, source = decision.forced_action, "forced"
            else:
                context = runtime.build_action_context(policy, env, decision, feedback)
                action = policy.select_action(decision.local_obs, decision.agent_index, decision.agent_id,
                    action_context=context, guidance_lambda=settings.guidance_lambda_stage3_end,
                    deterministic=deterministic)
                source = "policy"
                source_evidence_decisions += int(any(x.source_signal > 0 for x in policy.last_breakdowns))
            result, _ = collector.execute_and_record(decision, action, action_source=source)
            routing_count += 1
            if result.job_finalized:
                finalize(traces.get_finalized_trace(result.job_id))
        consume_outcomes()
    consume_outcomes()
    traces.assert_no_open_trace()
    traces.assert_no_unflushed_finalized_trace()
    from train_bgh_masac import calculate_service_metrics
    return {**calculate_service_metrics(env), "guidance_lambda": settings.guidance_lambda_stage3_end,
            "routing_decisions": routing_count, "source_evidence_decisions": source_evidence_decisions,
            "absorption": runtime.absorption_summary()}
