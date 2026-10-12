"""Independent Host SAC pretraining with per-DC validation and final publication.

Run: python -m schedulers.pretrain_host --help
Only a final, validated bundle is eligible for Routing training.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, fields, replace
import importlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

import config as conf
from environment.cloud_edge_env import CloudEdgeEnv
from schedulers.shared_host import (
    HostObservationBuilder, HostSACConfig, LocalHostSAC, atomic_json,
    export_environment, host_structure, parameter_digest, project_path,
    read_manifest, save_host_bundle,
)

support = importlib.import_module("schedulers.H-MASAC.training_support")


@dataclass(frozen=True)
class HostPretrainConfig:
    output_dir: str = "model/shared_host"
    old_env_path: str | None = None
    resume: str | None = None
    seed: int = conf.Seed
    device: str = conf.DEVICE
    min_episodes: int = 1000
    max_episodes: int = 5000
    min_samples_per_dc: int = 20000
    min_updates_per_dc: int = 5000
    eval_interval: int = 50
    eval_seeds: tuple[int, ...] = (1000001, 1000002, 1000003, 1000004, 1000005)
    load_factors: tuple[float, ...] = (0.5, 1.0, 1.5)
    stable_windows: int = 5
    confirmation_evaluations: int = 2
    reward_tolerance: float = 0.03
    rate_tolerance: float = 0.02
    cost_tolerance: float = 0.05
    min_reward_improvement: float = 0.02
    min_completion_rate: float = 0.5
    min_sla_rate: float = 0.3
    checkpoint_interval: int = 50
    replay_capacity: int = conf.HOST_REPLAY_CAPACITY
    batch_size: int = conf.HOST_BATCH_SIZE
    random_warmup_steps: int = conf.HOST_RANDOM_WARMUP_STEPS
    learning_starts: int = conf.HOST_LEARNING_STARTS
    train_every: int = conf.HOST_TRAIN_EVERY
    updates_per_train: int = conf.HOST_UPDATES_PER_TRAIN
    save_replay: bool = False

    def validate(self):
        for name in ("min_episodes", "max_episodes", "min_samples_per_dc", "min_updates_per_dc",
                     "eval_interval", "stable_windows", "confirmation_evaluations", "checkpoint_interval",
                     "batch_size", "learning_starts", "train_every", "updates_per_train"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须大于零")
        if self.min_episodes > self.max_episodes or self.stable_windows < 2:
            raise ValueError("训练轮数或稳定窗口配置不合法")
        if self.replay_capacity < self.batch_size or self.random_warmup_steps < 0:
            raise ValueError("Host replay/warmup 配置不合法")
        if not self.eval_seeds or len(set(self.eval_seeds)) != len(self.eval_seeds):
            raise ValueError("验证种子必须非空且互不重复")
        train_seeds = range(self.seed, self.seed + self.max_episodes)
        if any(seed in train_seeds for seed in self.eval_seeds):
            raise ValueError("训练种子与验证种子不能重叠")
        if not self.load_factors or any(value <= 0 or not np.isfinite(value) for value in self.load_factors):
            raise ValueError("负载系数必须为有限正数")
        for name in ("reward_tolerance", "rate_tolerance", "cost_tolerance", "min_reward_improvement"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} 必须位于 [0, 1)")
        if not 0 <= self.min_completion_rate <= 1 or not 0 <= self.min_sla_rate <= 1:
            raise ValueError("服务指标阈值必须位于 [0, 1]")


def reward_model(env):
    options = {
        "task_completion_reward": conf.TASK_COMPLETION_REWARD,
        "completion_time_cost_weight": conf.COMPLETION_TIME_COST_WEIGHT,
        "sla_violation_cost_weight": conf.SLA_VIOLATION_COST_WEIGHT,
        "remote_offload_base_penalty": conf.REMOTE_OFFLOAD_BASE_PENALTY,
        "remote_latency_cost_weight": conf.REMOTE_LATENCY_COST_WEIGHT,
        "sla_risk_cost_weight": conf.SLA_RISK_COST_WEIGHT,
        "timeout_drop_penalty": conf.TIMEOUT_DROP_PENALTY,
        "resource_drop_penalty": conf.RESOURCE_DROP_PENALTY,
        "energy_normalization_j": conf.ENERGY_NORMALIZATION_J,
        "energy_cost_weight": conf.ENERGY_COST_WEIGHT,
        "max_latency_s": env.max_latency, "max_job_duration_s": env.max_job_duration,
        "sla_deadline_ratio": env.sla_deadline_ratio, "drop_deadline_ratio": env.drop_deadline_ratio,
        "norm_eps": env.norm_eps,
    }
    return support.HMasacTrainingRewardModel(support.TrainingRewardConfig(**options))


class TerminalRecorder:
    def __init__(self, dc_id, replay=None):
        self.dc_id, self.replay = dc_id, replay
        self.transitions = []

    def add(self, transition):
        if transition.dc_id != self.dc_id:
            raise ValueError("Host 终局记录 DC 错误")
        self.transitions.append(transition)
        if self.replay is not None:
            self.replay.add(transition)


class DiscardRoutingReplay:
    def add(self, transition):
        pass


def run_host_episode(env, agents, settings, seed, *, replay=None, counters=None, rngs=None):
    """Reuse existing causal traces/rewards; Routing always takes Self and never learns."""
    training = replay is not None
    env.reset(seed=seed)
    builder = HostObservationBuilder(env)
    traces = support.PendingJobTraceStore()
    feedback = support.NeighborHistoricalFeedbackStore(env)
    routing_obs = support.RoutingObservationBuilder(env, use_neighbor_historical_feedback=False,
                                                    neighbor_feedback_provider=feedback)
    routing_state = support.RoutingCentralizedStateBuilder(env, routing_obs)
    model = reward_model(env)
    collector = support.TransitionCollector(env, routing_obs, routing_state, traces, model)
    collector.reset_episode()
    stats = support.EpisodeStatistics(episode=0, episode_seed=seed,
                                     per_agent_returns={dc: 0.0 for dc in env.edge_dc_ids},
                                     training_stage="host_pretrain" if training else "host_validation")
    recorders = {dc: TerminalRecorder(dc, replay[dc] if training else None) for dc in agents}
    arguments = dict(env=env, pending_trace_store=traces, routing_replay_buffer=DiscardRoutingReplay(),
                     host_replay_buffers=recorders, stats=stats, neighbor_feedback_store=feedback,
                     collect_neighbor_historical_feedback=False, training_reward_model=model)
    for agent in agents.values():
        agent.train_mode() if training else agent.eval_mode()
    decision = None
    while env.agents:
        if env.has_pending_host_decision():
            context = env.get_pending_host_decision()
            dc, job = str(context["dc_id"]), str(context["job_id"])
            obs, when = builder.build(dc_id=dc, job_id=job), float(env.current_time)
            source = "policy"
            if training and counters[dc] < settings.random_warmup_steps:
                action = int(rngs[dc].integers(agents[dc].action_dim))
                source = "random"
            else:
                action = agents[dc].select_action(obs, deterministic=not training)
            result = env.execute_pending_host_action(host_action=action)
            if (str(result["job_id"]), str(result["dc_id"]), int(result["host_action"])) != (job, dc, action):
                raise RuntimeError("Host 执行结果与决策不一致")
            traces.record_host_step(job_id=job, dc_id=dc, env_time=when, host_obs=obs,
                                    action=action, host_id=str(result["host_id"]), action_source=source)
            traces.record_host_result(job_id=job, result=str(result["execution_result"]),
                                      env_time=float(result["env_time"]))
            stats.record_host_decision(dc_id=dc, action_source=source)
            stats.record_host_result(dc_id=dc, execution_result=str(result["execution_result"]))
            support.consume_environment_outcome_events(**arguments)
            if training:
                counters[dc] += 1
                if (counters[dc] >= settings.learning_starts and counters[dc] % settings.train_every == 0
                        and replay[dc].can_sample(settings.batch_size)):
                    for _ in range(settings.updates_per_train):
                        agents[dc].update(replay[dc], settings.batch_size)
            decision = None
            continue
        if collector.drain_one_dead_agent():
            decision = None
            continue
        if decision is None:
            decision = collector.capture_decision()
        action = decision.forced_action if decision.forced_action is not None else support.get_self_routing_action(env, decision.agent_id)
        result, decision = collector.execute_and_record(decision, action,
                                                        "forced" if decision.forced_action is not None else "orchestrator")
        stats.record_routing_decision(agent_id=result.agent_id, reward=result.immediate_reward,
                                      action_type=result.action_type, action_source=result.action_source,
                                      target_dc_id=result.target_dc_id)
        if result.job_finalized:
            finalized = traces.get_finalized_trace(result.job_id)
            support.flush_finalized_trace_to_replay(finalized, arguments["routing_replay_buffer"],
                                                     recorders, stats, feedback, False)
            traces.pop_finalized_trace(result.job_id)
        support.consume_environment_outcome_events(**arguments)
    support.consume_environment_outcome_events(**arguments)
    traces.assert_no_open_trace()
    traces.assert_no_unflushed_finalized_trace()
    metrics = {}
    for dc, recorder in recorders.items():
        transitions = recorder.transitions
        completed = [env.job_map[t.job_id] for t in transitions if env.job_map[t.job_id].finish_time is not None]
        delays = [float(job.finish_time - job.arrive_time) for job in completed]
        energy = sum(float(getattr(env.job_map[t.job_id], "compute_energy_j", 0.0)) for t in transitions)
        count = len(transitions)
        metrics[dc] = {"samples": count, "reward": sum(t.reward for t in transitions) / max(count, 1),
                       "completion_rate": len(completed) / max(count, 1),
                       "sla_rate": sum(delay <= env.sla_deadline_ratio * job.duration
                                       for delay, job in zip(delays, completed)) / max(count, 1),
                       "completion_time": float(np.mean(delays)) if delays else 0.0,
                       "energy_per_job": energy / max(count, 1)}
    return metrics


def evaluate_hosts(env, agents, settings):
    # Evaluation cannot mutate training RNGs, networks, replay, or counters.
    before = {dc: parameter_digest(agent) for dc, agent in agents.items()}
    previous_modes = {dc: agent.actor.training for dc, agent in agents.items()}
    base_rate = env.lambda_rate
    results = {}
    try:
        for factor in settings.load_factors:
            env.lambda_rate = base_rate * factor
            env._configure_dc_arrival_profile()
            for seed in settings.eval_seeds:
                metrics = run_host_episode(env, agents, settings, seed)
                print(f"Host validation | load={factor:g} | seed={seed} | "
                      f"min_completion={min(row['completion_rate'] for row in metrics.values()):.3f} | "
                      f"min_sla={min(row['sla_rate'] for row in metrics.values()):.3f}", flush=True)
                for dc, row in metrics.items():
                    results.setdefault(f"{dc}@{factor:g}", []).append(row)
    finally:
        env.lambda_rate = base_rate
        env._configure_dc_arrival_profile()
        for dc, agent in agents.items():
            agent.train_mode() if previous_modes[dc] else agent.eval_mode()
    if before != {dc: parameter_digest(agent) for dc, agent in agents.items()}:
        raise RuntimeError("Host 验证期间模型发生变化")
    return {key: {name: float(np.mean([row[name] for row in rows])) for name in rows[0]}
            for key, rows in results.items()}


def convergence_passed(history, baseline, settings, episode, samples, updates):
    if not baseline or not samples or set(samples) != set(updates):
        return False
    if episode < settings.min_episodes or len(history) < settings.stable_windows:
        return False
    if min(samples.values()) < settings.min_samples_per_dc or min(updates.values()) < settings.min_updates_per_dc:
        return False
    windows = history[-settings.stable_windows:]
    if any(set(window) != set(baseline) for window in windows):
        return False
    for key in baseline:
        rows = [window[key] for window in windows]
        if any(row["samples"] <= 0 or any(not np.isfinite(value) for value in row.values()) for row in rows):
            return False
        for name in ("reward", "completion_rate", "sla_rate", "completion_time", "energy_per_job"):
            values = np.array([row[name] for row in rows])
            scale = max(abs(float(values.mean())), 1.0)
            tolerance = (settings.rate_tolerance if name.endswith("rate") else
                         settings.reward_tolerance if name == "reward" else settings.cost_tolerance)
            if np.ptp(values) > tolerance * (1.0 if name.endswith("rate") else scale):
                return False
        last = rows[-1]
        if (last["reward"] < baseline[key]["reward"] + settings.min_reward_improvement
                or last["completion_rate"] < max(settings.min_completion_rate, baseline[key]["completion_rate"] - settings.rate_tolerance)
                or last["sla_rate"] < max(settings.min_sla_rate, baseline[key]["sla_rate"] - settings.rate_tolerance)):
            return False
    return True


def capture_training_state(settings, episode, agents, replay, counters, rngs, history, baseline, confirmations):
    state = {"episode": episode, "counters": counters.copy(), "history": history,
             "baseline": baseline, "confirmations": confirmations,
             "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
             "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
             "action_rng": {dc: rng.bit_generator.state for dc, rng in rngs.items()},
             "samples": {dc: buffer.total_added for dc, buffer in replay.items()},
             "replay_policy": "restore" if settings.save_replay else "empty_on_resume"}
    if settings.save_replay:
        state["replay"] = {dc: copy.deepcopy(vars(buffer)) for dc, buffer in replay.items()}
    return state


def train(settings=None, *, env=None, agent_config=None):
    settings = settings or HostPretrainConfig()
    settings.validate()
    output = project_path(settings.output_dir).resolve()
    if (output / "final").exists():
        raise FileExistsError("正式 Host final 已存在；新实验请使用独立 output_dir")
    if not settings.resume and ((output / "latest.json").exists() or (output / "train_log.jsonl").exists()):
        raise FileExistsError("输出目录已有训练记录；请使用 --resume 或独立 output_dir")
    directory = metadata = None
    if settings.resume:
        directory, metadata = read_manifest(settings.resume, require_converged=False)
    support.set_global_random_seeds(settings.seed, device=settings.device)
    environment = env or CloudEdgeEnv(old_env_path=str(directory / "environment") if directory else settings.old_env_path,
                                      seed=settings.seed)
    output.mkdir(parents=True, exist_ok=True)
    export_environment(environment, output / "environment")
    # Separate objects for validation, with the same static hardware.
    validation_env = CloudEdgeEnv(env_source=copy.deepcopy(environment.env_source), seed=settings.eval_seeds[0])
    base_rate = float(environment.lambda_rate)
    validation_env.lambda_rate = base_rate
    validation_env._configure_dc_arrival_profile()
    builder = HostObservationBuilder(environment)
    if agent_config is None:
        if directory:
            first_dc = builder.edge_dc_ids[0]
            saved = torch.load(directory / "hosts" / f"{first_dc}.pt", map_location="cpu", weights_only=False)
            agent_config = HostSACConfig(**{**saved["config"], "device": settings.device,
                                           "allow_cpu": settings.device == "cpu"})
        else:
            options = {item.name: getattr(conf, f"HOST_{item.name.upper()}") for item in fields(HostSACConfig)
                       if item.name not in {"seed", "device", "allow_cpu"}}
            agent_config = HostSACConfig(**options, seed=settings.seed, device=settings.device,
                                         allow_cpu=settings.device == "cpu")
    agents = {dc: LocalHostSAC(builder.get_obs_dim(dc), builder.get_action_dim(dc), agent_config)
              for dc in builder.edge_dc_ids}
    replay = {dc: support.HostReplayBuffer(dc_id=dc, capacity=settings.replay_capacity,
                                          obs_dim=agent.obs_dim, action_dim=agent.action_dim,
                                          seed=settings.seed + index + 20000)
              for index, (dc, agent) in enumerate(agents.items())}
    counters = {dc: 0 for dc in agents}
    rngs = {dc: np.random.default_rng(settings.seed + index + 30000) for index, dc in enumerate(agents)}
    episode, history, confirmations = 0, [], 0
    samples_before_resume = {dc: 0 for dc in agents}
    baseline = None
    if directory:
        if metadata["structure"] != host_structure(environment):
            raise ValueError("Host 续训环境不兼容")
        if metadata["training"]["reward_config"] != asdict(reward_model(environment).config):
            raise ValueError("Host 续训奖励配置发生变化")
        saved_settings = metadata["training"]["settings"]
        mutable = {"max_episodes", "output_dir", "resume", "device", "checkpoint_interval", "save_replay"}
        current = json.loads(json.dumps(asdict(settings)))
        if any(current[name] != value for name, value in saved_settings.items() if name not in mutable):
            raise ValueError("Host 续训不允许修改验证集、学习配置或收敛标准")
        for dc, agent in agents.items():
            saved = torch.load(directory / "hosts" / f"{dc}.pt", map_location="cpu", weights_only=False)
            requested = asdict(agent.config)
            if any(requested[key] != value for key, value in saved["config"].items()
                   if key not in {"device", "allow_cpu"}):
                raise ValueError(f"Host 续训网络/学习配置不兼容：{dc}")
            agent.load(directory / "hosts" / f"{dc}.pt", load_optimizers=True)
        state = torch.load(directory / "training_state.pt", map_location="cpu", weights_only=False)
        episode, counters = state["episode"], state["counters"]
        history, baseline, confirmations = state["history"], state["baseline"], state["confirmations"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        for dc, rng in rngs.items():
            rng.bit_generator.state = state["action_rng"][dc]
        if "replay" in state:
            for dc, buffer in replay.items():
                vars(buffer).update(state["replay"][dc])
        else:
            samples_before_resume = state["samples"]
    if baseline is None:
        atomic_json({"phase": "baseline_validation", "episode": episode,
                     "settings": asdict(settings)}, output / "progress.json")
        print("Host baseline validation started", flush=True)
        baseline = evaluate_hosts(validation_env, agents, settings)
    def save(name, status):
        training = {"settings": asdict(settings), "episode": episode,
                    "reward_config": asdict(reward_model(environment).config),
                    "baseline": baseline, "last_evaluation": history[-1] if history else None,
                    "samples": {dc: samples_before_resume[dc] + replay[dc].total_added for dc in agents},
                    "updates": {dc: agent.update_step for dc, agent in agents.items()},
                    "replay_policy": "restore" if settings.save_replay else "empty_on_resume"}
        state = capture_training_state(settings, episode, agents, replay, counters, rngs, history, baseline, confirmations)
        state["samples"] = training["samples"]
        target = save_host_bundle(agents, environment, output / name, status=status,
                                  training=training, resume_state=state)
        atomic_json({"checkpoint": str(target), "status": status, "episode": episode}, output / "latest.json")
        return target
    converged = False
    completed_episode = episode
    started = time.perf_counter()
    if directory is None:
        save("initial", "training")
    try:
        for episode in range(episode + 1, settings.max_episodes + 1):
            factor = settings.load_factors[(episode - 1) % len(settings.load_factors)]
            environment.lambda_rate = base_rate * factor
            environment._configure_dc_arrival_profile()
            metrics = run_host_episode(environment, agents, settings, settings.seed + episode - 1,
                                       replay=replay, counters=counters, rngs=rngs)
            completed_episode = episode
            samples = {dc: samples_before_resume[dc] + replay[dc].total_added for dc in agents}
            updates = {dc: agent.update_step for dc, agent in agents.items()}
            record = {"episode": episode, "load_factor": factor, "samples": samples, "updates": updates,
                      "metrics": metrics, "elapsed_seconds": time.perf_counter() - started}
            if episode % settings.eval_interval == 0:
                evaluation = evaluate_hosts(validation_env, agents, settings)
                history.append(evaluation)
                passed = convergence_passed(history, baseline, settings, episode, samples, updates)
                confirmations = confirmations + 1 if passed else 0
                converged = confirmations > settings.confirmation_evaluations
                record.update(evaluation=evaluation, stable=passed, confirmations=confirmations)
            with (output / "train_log.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
            atomic_json(record, output / "progress.json")
            print(f"Host episode {episode}/{settings.max_episodes} | min_samples={min(samples.values())} "
                  f"| min_updates={min(updates.values())} | confirmations={confirmations} | converged={converged}", flush=True)
            if converged:
                # Current weights, after confirmation training, are the final result.
                target = save("final", "converged")
                print(f"Host 收敛验收通过，最终 checkpoint：{target}", flush=True)
                return target
            if episode % settings.checkpoint_interval == 0:
                save(f"episode_{episode:06d}", "training")
        return save(f"unconverged_{episode:06d}", "unconverged")
    except (KeyboardInterrupt, Exception):
        # Do not claim a mid-episode state is an exact resumable episode boundary.
        if episode == completed_episode:
            save(f"interrupted_{episode:06d}_{time.time_ns()}", "interrupted")
        raise
    finally:
        environment.close()
        validation_env.close()


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    for item in fields(HostPretrainConfig):
        option = "--" + item.name.replace("_", "-")
        if item.name in {"eval_seeds", "load_factors"}:
            parser.add_argument(option, nargs="+", type=int if item.name == "eval_seeds" else float)
        elif item.name == "save_replay":
            parser.add_argument(option, action="store_true", default=None)
        else:
            parser.add_argument(option, type=type(item.default) if item.default is not None else str)
    values = {name: value for name, value in vars(parser.parse_args()).items() if value is not None}
    if values.get("resume"):
        directory, metadata = read_manifest(values["resume"], require_converged=False)
        inherited = metadata["training"]["settings"].copy()
        inherited.update(values)
        if "output_dir" not in values:
            inherited["output_dir"] = str(directory.parent)
        values = inherited
    for name in ("eval_seeds", "load_factors"):
        if name in values:
            values[name] = tuple(values[name])
    target = train(HostPretrainConfig(**values))
    _, metadata = read_manifest(target, require_converged=False)
    if metadata["status"] != "converged":
        raise SystemExit("达到训练上限但尚未通过收敛验收；已保存 checkpoint，可用 --resume 继续。")


if __name__ == "__main__":
    main()
