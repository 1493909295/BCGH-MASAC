"""BCGH2-MASCA training entry point and three-stage orchestration."""

from __future__ import annotations

import sys
from pathlib import Path

if not __package__:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    __package__ = "schedulers.BCGH2-MASCA"

from .training_support import *
from .h_masac_agent import resolve_training_device


def train(
    train_config: TrainConfig,
    routing_masac_config: Optional[RoutingMASACConfig] = None,
    host_sac_config: Optional[HostSACConfig] = None,
) -> Tuple[
    RoutingMASAC,
    Dict[str, LocalHostSAC],
]:
    validate_training_stage_config(train_config)

    validate_bcgh_feature_config(train_config)
    if routing_masac_config is None:
        routing_masac_config = RoutingMASACConfig(
            gamma=conf.ROUTING_GAMMA,
            tau=conf.ROUTING_TAU,
            actor_lr=conf.ROUTING_ACTOR_LR,
            critic_lr=conf.ROUTING_CRITIC_LR,
            alpha_lr=conf.ROUTING_ALPHA_LR,
            actor_hidden_dim=conf.ROUTING_ACTOR_HIDDEN_DIM,
            critic_hidden_dim=conf.ROUTING_CRITIC_HIDDEN_DIM,
            initial_alpha=conf.ROUTING_INITIAL_ALPHA,
            target_entropy_ratio=conf.ROUTING_TARGET_ENTROPY_RATIO,
            max_grad_norm=conf.ROUTING_MAX_GRAD_NORM,
            policy_update_interval=conf.ROUTING_POLICY_UPDATE_INTERVAL,
            target_update_interval=conf.ROUTING_TARGET_UPDATE_INTERVAL,
            device=conf.DEVICE,
            seed=train_config.seed,
        )
    routing_masac_config = replace(
        routing_masac_config,
        enable_heuristic_guidance=train_config.enable_heuristic_guidance,
        enable_reward_shaping=train_config.enable_reward_shaping,
        enable_discount_reduction=train_config.enable_discount_reduction,
        heuristic_beta=train_config.heuristic_beta,
    )
    for agent_config in (routing_masac_config, host_sac_config):
        resolve_training_device(
            conf.DEVICE if agent_config is None else agent_config.device
        )

    bcgh_runtime_mode = get_bcgh_runtime_mode(train_config)

    print(
        "\n"
        "============================================================\n"
        "🧭 BCGH2-MASCA Runtime Configuration\n"
        f"Runtime mode              : {bcgh_runtime_mode}\n"
        f"Bayesian Game enabled     : "
        f"{train_config.enable_bayesian_game}\n"
        f"Congestion Game enabled   : "
        f"{train_config.enable_bayesian_game}\n"
        f"Heuristic Guidance enabled: "
        f"{train_config.enable_heuristic_guidance}\n"
        f"Cloud Action enabled      : "
        f"{bool(conf.ENABLE_CLOUD_ACTION)}\n"
        f"Zero-Diff Mode            : "
        f"{is_bcgh_zero_diff_mode(train_config)}\n"
        "============================================================\n",
        flush=True,
    )

    set_global_random_seeds(train_config.seed)

    env = build_environment(
        seed=train_config.seed,
        old_env_path=train_config.old_env_path,
    )

    print(
        "📥 DC Arrival Configuration | "
        f"mode={getattr(env, 'dc_arrival_mode', 'uniform')} | "
        f"global_lambda={float(getattr(env, 'lambda_rate', 0.0)):.6f} | "
        "profile="
        f"{json.dumps(getattr(env, 'dc_arrival_profile', {}), ensure_ascii=False, sort_keys=True)}",
        flush=True,
    )

    bayesian_static_context: BayesianStaticRoutingContext = (
        build_bayesian_static_routing_context(env)
    )

    bayesian_game_definition: BayesianRoutingGameDefinition = (
        build_bayesian_routing_game_definition(static_context=(bayesian_static_context))
    )

    bayesian_game: Optional[BayesianCongestionGame] = None
    if train_config.enable_bayesian_game:
        bayesian_game = build_short_window_game(
            env, train_config, bayesian_game_definition
        )

    short_window_runtime = (
        ShortWindowRuntime(bayesian_game) if bayesian_game is not None else None
    )

    bayesian_game_metadata_json = json.dumps(
        bayesian_game_definition.to_metadata(),
        ensure_ascii=False,
        sort_keys=True,
    )

    print(
        "\n"
        "============================================================\n"
        "🎲 BCGH2-MASCA Bayesian Routing Game Definition\n"
        f"{bayesian_game_metadata_json}\n"
        "============================================================\n",
        flush=True,
    )

    host_observation_builder = HostObservationBuilder(
        env=env,
    )

    host_sac_agents: Dict[str, LocalHostSAC] = {}

    host_replay_buffers: Dict[
        str,
        HostReplayBuffer,
    ] = {}

    for host_dc_index, dc_id in enumerate(env.edge_dc_ids):
        dc_id = str(dc_id)

        if host_sac_config is None:

            host_sac_config = HostSACConfig(
                gamma=(conf.HOST_GAMMA),
                tau=(conf.HOST_TAU),
                actor_lr=(conf.HOST_ACTOR_LR),
                critic_lr=(conf.HOST_CRITIC_LR),
                alpha_lr=(conf.HOST_ALPHA_LR),
                actor_hidden_dim=(conf.HOST_ACTOR_HIDDEN_DIM),
                critic_hidden_dim=(conf.HOST_CRITIC_HIDDEN_DIM),
                initial_alpha=(conf.HOST_INITIAL_ALPHA),
                target_entropy_ratio=(conf.HOST_TARGET_ENTROPY_RATIO),
                max_grad_norm=(conf.HOST_MAX_GRAD_NORM),
                policy_update_interval=(conf.HOST_POLICY_UPDATE_INTERVAL),
                target_update_interval=(conf.HOST_TARGET_UPDATE_INTERVAL),
                device=(conf.DEVICE),
                seed=int(train_config.seed),
            )

        host_obs_dim = int(host_observation_builder.get_obs_dim(dc_id))

        host_action_dim = int(host_observation_builder.get_action_dim(dc_id))

        host_sac_agents[dc_id] = LocalHostSAC(
            obs_dim=(host_obs_dim),
            action_dim=(host_action_dim),
            config=(host_sac_config),
        )

        host_replay_buffers[dc_id] = HostReplayBuffer(
            dc_id=dc_id,
            capacity=int(train_config.host_replay_capacity),
            obs_dim=(host_obs_dim),
            action_dim=(host_action_dim),
            seed=(int(train_config.seed) + 20_000 + host_dc_index),
        )
    host_training_action_steps: Dict[
        str,
        int,
    ] = {str(dc_id): 0 for dc_id in env.edge_dc_ids}

    host_action_rngs: Dict[
        str,
        np.random.Generator,
    ] = {
        str(dc_id): np.random.default_rng(int(train_config.seed) + 30_000 + dc_index)
        for dc_index, dc_id in enumerate(env.edge_dc_ids)
    }

    neighbor_feedback_store = NeighborHistoricalFeedbackStore(
        env=env,
        ewma_alpha=(train_config.neighbor_feedback_ewma_alpha),
        age_scale_samples=(train_config.neighbor_feedback_age_scale_samples),
        confidence_scale_samples=(
            train_config.neighbor_feedback_confidence_scale_samples
        ),
    )

    if bayesian_game is not None:
        neighbor_feedback_store = ShortWindowFeedbackStore(
            env,
            window_s=train_config.short_window_s,
            confidence_scale_samples=train_config.bayesian_confidence_scale,
        )

    routing_observation_builder = RoutingObservationBuilder(
        env=env,
        use_neighbor_historical_feedback=(
            train_config.use_neighbor_historical_feedback
        ),
        neighbor_feedback_provider=(neighbor_feedback_store),
    )
    routing_obs_dim = int(routing_observation_builder.obs_dim)

    routing_state_builder = RoutingCentralizedStateBuilder(
        env=env,
        routing_observation_builder=(routing_observation_builder),
    )

    routing_global_state_dim = int(routing_state_builder.state_dim)

    pending_trace_store = PendingJobTraceStore()
    state_heuristic = RoutingStateHeuristic(
        short_window_runtime,
        neighbor_feedback_store,
        train_config,
        env.routing_action_target_dc_ids,
    )

    training_reward_model = HMasacTrainingRewardModel(
        TrainingRewardConfig(
            task_completion_reward=(conf.TASK_COMPLETION_REWARD),
            completion_time_cost_weight=(conf.COMPLETION_TIME_COST_WEIGHT),
            sla_violation_cost_weight=(conf.SLA_VIOLATION_COST_WEIGHT),
            remote_offload_base_penalty=(conf.REMOTE_OFFLOAD_BASE_PENALTY),
            remote_latency_cost_weight=(conf.REMOTE_LATENCY_COST_WEIGHT),
            sla_risk_cost_weight=(conf.SLA_RISK_COST_WEIGHT),
            timeout_drop_penalty=(conf.TIMEOUT_DROP_PENALTY),
            resource_drop_penalty=(conf.RESOURCE_DROP_PENALTY),
            energy_normalization_j=(conf.ENERGY_NORMALIZATION_J),
            energy_cost_weight=(conf.ENERGY_COST_WEIGHT),
            max_latency_s=float(env.max_latency),
            max_job_duration_s=float(env.max_job_duration),
            sla_deadline_ratio=float(env.sla_deadline_ratio),
            drop_deadline_ratio=float(env.drop_deadline_ratio),
            norm_eps=float(env.norm_eps),
        )
    )

    collector = TransitionCollector(
        env=env,
        short_window_runtime=short_window_runtime,
        state_heuristic=state_heuristic,
        routing_observation_builder=(routing_observation_builder),
        routing_state_builder=(routing_state_builder),
        pending_trace_store=(pending_trace_store),
        training_reward_model=(training_reward_model),
    )

    routing_replay_buffer = RoutingReplayBuffer(
        capacity=int(train_config.routing_replay_capacity),
        local_obs_dim=(routing_obs_dim),
        global_state_dim=(routing_global_state_dim),
        seed=int(train_config.seed),
    )

    routing_masac = RoutingMASAC(
        local_obs_dim=(routing_obs_dim),
        global_state_dim=(routing_global_state_dim),
        action_dim=int(env.action_dim),
        num_agents=int(len(env.possible_agents)),
        config=(routing_masac_config),
    )

    routing_masac.environment_structure = build_checkpoint_structure_metadata(
        env, routing_masac, host_sac_agents
    )
    routing_masac.train_mode()

    if train_config.routing_init_checkpoint:
        routing_masac.initialize_weights(train_config.routing_init_checkpoint)

    action_rng = np.random.default_rng(int(train_config.seed))

    (
        start_episode,
        global_decision_steps,
        routing_normal_action_steps,
        host_training_action_steps,
        best_episode_return,
    ) = load_two_layer_checkpoint_if_needed(
        env=env,
        train_config=(train_config),
        routing_masac=(routing_masac),
        host_sac_agents=(host_sac_agents),
        resume_checkpoint=(train_config.resume_checkpoint),
    )

    checkpoint_dir = Path(train_config.checkpoint_dir)

    (
        episode_log_csv_path,
        dc_log_csv_path,
    ) = build_run_log_paths(
        episode_base_log_path=(train_config.episode_log_csv_path),
        dc_base_log_path=(train_config.dc_log_csv_path),
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    episode_log_csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dc_log_csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    current_log_pointer_path = episode_log_csv_path.parent / "current_train_log.txt"

    atomic_checkpoint_text_write(
        str(episode_log_csv_path.resolve()),
        current_log_pointer_path,
    )

    current_dc_log_pointer_path = dc_log_csv_path.parent / "current_dc_log.txt"

    atomic_checkpoint_text_write(
        str(dc_log_csv_path.resolve()),
        current_dc_log_pointer_path,
    )

    print(
        "\n"
        "============================================================\n"
        "📊 BCGH2-MASCA 双层训练日志\n"
        f"Episode CSV : "
        f"{episode_log_csv_path.resolve()}\n"
        f"DC CSV      : "
        f"{dc_log_csv_path.resolve()}\n"
        f"Episode Ptr : "
        f"{current_log_pointer_path.resolve()}\n"
        f"DC Ptr      : "
        f"{current_dc_log_pointer_path.resolve()}\n"
        "\n"
        "每个 Episode 结束后立即 flush + fsync。\n"
        "============================================================\n",
        flush=True,
    )

    # 暂停评分诊断 JSONL 的收集和写入；恢复时取消以下四处日志调用的注释。
    # RoutingStateHeuristic.log 默认是 None，训练所需的状态评分照常计算。
    # score_log = (
    #     StateScoreLog(episode_log_csv_path.with_suffix(".scores.jsonl"))
    #     if train_config.enable_heuristic_guidance
    #     else None
    # )
    # state_heuristic.log = score_log
    try:
        for episode in range(int(start_episode), int(train_config.num_episodes) + 1):

            if train_config.vary_episode_seed:
                episode_seed = int(train_config.seed + episode - 1)
            else:
                episode_seed = int(train_config.seed)

            pending_trace_store.reset_episode()
            neighbor_feedback_store.reset_episode_counters()
            env.reset(seed=episode_seed)
            if short_window_runtime is not None:
                short_window_runtime.reset()

            collector.reset_episode()
            # if score_log is not None:
            #     score_log.episode = episode
            #     score_log.flush()

            training_stage = resolve_training_stage(
                episode=episode,
                train_config=(train_config),
            )

            if episode == training_stage_start_episode(
                stage=training_stage,
                train_config=train_config,
            ):
                print(
                    "\n"
                    "============================================================\n"
                    f"🚦 Training Stage Start: {training_stage.value}\n"
                    f"Episode Range: "
                    f"{training_stage_start_episode(training_stage, train_config)}"
                    f" -> "
                    f"{training_stage_end_episode(training_stage, train_config)}\n"
                    "============================================================\n",
                    flush=True,
                )

                if training_stage == TrainingStage.JOINT_FINETUNE:
                    save_two_layer_checkpoint(
                        env=env,
                        training_stage=training_stage,
                        train_config=train_config,
                        routing_masac=routing_masac,
                        host_sac_agents=host_sac_agents,
                        model_path=checkpoint_dir / "joint_finetune_start.pt",
                        next_episode=episode,
                        global_decision_steps=global_decision_steps,
                        routing_normal_action_steps=routing_normal_action_steps,
                        host_training_action_steps=host_training_action_steps,
                        best_episode_return=best_episode_return,
                    )

            apply_training_stage_modes(
                stage=(training_stage),
                routing_masac=(routing_masac),
                host_sac_agents=(host_sac_agents),
            )

            per_agent_returns = {}
            for agent_id in env.possible_agents:
                per_agent_returns[str(agent_id)] = 0.0

            stats = EpisodeStatistics(
                episode=int(episode),
                episode_seed=(episode_seed),
                per_agent_returns=(per_agent_returns),
                training_stage=(training_stage.value),
                bayesian_game_enabled=train_config.enable_bayesian_game,
                heuristic_beta=train_config.heuristic_beta,
                routing_gamma=routing_masac_config.gamma,
                reward_shaping_enabled=bool(
                    train_config.enable_heuristic_guidance
                    and train_config.enable_reward_shaping
                ),
                discount_reduction_enabled=bool(
                    train_config.enable_heuristic_guidance
                    and train_config.enable_discount_reduction
                ),
            )

            episode_wall_start = time.perf_counter()
            decision: Optional[DecisionSnapshot] = None

            while env.agents:

                if env.has_pending_host_decision():

                    host_context = env.get_pending_host_decision()

                    host_job_id = str(host_context["job_id"])

                    host_dc_id = str(host_context["dc_id"])

                    host_decision_time = float(env.current_time)

                    host_obs = host_observation_builder.build(
                        dc_id=host_dc_id,
                        job_id=host_job_id,
                    )

                    host_agent = host_sac_agents[host_dc_id]

                    host_replay = host_replay_buffers[host_dc_id]

                    if training_stage == TrainingStage.ROUTING_TRAIN:

                        host_action = host_agent.select_action(
                            host_obs=host_obs,
                            deterministic=True,
                        )

                        host_action_source = "policy"

                    else:

                        host_training_steps = int(
                            host_training_action_steps[host_dc_id]
                        )

                        if host_training_steps < int(
                            train_config.host_random_warmup_steps
                        ):

                            host_action = choose_random_host_action(
                                action_dim=(host_agent.action_dim),
                                rng=(host_action_rngs[host_dc_id]),
                            )

                            host_action_source = "random"

                        else:

                            host_action = host_agent.select_action(
                                host_obs=host_obs,
                                deterministic=False,
                            )

                            host_action_source = "policy"

                        host_training_action_steps[host_dc_id] += 1

                    host_result = env.execute_pending_host_action(
                        host_action=host_action,
                    )

                    result_job_id = str(host_result["job_id"])

                    result_dc_id = str(host_result["dc_id"])

                    result_host_action = int(host_result["host_action"])

                    if result_job_id != host_job_id:
                        raise RuntimeError(
                            "Host execution 返回 Job 不一致："
                            f"expected={host_job_id}, "
                            f"actual={result_job_id}"
                        )

                    if result_dc_id != host_dc_id:
                        raise RuntimeError(
                            "Host execution 返回 DC 不一致："
                            f"job={host_job_id}, "
                            f"expected={host_dc_id}, "
                            f"actual={result_dc_id}"
                        )

                    if result_host_action != int(host_action):
                        raise RuntimeError(
                            "Host execution 返回 action 不一致："
                            f"job={host_job_id}, "
                            f"expected={host_action}, "
                            f"actual={result_host_action}"
                        )

                    actual_host_id = str(host_result["host_id"])

                    pending_trace_store.record_host_step(
                        job_id=host_job_id,
                        dc_id=host_dc_id,
                        env_time=host_decision_time,
                        host_obs=host_obs,
                        action=int(host_action),
                        host_id=actual_host_id,
                        action_source=(host_action_source),
                    )

                    pending_trace_store.record_host_result(
                        job_id=host_job_id,
                        result=str(host_result["execution_result"]),
                        env_time=float(host_result["env_time"]),
                    )
                    if short_window_runtime is not None:
                        short_window_runtime.record_host_result(
                            job_id=host_job_id,
                            dc_id=host_dc_id,
                            execution_result=str(host_result["execution_result"]),
                            now=float(host_result["env_time"]),
                        )
                    stats.record_host_decision(
                        dc_id=(host_dc_id),
                        action_source=(host_action_source),
                    )

                    stats.record_host_result(
                        dc_id=(host_dc_id),
                        execution_result=str(host_result["execution_result"]),
                    )

                    consume_environment_outcome_events(
                        env=env,
                        training_reward_model=(training_reward_model),
                        pending_trace_store=(pending_trace_store),
                        routing_replay_buffer=(routing_replay_buffer),
                        host_replay_buffers=(host_replay_buffers),
                        stats=stats,
                        neighbor_feedback_store=(neighbor_feedback_store),
                        collect_neighbor_historical_feedback=(
                            train_config.collect_neighbor_historical_feedback
                        ),
                        bayesian_game=bayesian_game,
                    )

                    if stage_trains_host(training_stage):

                        host_steps = int(host_training_action_steps[host_dc_id])

                        ready_to_update_host = (
                            host_steps >= int(train_config.host_learning_starts)
                            and host_steps % int(train_config.host_train_every) == 0
                            and host_replay.can_sample(
                                batch_size=int(train_config.host_batch_size)
                            )
                        )

                        if ready_to_update_host:

                            host_update_infos = []

                            for _ in range(int(train_config.host_updates_per_train)):
                                host_update_infos.append(
                                    host_agent.update(
                                        replay_buffer=(host_replay),
                                        batch_size=int(train_config.host_batch_size),
                                    )
                                )

                            record_host_update_block(
                                stats=stats,
                                dc_id=(host_dc_id),
                                update_infos=(host_update_infos),
                            )

                    decision = None

                    continue

                if collector.drain_one_dead_agent():
                    decision = None
                    continue

                if decision is None:
                    decision = collector.capture_decision()

                if decision.forced_action is not None:

                    action = int(decision.forced_action)

                    action_source = "forced"

                elif training_stage == TrainingStage.HOST_PRETRAIN:

                    action = get_self_routing_action(
                        env=env,
                        agent_id=(decision.agent_id),
                    )

                    action_source = "orchestrator"

                elif routing_normal_action_steps < int(
                    train_config.routing_random_warmup_steps
                ):

                    action = choose_random_routing_action(
                        action_dim=(env.action_dim),
                        rng=(action_rng),
                    )

                    action_source = "random"

                else:

                    action = routing_masac.select_action(
                        local_obs=decision.local_obs,
                        agent_index=decision.agent_index,
                        deterministic=False,
                    )

                    action_source = "policy"

                routing_result, next_decision = collector.execute_and_record(
                    decision=decision,
                    action=action,
                    action_source=(action_source),
                )

                decision = next_decision

                stats.record_routing_decision(
                    agent_id=(routing_result.agent_id),
                    reward=(routing_result.immediate_reward),
                    action_type=(routing_result.action_type),
                    action_source=(routing_result.action_source),
                    target_dc_id=(routing_result.target_dc_id),
                )

                if routing_result.job_finalized:
                    finalized_trace = pending_trace_store.get_finalized_trace(
                        routing_result.job_id
                    )

                    flush_finalized_trace_to_replay(
                        finalized_trace=(finalized_trace),
                        routing_replay_buffer=(routing_replay_buffer),
                        host_replay_buffers=(host_replay_buffers),
                        stats=stats,
                        neighbor_feedback_store=(neighbor_feedback_store),
                        collect_neighbor_historical_feedback=(
                            train_config.collect_neighbor_historical_feedback
                        ),
                        bayesian_game=bayesian_game,
                        env=env,
                    )

                    pending_trace_store.pop_finalized_trace(routing_result.job_id)

                consume_environment_outcome_events(
                    env=env,
                    training_reward_model=(training_reward_model),
                    pending_trace_store=(pending_trace_store),
                    routing_replay_buffer=(routing_replay_buffer),
                    host_replay_buffers=(host_replay_buffers),
                    stats=stats,
                    neighbor_feedback_store=(neighbor_feedback_store),
                    collect_neighbor_historical_feedback=(
                        train_config.collect_neighbor_historical_feedback
                    ),
                    bayesian_game=bayesian_game,
                )

                global_decision_steps += 1
                if action_source in {
                    "random",
                    "policy",
                }:

                    routing_normal_action_steps += 1
                    if routing_normal_action_steps == int(
                        train_config.routing_random_warmup_steps
                    ):
                        print(
                            "\n"
                            "============================================================\n"
                            "✅ Routing 随机动作预热结束\n"
                            "============================================================\n"
                        )

                ready_to_update_routing = (
                    stage_trains_routing(training_stage)
                    and action_source
                    in {
                        "random",
                        "policy",
                    }
                    and routing_normal_action_steps
                    >= int(train_config.routing_learning_starts)
                    and routing_normal_action_steps
                    % int(train_config.routing_train_every)
                    == 0
                    and routing_replay_buffer.can_sample(
                        batch_size=int(train_config.routing_batch_size),
                        include_forced_actions=False,
                    )
                )

                if ready_to_update_routing:

                    update_info_block = []

                    for _ in range(int(train_config.routing_updates_per_train)):
                        update_info = routing_masac.update(
                            replay_buffer=(routing_replay_buffer),
                            batch_size=int(train_config.routing_batch_size),
                        )

                        update_info_block.append(update_info)

                    record_routing_update_block(
                        stats=stats,
                        update_infos=(update_info_block),
                    )

            consume_environment_outcome_events(
                env=env,
                training_reward_model=(training_reward_model),
                pending_trace_store=(pending_trace_store),
                routing_replay_buffer=(routing_replay_buffer),
                host_replay_buffers=(host_replay_buffers),
                stats=stats,
                neighbor_feedback_store=(neighbor_feedback_store),
                collect_neighbor_historical_feedback=(
                    train_config.collect_neighbor_historical_feedback
                ),
                bayesian_game=bayesian_game,
            )
            pending_trace_store.assert_no_open_trace()
            pending_trace_store.assert_no_unflushed_finalized_trace()

            flush_pending_update_metrics(stats)

            wall_time_seconds = time.perf_counter() - episode_wall_start

            service_metrics = calculate_service_metrics(env)

            energy_metrics = calculate_episode_energy_metrics(env)

            load_metrics = calculate_episode_load_metrics(env)

            episode_log_row = build_episode_log_row(
                stats=stats,
                env=env,
                routing_replay_buffer=(routing_replay_buffer),
                host_replay_buffers=(host_replay_buffers),
                routing_masac=(routing_masac),
                host_sac_agents=(host_sac_agents),
                host_training_action_steps=(host_training_action_steps),
                pending_trace_store=(pending_trace_store),
                neighbor_feedback_store=(neighbor_feedback_store),
                global_decision_steps=(global_decision_steps),
                routing_normal_action_steps=(routing_normal_action_steps),
                wall_time_seconds=(wall_time_seconds),
                service_metrics=(service_metrics),
                energy_metrics=(energy_metrics),
                load_metrics=(load_metrics),
            )

            dc_log_rows = build_dc_log_rows(
                stats=stats,
                env=env,
                host_sac_agents=(host_sac_agents),
                host_replay_buffers=(host_replay_buffers),
                host_training_action_steps=(host_training_action_steps),
                load_metrics=(load_metrics),
                neighbor_feedback_store=(neighbor_feedback_store),
            )

            append_csv_log(
                csv_path=(episode_log_csv_path),
                row=(episode_log_row),
            )
            # if score_log is not None:
            #     score_log.flush()

            for dc_log_row in dc_log_rows:
                append_csv_log(
                    csv_path=(dc_log_csv_path),
                    row=(dc_log_row),
                )

            if episode % int(train_config.log_interval) == 0:
                print_episode_summary(episode_log_row)

            if stats.episode_return > best_episode_return:
                best_episode_return = float(stats.episode_return)

    finally:
        # if score_log is not None:
        #     score_log.close()
        close_method = getattr(env, "close", None)
        if callable(close_method):
            close_method()

    save_two_layer_checkpoint(
        env=env,
        training_stage=(
            resolve_training_stage(
                episode=int(train_config.num_episodes),
                train_config=(train_config),
            )
        ),
        train_config=(train_config),
        routing_masac=(routing_masac),
        host_sac_agents=(host_sac_agents),
        model_path=(checkpoint_dir / "final.pt"),
        next_episode=(int(train_config.num_episodes) + 1),
        global_decision_steps=(global_decision_steps),
        routing_normal_action_steps=(routing_normal_action_steps),
        host_training_action_steps=(host_training_action_steps),
        best_episode_return=(best_episode_return),
    )

    return (
        routing_masac,
        host_sac_agents,
    )


def main() -> None:
    train_config = TrainConfig(
        num_episodes=conf.Episodes,
        routing_replay_capacity=(conf.ROUTING_REPLAY_CAPACITY),
        host_replay_capacity=(conf.HOST_REPLAY_CAPACITY),
        routing_batch_size=(conf.ROUTING_BATCH_SIZE),
        routing_random_warmup_steps=(conf.ROUTING_RANDOM_WARMUP_STEPS),
        routing_learning_starts=(conf.ROUTING_LEARNING_STARTS),
        routing_train_every=(conf.ROUTING_TRAIN_EVERY),
        routing_updates_per_train=(conf.ROUTING_UPDATES_PER_TRAIN),
        host_batch_size=(conf.HOST_BATCH_SIZE),
        host_random_warmup_steps=(conf.HOST_RANDOM_WARMUP_STEPS),
        host_learning_starts=(conf.HOST_LEARNING_STARTS),
        host_train_every=(conf.HOST_TRAIN_EVERY),
        host_updates_per_train=(conf.HOST_UPDATES_PER_TRAIN),
        host_pretrain_episodes=(conf.HOST_PRETRAIN_EPISODES),
        routing_train_episodes=(conf.ROUTING_TRAIN_EPISODES),
        joint_finetune_episodes=(conf.JOINT_FINETUNE_EPISODES),
        log_interval=conf.Log_interval,
        checkpoint_interval=conf.Checkpoint_Interval,
        seed=conf.Seed,
        checkpoint_dir="model/BCGH2-MASCA/checkpoints",
        episode_log_csv_path="result/BCGH2-MASCA/episode_log.csv",
        dc_log_csv_path="result/BCGH2-MASCA/dc_log.csv",
        old_env_path=conf.Old_Env_Path,
        resume_checkpoint=None,
        vary_episode_seed=conf.Vary_Episode_Seed,
        collect_neighbor_historical_feedback=(
            conf.COLLECT_NEIGHBOR_HISTORICAL_FEEDBACK
        ),
        use_neighbor_historical_feedback=(conf.USE_NEIGHBOR_HISTORICAL_FEEDBACK),
        neighbor_feedback_ewma_alpha=(conf.NEIGHBOR_FEEDBACK_EWMA_ALPHA),
        neighbor_feedback_age_scale_samples=(conf.NEIGHBOR_FEEDBACK_AGE_SCALE_SAMPLES),
        neighbor_feedback_confidence_scale_samples=(
            conf.NEIGHBOR_FEEDBACK_CONFIDENCE_SCALE_SAMPLES
        ),
        enable_bayesian_game=True,
        enable_heuristic_guidance=True,
    )

    host_sac_config = HostSACConfig(
        gamma=(conf.HOST_GAMMA),
        tau=(conf.HOST_TAU),
        actor_lr=(conf.HOST_ACTOR_LR),
        critic_lr=(conf.HOST_CRITIC_LR),
        alpha_lr=(conf.HOST_ALPHA_LR),
        actor_hidden_dim=(conf.HOST_ACTOR_HIDDEN_DIM),
        critic_hidden_dim=(conf.HOST_CRITIC_HIDDEN_DIM),
        initial_alpha=(conf.HOST_INITIAL_ALPHA),
        target_entropy_ratio=(conf.HOST_TARGET_ENTROPY_RATIO),
        max_grad_norm=(conf.HOST_MAX_GRAD_NORM),
        policy_update_interval=(conf.HOST_POLICY_UPDATE_INTERVAL),
        target_update_interval=(conf.HOST_TARGET_UPDATE_INTERVAL),
        device=(conf.DEVICE),
        seed=(conf.Seed),
    )

    routing_masac_config = RoutingMASACConfig(
        gamma=(conf.ROUTING_GAMMA),
        tau=(conf.ROUTING_TAU),
        actor_lr=(conf.ROUTING_ACTOR_LR),
        critic_lr=(conf.ROUTING_CRITIC_LR),
        alpha_lr=(conf.ROUTING_ALPHA_LR),
        actor_hidden_dim=(conf.ROUTING_ACTOR_HIDDEN_DIM),
        critic_hidden_dim=(conf.ROUTING_CRITIC_HIDDEN_DIM),
        initial_alpha=(conf.ROUTING_INITIAL_ALPHA),
        target_entropy_ratio=(conf.ROUTING_TARGET_ENTROPY_RATIO),
        max_grad_norm=(conf.ROUTING_MAX_GRAD_NORM),
        policy_update_interval=(conf.ROUTING_POLICY_UPDATE_INTERVAL),
        target_update_interval=(conf.ROUTING_TARGET_UPDATE_INTERVAL),
        device=(conf.DEVICE),
        seed=(conf.Seed),
    )

    train(
        train_config=(train_config),
        routing_masac_config=(routing_masac_config),
        host_sac_config=(host_sac_config),
    )


if __name__ == "__main__":
    main()
