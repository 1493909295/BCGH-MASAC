"""H-MASAC training entry point and three-stage orchestration."""

from __future__ import annotations

import sys
from pathlib import Path

if not __package__:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    __package__ = "schedulers.H-MASAC"

from .training_support import *


def train(
    train_config: TrainConfig,
    routing_masac_config: Optional[RoutingMASACConfig] = None,
    host_sac_config: Optional[HostSACConfig] = None,
    *,
    routing_observation_builder_type=None,
    routing_agent_type=None,
    host_initializer=None,
) -> Tuple[
    RoutingMASAC,
    Dict[str, LocalHostSAC],
]:
    validate_training_stage_config(train_config)
    routing_observation_builder_type = routing_observation_builder_type or RoutingObservationBuilder
    routing_agent_type = routing_agent_type or RoutingMASAC

    set_global_random_seeds(
        train_config.seed,
        device=getattr(routing_masac_config, "device", None) or conf.DEVICE,
    )

    # 创建初始环境
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
    host_observation_builder = HostObservationBuilder(
        env=env,
    )

    # ==========================================================
    # 每个 Edge DC 建立独立 Local Host SAC。
    #
    # Host 层：
    #   - 不使用 PettingZoo
    #   - 不共享 Actor
    #   - 不共享 Critic
    #   - 不使用 Mask
    # ==========================================================

    host_sac_agents: Dict[str, LocalHostSAC] = {}

    host_replay_buffers: Dict[
        str,
        HostReplayBuffer,
    ] = {}

    for host_dc_index, dc_id in enumerate(env.edge_dc_ids):
        dc_id = str(dc_id)

        if host_sac_config is None:
            # ==========================================================
            # Local Host SAC fallback configuration
            #
            # 即使 train() 被其他入口直接调用，
            # 没有显式传入 HostSACConfig，
            # 也必须使用 HOST_* 专属参数。
            #
            # 禁止退回 Flat-MASAC / Routing 公共超参数。
            # ==========================================================

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

        # ==========================================================
        # 每个 DC 独立 Local Host SAC。
        # ==========================================================

        host_sac_agents[dc_id] = LocalHostSAC(
            obs_dim=(host_obs_dim),
            action_dim=(host_action_dim),
            config=(host_sac_config),
        )

        # ==========================================================
        # 第十九步：
        # 每个 Local Host SAC 对应自己的 Host ReplayBuffer。
        #
        # 不共享：
        #   Actor
        #   Critic
        #   ReplayBuffer
        # ==========================================================

        host_replay_buffers[dc_id] = HostReplayBuffer(
            dc_id=dc_id,
            capacity=int(train_config.host_replay_capacity),
            obs_dim=(host_obs_dim),
            action_dim=(host_action_dim),
            seed=(int(train_config.seed) + 20_000 + host_dc_index),
        )
    if host_initializer is not None:
        host_initializer(env, host_sac_agents)

    host_training_action_steps: Dict[
        str,
        int,
    ] = {str(dc_id): 0 for dc_id in env.edge_dc_ids}

    # ==============================================================
    # Local Host SAC Independent Action RNG
    #
    # 每个 DC 使用自己独立的 NumPy RNG，
    # 用于 Host random warmup 阶段随机选择 Host。
    #
    # 这样不同 DC 的随机 Host action stream 不会共用
    # 同一个随机生成器。
    # ==============================================================

    host_action_rngs: Dict[
        str,
        np.random.Generator,
    ] = {
        str(dc_id): np.random.default_rng(int(train_config.seed) + 30_000 + dc_index)
        for dc_index, dc_id in enumerate(env.edge_dc_ids)
    }

    # ==============================================================
    # Neighbor Historical Feedback Store
    #
    # 支持两种运行方式：
    #
    #   1. Collect-Only：收集历史结果，但 Actor 看到固定零特征；
    #   2. Collect-and-Use：收集历史结果，并拼接进 Actor observation。
    #
    # 如果启用 USE 却关闭 COLLECT，Store 将始终没有新样本，
    # Actor 会长期收到零反馈，因此将其视为无效配置。
    # ==============================================================

    if (
        train_config.use_neighbor_historical_feedback
        and not train_config.collect_neighbor_historical_feedback
    ):
        raise RuntimeError(
            "USE_NEIGHBOR_HISTORICAL_FEEDBACK=True 时，"
            "COLLECT_NEIGHBOR_HISTORICAL_FEEDBACK 必须同时为 True。"
        )

    neighbor_feedback_store = NeighborHistoricalFeedbackStore(
        env=env,
        ewma_alpha=(train_config.neighbor_feedback_ewma_alpha),
        age_scale_samples=(train_config.neighbor_feedback_age_scale_samples),
        confidence_scale_samples=(train_config.neighbor_feedback_confidence_scale_samples),
    )

    routing_observation_builder = routing_observation_builder_type(
        env=env,
        # False 时返回固定全 0 Feedback block；True 时从
        # Store 读取历史反馈并拼接进 Routing Actor observation。
        use_neighbor_historical_feedback=(train_config.use_neighbor_historical_feedback),
        # Store 同时作为历史反馈 Provider 和日志数据源。
        neighbor_feedback_provider=(neighbor_feedback_store),
    )
    routing_obs_dim = int(routing_observation_builder.obs_dim)

    routing_state_builder = RoutingCentralizedStateBuilder(
        env=env,
        routing_observation_builder=(routing_observation_builder),
    )

    routing_global_state_dim = int(routing_state_builder.state_dim)

    pending_trace_store = PendingJobTraceStore()

    # ==============================================================
    # H-MASAC Training Reward Model
    #
    # Reward Model 属于 Trainer，
    # Environment 只提供物理事实。
    #
    # 当前参数数值与第三十一步之前完全一致。
    # ==============================================================

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

    # 创建 Transition 采集器
    collector = TransitionCollector(
        env=env,
        routing_observation_builder=(routing_observation_builder),
        routing_state_builder=(routing_state_builder),
        pending_trace_store=(pending_trace_store),
        # Training Reward 由 Trainer-side Reward Model 负责。
        training_reward_model=(training_reward_model),
    )

    # ==============================================================
    # Routing MASAC 专用 ReplayBuffer
    #
    # 与 Host Replay 完全独立。
    #
    # 这里只接收 Job terminal 后生成的
    # Finalized RoutingTransition。
    # ==============================================================

    routing_replay_buffer = RoutingReplayBuffer(
        capacity=int(train_config.routing_replay_capacity),
        local_obs_dim=(routing_obs_dim),
        global_state_dim=(routing_global_state_dim),
        seed=int(train_config.seed),
    )

    # 没有传入算法配置时，使用 MASACConfig 默认值，但让算法随机种子与训练配置保持一致
    # ==============================================================
    # Routing MASAC + CTDE
    #
    # 这里创建的是整个系统唯一的一套 Routing MASAC。
    #
    # Host SAC 不使用本对象。
    # 每个 Edge DC 的 LocalHostSAC 已在前面独立创建。
    # ==============================================================

    if routing_masac_config is None:
        # ==========================================================
        # Routing MASAC fallback configuration
        #
        # config.py 是 H-MASAC 实验的统一参数入口。
        # 即使 train() 被直接调用，
        # Routing 也必须继续使用 ROUTING_* 配置。
        # ==========================================================

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
            seed=int(train_config.seed),
        )

    routing_masac = routing_agent_type(
        # Routing Actor local input
        local_obs_dim=(routing_obs_dim),
        # Routing CTDE centralized Critic input
        global_state_dim=(routing_global_state_dim),
        action_dim=int(env.action_dim),
        num_agents=int(len(env.possible_agents)),
        config=(routing_masac_config),
    )

    metadata_builder = getattr(routing_observation_builder, "checkpoint_metadata", None)
    if callable(metadata_builder):
        routing_masac.observation_metadata = metadata_builder()

    routing_masac.train_mode()

    action_rng = np.random.default_rng(int(train_config.seed))

    (
        start_episode,
        global_decision_steps,
        routing_normal_action_steps,
        host_training_action_steps,
        best_episode_return,
    ) = load_two_layer_checkpoint_if_needed(
        # ==========================================================
        # Resume 前需要当前 Environment / TrainConfig，
        # 用于验证 Cloud、DC、Host、Observation、
        # Action 以及三阶段训练 metadata。
        # ==========================================================
        env=env,
        train_config=(train_config),
        routing_masac=(routing_masac),
        host_sac_agents=(host_sac_agents),
        resume_checkpoint=(train_config.resume_checkpoint),
    )

    checkpoint_dir = Path(train_config.checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = PROJECT_ROOT / checkpoint_dir
    checkpoint_dir = checkpoint_dir.resolve()

    # ==============================================================
    # Two-Level Training Logs
    #
    # 同一次训练共用相同 timestamp：
    #
    #   episode_log_YYYYMMDD_HHMMSS.csv
    #   dc_log_YYYYMMDD_HHMMSS.csv
    # ==============================================================

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

    # ==============================================================
    # Compatibility Pointer
    #
    # current_train_log.txt 继续指向主 Episode Log，
    # 防止已有服务器查看脚本失效。
    # ==============================================================

    current_log_pointer_path = episode_log_csv_path.parent / "current_train_log.txt"

    atomic_checkpoint_text_write(
        str(episode_log_csv_path.resolve()),
        current_log_pointer_path,
    )

    # 新增 DC 日志 pointer。
    current_dc_log_pointer_path = dc_log_csv_path.parent / "current_dc_log.txt"

    atomic_checkpoint_text_write(
        str(dc_log_csv_path.resolve()),
        current_dc_log_pointer_path,
    )

    print(
        "\n"
        "============================================================\n"
        f"📊 {getattr(routing_masac, 'scheduler_name', 'H-MASAC')} 双层训练日志\n"
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

    try:
        # 从 start_episode 训练到 num_episodes，包含最后一个 episode。
        for episode in range(int(start_episode), int(train_config.num_episodes) + 1):

            ##################### 一轮 episode 开始前的准备 ####################
            # 根据配置决定当前 episode 使用哪个环境 seed
            if train_config.vary_episode_seed:
                episode_seed = int(train_config.seed + episode - 1)
            else:
                episode_seed = int(train_config.seed)

            # 重启环境
            pending_trace_store.reset_episode()
            neighbor_feedback_store.reset_episode_counters()
            env.reset(seed=episode_seed)

            collector.reset_episode()

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
                    # Save the state before the first joint optimization episode.
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

            # 为每个智能体创建奖励累计字典
            per_agent_returns = {}
            for agent_id in env.possible_agents:
                per_agent_returns[str(agent_id)] = 0.0

            # 创建当前 episode 的统计对象
            stats = EpisodeStatistics(
                episode=int(episode),
                episode_seed=(episode_seed),
                per_agent_returns=(per_agent_returns),
                training_stage=(training_stage.value),
            )

            # 记录episode开始的时间
            episode_wall_start = time.perf_counter()
            ###################################################################
            decision: Optional[DecisionSnapshot] = None

            # 只要 PettingZoo 的活跃智能体列表不为空，就继续循环
            while env.agents:

                # ==========================================================
                # Phase 1：Host decision
                #
                # Host 必须优先于任何 PettingZoo 操作处理。
                # Self Routing 后：
                #
                #   agent_selection == None
                #   pending_host_job_id != None
                #
                # 因此此处绝对不能调用：
                #   collector.capture_decision()
                #   env.step()
                # ==========================================================
                if env.has_pending_host_decision():
                    # ==========================================================
                    # Host Phase
                    #
                    # Host 层完全脱离 PettingZoo。
                    # 这里使用与 Routing Collector 相同的
                    # pending_trace_store。
                    # ==========================================================

                    host_context = env.get_pending_host_decision()

                    host_job_id = str(host_context["job_id"])

                    host_dc_id = str(host_context["dc_id"])

                    # ==========================================================
                    # 保存 Host SAC 真正做决策的时间。
                    #
                    # execute_pending_host_action() 后环境可能已经向前推进，
                    # 因此不能在执行之后再读取 decision time。
                    # ==========================================================

                    host_decision_time = float(env.current_time)

                    # Host Observation 完全独立于 PettingZoo。
                    host_obs = host_observation_builder.build(
                        dc_id=host_dc_id,
                        job_id=host_job_id,
                    )

                    # 当前 DC 自己的 Local Host SAC 决策。
                    host_agent = host_sac_agents[host_dc_id]

                    host_replay = host_replay_buffers[host_dc_id]

                    # ==========================================================
                    # Training Stage 决定 Host action 行为。
                    # ==========================================================

                    if training_stage == TrainingStage.ROUTING_TRAIN:
                        # ======================================================
                        # Stage 2：
                        #
                        # Host 网络完全冻结。
                        #
                        # 使用 deterministic policy，
                        # 降低 Routing MASAC 所面对环境的非平稳性。
                        # ======================================================

                        host_action = host_agent.select_action(
                            host_obs=host_obs,
                            deterministic=True,
                        )

                        host_action_source = "policy"

                    else:

                        # ======================================================
                        # Stage 1 / Stage 3：
                        #
                        # Host SAC 参与训练。
                        # 每个 DC 独立进行 random warmup。
                        # ======================================================

                        host_training_steps = int(host_training_action_steps[host_dc_id])

                        if host_training_steps < int(train_config.host_random_warmup_steps):

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

                    # ==========================================================
                    # 非 PettingZoo Host execution。
                    #
                    # Environment 会返回：
                    #
                    #   job_id
                    #   dc_id
                    #   host_action
                    #   host_id
                    #   execution_result
                    #   env_time
                    # ==========================================================

                    host_result = env.execute_pending_host_action(
                        host_action=host_action,
                    )

                    # ==========================================================
                    # 防御性检查：
                    # Trace 中记录的 Job/DC/Action 必须与 Environment
                    # 真正执行的对象完全一致。
                    # ==========================================================

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

                    # ==========================================================
                    # 将 Host Decision 正式加入 Job Causal Trace。
                    #
                    # 此时仅记录事实。
                    # 仍然不写 Host ReplayBuffer。
                    # ==========================================================

                    pending_trace_store.record_host_step(
                        job_id=host_job_id,
                        dc_id=host_dc_id,
                        env_time=host_decision_time,
                        host_obs=host_obs,
                        action=int(host_action),
                        host_id=actual_host_id,
                        # Stage 1 warmup 可以是 random；
                        # Stage 1/3 后期以及 Stage 2 为 policy。
                        action_source=(host_action_source),
                    )

                    # ==========================================================
                    # 回填 Host placement 的即时执行结果：
                    #
                    #   started
                    #   queued
                    #   dropped
                    #
                    # 它不是最终 SLA/completion reward。
                    # ==========================================================

                    pending_trace_store.record_host_result(
                        job_id=host_job_id,
                        result=str(host_result["execution_result"]),
                        env_time=float(host_result["env_time"]),
                    )
                    stats.record_host_decision(
                        dc_id=(host_dc_id),
                        action_source=(host_action_source),
                    )

                    stats.record_host_result(
                        dc_id=(host_dc_id),
                        execution_result=str(host_result["execution_result"]),
                    )

                    # ==========================================================
                    # execute_pending_host_action() 内部会继续推进事件，
                    # 因而期间可能已经发生：
                    #
                    #   completed
                    #   waiting_timeout
                    #   resource failure
                    #
                    # 必须在 Host Step 已经写入以后，
                    # 再消费这些 delayed outcome。
                    # ==========================================================

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
                    )
                    # ==========================================================
                    # Local Host SAC Update
                    #
                    # 只在：
                    #
                    #   Stage 1 Host Pretrain
                    #   Stage 3 Joint Fine-tune
                    #
                    # 更新。
                    #
                    # Stage 2 Host 完全冻结。
                    # ==========================================================

                    if stage_trains_host(training_stage):

                        host_steps = int(host_training_action_steps[host_dc_id])

                        ready_to_update_host = (
                            host_steps >= int(train_config.host_learning_starts)
                            and host_steps % int(train_config.host_train_every) == 0
                            and host_replay.can_sample(batch_size=int(train_config.host_batch_size))
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

                # ==========================================================
                # Phase 2：PettingZoo Routing decision
                # ==========================================================

                if collector.drain_one_dead_agent():
                    decision = None
                    continue

                if decision is None:
                    decision = collector.capture_decision()

                # ----------------------------------------------------------
                # 以下继续保留现有 Routing：
                #
                # forced
                # random warmup
                # Routing MASAC policy
                # ----------------------------------------------------------

                if decision.forced_action is not None:

                    # Environment lifecycle forced drop
                    action = int(decision.forced_action)

                    action_source = "forced"

                elif training_stage == TrainingStage.HOST_PRETRAIN:

                    # ======================================================
                    # Stage 1:
                    #
                    # Routing 不参与学习。
                    # 所有正常 Job 都进入当前 DC 的 Host 层。
                    #
                    # 该 Self action：
                    #   - 写入 Causal Trace；
                    #   - 不进入 Routing Replay；
                    #   - 不增加 Routing warmup step。
                    # ======================================================

                    action = get_self_routing_action(
                        env=env,
                        agent_id=(decision.agent_id),
                    )

                    action_source = "orchestrator"

                elif routing_normal_action_steps < int(train_config.routing_random_warmup_steps):

                    action = choose_random_routing_action(
                        action_dim=(env.action_dim),
                        rng=(action_rng),
                    )

                    action_source = "random"

                else:

                    action = routing_masac.select_action(
                        local_obs=(decision.local_obs),
                        agent_index=(decision.agent_index),
                        deterministic=False,
                    )

                    action_source = "policy"

                # ==========================================================
                # Routing action 的真实语义统一由 Collector
                # 调用 Environment._decode_action() 决定。
                #
                # Trainer 不再复制 action_type 推断逻辑。
                # ==========================================================

                routing_result, next_decision = collector.execute_and_record(
                    decision=decision,
                    action=action,
                    action_source=(action_source),
                )

                decision = next_decision

                # 记录这条经验的奖励和动作类型
                # ==============================================================
                # Routing Layer Episode Statistics
                #
                # RoutingActionResult 已经由 Collector 根据 Environment
                # 的真实动作语义生成。
                #
                # 这里同时记录：
                #   - source DC
                #   - target DC
                #   - action type
                #   - action source
                #   - immediate reward
                #
                # 从而可以生成完整 source -> target Routing Matrix。
                # ==============================================================

                stats.record_routing_decision(
                    agent_id=(routing_result.agent_id),
                    reward=(routing_result.immediate_reward),
                    action_type=(routing_result.action_type),
                    action_source=(routing_result.action_source),
                    target_dc_id=(routing_result.target_dc_id),
                )

                # ==========================================================
                # Forced Drop / Actor Drop 会在 execute_and_collect()
                # 内部直接完成 Job Finalize。
                #
                # 它不会产生后续 Environment reward correction，
                # 所以这里必须检查并立即 Flush。
                # ==========================================================

                # ==========================================================
                # 当前 action 如果让 Job 在 Collector 中直接 Finalize，
                # 典型情况就是 forced drop。
                #
                # 此类 Job 不会再产生 Environment terminal correction，
                # 因此这里立即 Flush。
                # ==========================================================

                if routing_result.job_finalized:
                    finalized_trace = pending_trace_store.get_finalized_trace(routing_result.job_id)

                    flush_finalized_trace_to_replay(
                        finalized_trace=(finalized_trace),
                        routing_replay_buffer=(routing_replay_buffer),
                        host_replay_buffers=(host_replay_buffers),
                        stats=stats,
                        neighbor_feedback_store=(neighbor_feedback_store),
                        collect_neighbor_historical_feedback=(
                            train_config.collect_neighbor_historical_feedback
                        ),
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
                )

                # 增加计数
                global_decision_steps += 1
                if action_source in {
                    "random",
                    "policy",
                }:

                    routing_normal_action_steps += 1
                    if routing_normal_action_steps == int(train_config.routing_random_warmup_steps):
                        print(
                            "\n"
                            "============================================================\n"
                            "✅ Routing 随机动作预热结束\n"
                            "============================================================\n"
                        )

                # 同时满足以下条件才允许更新网络：
                # 1. 普通动作总数已经达到 learning_starts；
                # 2. ReplayBuffer 中普通经验足够采样一个 batch。
                ready_to_update_routing = (
                    stage_trains_routing(training_stage)
                    and action_source
                    in {
                        "random",
                        "policy",
                    }
                    and routing_normal_action_steps >= int(train_config.routing_learning_starts)
                    and routing_normal_action_steps % int(train_config.routing_train_every) == 0
                    and routing_replay_buffer.can_sample(
                        batch_size=int(train_config.routing_batch_size),
                        include_forced_actions=False,
                    )
                )

                # 网络更新
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

            # ==============================================================
            # Episode 结束后的最后一次 delayed outcome flush。
            #
            # 正常情况下 Routing/Host Branch 已经实时消费；
            # 这里作为 Episode tail 的防御性收尾，
            # 防止最后一批 terminal correction 留在 Environment 中。
            # ==============================================================

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
            )
            pending_trace_store.assert_no_open_trace()
            pending_trace_store.assert_no_unflushed_finalized_trace()

            flush_pending_update_metrics(stats)

            # 计算当前 episode 的真实运行秒数。
            wall_time_seconds = time.perf_counter() - episode_wall_start

            # ==============================================================
            # Episode-level Physical Metrics
            #
            # 三套真实系统指标只计算一次，
            # Episode Log / DC Log 共用同一份 snapshot。
            # ==============================================================

            service_metrics = calculate_service_metrics(env)

            energy_metrics = calculate_episode_energy_metrics(env)

            load_metrics = calculate_episode_load_metrics(env)

            # ==============================================================
            # Episode Log
            # ==============================================================

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

            # ==============================================================
            # Per-DC Log
            # ==============================================================

            dc_log_rows = build_dc_log_rows(
                stats=stats,
                env=env,
                host_sac_agents=(host_sac_agents),
                host_replay_buffers=(host_replay_buffers),
                host_training_action_steps=(host_training_action_steps),
                load_metrics=(load_metrics),
                neighbor_feedback_store=(neighbor_feedback_store),
            )

            # ==============================================================
            # Immediate Disk Persistence
            # ==============================================================

            append_csv_log(
                csv_path=(episode_log_csv_path),
                row=(episode_log_row),
            )

            for dc_log_row in dc_log_rows:
                append_csv_log(
                    csv_path=(dc_log_csv_path),
                    row=(dc_log_row),
                )

            # ==============================================================
            # Console Summary
            # ==============================================================

            if episode % int(train_config.log_interval) == 0:
                print_episode_summary(episode_log_row)

            # Keep the best return in the resumable trainer state.
            if stats.episode_return > best_episode_return:
                best_episode_return = float(stats.episode_return)

    finally:
        close_method = getattr(env, "close", None)
        if callable(close_method):
            close_method()

    # 全部 episode 完成后保存 final checkpoint
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
        # ==========================================================
        # Routing MASAC Training Schedule
        # ==========================================================
        routing_batch_size=(conf.ROUTING_BATCH_SIZE),
        routing_random_warmup_steps=(conf.ROUTING_RANDOM_WARMUP_STEPS),
        routing_learning_starts=(conf.ROUTING_LEARNING_STARTS),
        routing_train_every=(conf.ROUTING_TRAIN_EVERY),
        routing_updates_per_train=(conf.ROUTING_UPDATES_PER_TRAIN),
        # ==========================================================
        # Local Host SAC Training Schedule
        # ==========================================================
        host_batch_size=(conf.HOST_BATCH_SIZE),
        host_random_warmup_steps=(conf.HOST_RANDOM_WARMUP_STEPS),
        host_learning_starts=(conf.HOST_LEARNING_STARTS),
        host_train_every=(conf.HOST_TRAIN_EVERY),
        host_updates_per_train=(conf.HOST_UPDATES_PER_TRAIN),
        # ==========================================================
        # Three-Stage Training
        # ==========================================================
        host_pretrain_episodes=(conf.HOST_PRETRAIN_EPISODES),
        routing_train_episodes=(conf.ROUTING_TRAIN_EPISODES),
        joint_finetune_episodes=(conf.JOINT_FINETUNE_EPISODES),
        log_interval=conf.Log_interval,
        checkpoint_interval=conf.Checkpoint_Interval,
        seed=conf.Seed,
        checkpoint_dir=conf.Checkpoint_Dir,
        episode_log_csv_path=(conf.H_MASAC_EPISODE_LOG_CSV_PATH),
        dc_log_csv_path=(conf.H_MASAC_DC_LOG_CSV_PATH),
        old_env_path=conf.Old_Env_Path,
        resume_checkpoint=conf.Resume_Checkpoint,
        vary_episode_seed=conf.Vary_Episode_Seed,
        # ==========================================================
        # Neighbor Historical Feedback
        # ==========================================================
        collect_neighbor_historical_feedback=(conf.COLLECT_NEIGHBOR_HISTORICAL_FEEDBACK),
        use_neighbor_historical_feedback=(conf.USE_NEIGHBOR_HISTORICAL_FEEDBACK),
        neighbor_feedback_ewma_alpha=(conf.NEIGHBOR_FEEDBACK_EWMA_ALPHA),
        neighbor_feedback_age_scale_samples=(conf.NEIGHBOR_FEEDBACK_AGE_SCALE_SAMPLES),
        neighbor_feedback_confidence_scale_samples=(
            conf.NEIGHBOR_FEEDBACK_CONFIDENCE_SCALE_SAMPLES
        ),
    )

    host_sac_config = HostSACConfig(
        # ======================================================
        # Host SAC 使用完全独立的算法超参数。
        #
        # 正常 main() 入口与 train() fallback 必须保持一致，
        # 防止 Host 又退回旧 Flat-MASAC 公共参数。
        # ======================================================
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
