from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
if __package__:
    from .routing_observation import RoutingObservationBuilder
    from .routing_centralized_state import RoutingCentralizedStateBuilder
    from .pending_job_trace import PendingJobTraceStore
    from .training_reward import HMasacTrainingRewardModel
else:
    from routing_observation import RoutingObservationBuilder
    from routing_centralized_state import RoutingCentralizedStateBuilder
    from pending_job_trace import PendingJobTraceStore
    from training_reward import HMasacTrainingRewardModel
FloatArray = NDArray[np.float32]


# 接口清单类 TransitionCollector 的对象，都应该至少具备它描述的这些接口
# CloudEdgeEnvLike 类不负责实现逻辑
class CloudEdgeEnvLike(Protocol):

    # pettingzoo 需要区分全部可能智能体与现在活着的智能体，在我的环境中这俩个没区别
    possible_agents: Sequence[str]
    agents: Sequence[str]

    # 当前执行step的智能体与当前做决策的智能体，在我环境中这两个也相同
    agent_selection: Optional[str]
    current_agent_id: Optional[str]

    # current_job_id: Optional[str]
    # pending_host_job_id: Optional[str]
    # pending_host_dc_id: Optional[str]
    current_job_id: Optional[str]
    current_time: float
    # local_obs_dim: int
    # global_state_dim: int
    action_dim: int
    drop_action: int
    has_reset: bool

    # 把str类型的agent id映射成数字
    agent_name_mapping: Mapping[str, int]


    # 正常停止与异常停止标记，但我其实没实现异常停止 ·_·
    terminations: Mapping[str, bool]
    truncations: Mapping[str, bool]

    # def observe(self, agent: str) -> Dict[str, np.ndarray]:
    #     ...
    # def state(self) -> np.ndarray:
    #     ...
    def step(self, action: Optional[int]) -> None:
        ...

    def pop_last_routing_action_facts(
            self,
    ) -> Dict[str, Any]:
        ...

# 动作执行前快照捕捉，True代表不可重新赋值
@dataclass(frozen=True)
class DecisionSnapshot:
    agent_id: str
    agent_index: int
    job_id: str
    env_time: float
    local_obs: FloatArray

    global_state: FloatArray

    # 等于None表示由actor给工作，等于-1表示动作是丢弃
    forced_action: Optional[int] = None

    # 转化成字典方便打印和传递
    def as_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_index": self.agent_index,
            "job_id": self.job_id,
            "env_time": self.env_time,
            "local_obs": self.local_obs.copy(),

            "global_state": self.global_state.copy(),
            "forced_action": self.forced_action,
        }

@dataclass(frozen=True)
class RoutingActionResult:
    """
    一次 Routing action 执行完成后的即时结果。

    注意：
        RoutingActionResult 不是 Replay Transition。

    它只服务于 Trainer：
        - Episode statistics；
        - 判断当前 Job 是否已经 Finalize；
        - 识别本次动作类型；
        - 继续环境控制流。

    正式 RoutingTransition 只能在 Job terminal 后，
    由 PendingJobTraceStore / FinalizedJobTrace 构造。
    """

    agent_id: str
    agent_index: int

    job_id: str

    # 真正做 Routing decision 的时间。
    decision_time: float

    # env.step() 完成后的环境时间。
    result_time: float

    action: int
    action_type: str
    action_source: str

    # 当前 Routing action 当时立即产生的 reward。
    immediate_reward: float

    # Self:
    #     当前 DC
    #
    # Edge:
    #     目标 Edge DC
    #
    # Cloud:
    #     cloud
    #
    # Drop:
    #     None
    target_dc_id: Optional[str]

    # 当前 Job 是否已经在本次调用中完成 Finalize。
    # 目前主要对应 forced drop。
    job_finalized: bool

    def as_dict(
            self,
    ) -> Dict[str, Any]:

        return {
            "agent_id": self.agent_id,
            "agent_index": self.agent_index,
            "job_id": self.job_id,
            "decision_time": self.decision_time,
            "result_time": self.result_time,
            "action": self.action,
            "action_type": self.action_type,
            "action_source": self.action_source,
            "immediate_reward": self.immediate_reward,
            "target_dc_id": self.target_dc_id,
            "job_finalized": self.job_finalized,
        }

# 采集器，从环境中抽取经验
class TransitionCollector:
    def __init__(
            self,
            env: CloudEdgeEnvLike,

            routing_observation_builder:
            RoutingObservationBuilder,

            routing_state_builder:
            RoutingCentralizedStateBuilder,

            pending_trace_store:
            PendingJobTraceStore,

            training_reward_model:
            HMasacTrainingRewardModel,
    ) -> None:

        self.env = env

        self.routing_observation_builder = (
            routing_observation_builder
        )

        self.routing_state_builder = (
            routing_state_builder
        )

        self.pending_trace_store = (
            pending_trace_store
        )
        self.training_reward_model = (
            training_reward_model
        )
        # ==========================================================
        # Collector 现在统计的是实际执行的 Routing actions，
        # 不再统计“生成了多少条 Replay Transition”。
        # ==========================================================

        self.total_routing_action_count = 0
        self.episode_routing_action_count = 0

        self._validate_environment_interface()

    # 重置计数器
    def reset_episode(self) -> None:
        self.episode_routing_action_count = 0




    # 获取当前环境所对应的决策状态快照
    def _build_current_decision_snapshot(self,) -> DecisionSnapshot:
        """
        根据环境当前时刻构造 Routing DecisionSnapshot。

        本函数只服务于 Routing 层 PettingZoo Agent。

        主要职责：
            1. 获取当前真正需要做 Routing 决策的 Edge DC；
            2. 构造 Routing Actor 的 local observation；
            3. 构造 Routing centralized critic state；
            4. 构造当前 DecisionSnapshot；
            5. 如果当前 Job 是之前某个 Edge->Edge Routing action
               到达下一 Edge 后产生的 same-job successor，
               则在 PendingJobTraceStore 中完成上一跳 next_state 回填。

        注意：
            Host 层不会调用本函数。
        """

        # ==========================================================
        # 1. 基础环境状态检查
        # ==========================================================

        self._require_reset()

        # 当前必须存在真正活跃的 PettingZoo Routing Agent。
        # Host phase 下 agent_selection=None，因此不能进入这里。
        agent_id = (
            self._get_live_selected_agent()
        )

        if self.env.current_job_id is None:
            raise RuntimeError(
                "构造 Routing DecisionSnapshot 时 "
                "current_job_id 为 None："
                f"agent={agent_id}"
            )

        job_id = str(
            self.env.current_job_id
        )

        # ==========================================================
        # 2. Routing Actor Local Observation
        #
        # 这里只构造 Routing 层 Observation。
        # Host Observation 由 HostObservationBuilder 单独负责。
        # ==========================================================

        local_obs = np.asarray(
            self.routing_observation_builder.build(
                agent_id
            ),
            dtype=np.float32,
        ).copy()

        # ==========================================================
        # 3. Routing Centralized Critic State
        #
        # 该 State 只用于 CTDE 训练阶段的 Routing Critic。
        # Routing Actor 执行阶段仍然只使用 local_obs。
        # ==========================================================

        global_state = np.asarray(
            self.routing_state_builder.build(),
            dtype=np.float32,
        ).copy()

        # ==========================================================
        # 4. 构造当前 Routing DecisionSnapshot
        # ==========================================================

        snapshot = DecisionSnapshot(
            agent_id=agent_id,

            agent_index=int(
                self.env.agent_name_mapping[
                    agent_id
                ]
            ),

            job_id=job_id,

            env_time=float(
                self.env.current_time
            ),

            local_obs=local_obs,

            global_state=global_state,

            forced_action=(
                self._get_forced_action(
                    job_id
                )
            ),
        )

        # ==========================================================
        # 5. Pending Job Causal Trace：
        #    回填上一跳 Edge -> Edge Routing 的真实 successor
        #
        # 例如：
        #
        #   Job42 @ DC1
        #       ↓ action = DC3
        #
        # 前一跳只记录：
        #
        #   state      = obs(Job42 @ DC1)
        #   action     = DC3
        #   next_state = UNKNOWN
        #
        # 当 Job42 真正到达 DC3，并进入本函数时，
        # 当前 snapshot 就是上一跳真正的 same-job successor：
        #
        #   next_state = obs(Job42 @ DC3)
        #
        # 如果当前 Job 没有未解析的 Edge predecessor，
        # resolve_routing_successor() 返回 False，
        # 不执行任何修改。
        # ==========================================================

        self.pending_trace_store.resolve_routing_successor(
            job_id=snapshot.job_id,

            next_agent_id=(
                snapshot.agent_id
            ),

            next_agent_index=(
                snapshot.agent_index
            ),

            next_env_time=(
                snapshot.env_time
            ),

            next_local_obs=(
                snapshot.local_obs
            ),

            next_global_state=(
                snapshot.global_state
            ),
        )

        # ==========================================================
        # 6. 返回当前 Routing Decision
        # ==========================================================

        return snapshot

    def capture_decision(self) -> DecisionSnapshot:
        self._require_reset()
        return self._build_current_decision_snapshot()
    # 这里相当于把原来的 capture_decision() 拆成 _build_current_decision_snapshot()->capture_decision(),这样后面 execute_and_collect() 也可以调用同一个函数

    def _validate_decision_is_current(
            self,
            decision: DecisionSnapshot,
    ) -> None:
        """
        检查 Trainer 保存的 Routing DecisionSnapshot
        是否仍然对应 Environment 当前真正等待执行的
        Routing decision。

        主要防止：

            1. Host Phase 后错误复用旧 Routing snapshot；
            2. 当前 Routing Agent 已经改变；
            3. 当前 Job 已经改变；
            4. 对过期状态执行 Actor action。

        注意：
            本函数只验证 Routing decision 的 identity，
            不负责构造 Observation 或执行 action。
        """

        # ==========================================================
        # 1. 当前必须仍然存在真正活跃的 Routing Agent。
        #
        # Host phase 下 agent_selection=None，
        # _get_live_selected_agent() 会直接拒绝。
        # ==========================================================

        current_agent_id = (
            self._get_live_selected_agent()
        )

        # ==========================================================
        # 2. Agent 必须与 Snapshot 一致。
        # ==========================================================

        if (
                current_agent_id
                != str(
            decision.agent_id
        )
        ):
            raise RuntimeError(
                "Routing DecisionSnapshot 已过期："
                f"snapshot_agent={decision.agent_id}, "
                f"current_agent={current_agent_id}, "
                f"job={decision.job_id}"
            )

        # ==========================================================
        # 3. 当前必须仍然存在 Routing Job。
        # ==========================================================

        if self.env.current_job_id is None:
            raise RuntimeError(
                "执行 Routing action 时 "
                "Environment current_job_id 为 None："
                f"snapshot_agent={decision.agent_id}, "
                f"snapshot_job={decision.job_id}"
            )

        current_job_id = str(
            self.env.current_job_id
        )

        # ==========================================================
        # 4. Job 必须与 Snapshot 一致。
        #
        # 如果不同，说明 Environment 已经推进到另一个 Job，
        # 不能再用旧 Observation 执行动作。
        # ==========================================================

        if (
                current_job_id
                != str(
            decision.job_id
        )
        ):
            raise RuntimeError(
                "Routing DecisionSnapshot Job 已过期："
                f"snapshot_job={decision.job_id}, "
                f"current_job={current_job_id}, "
                f"agent={current_agent_id}"
            )


    def execute_and_record(
            self,
            decision: DecisionSnapshot,
            action: int,
            action_source: str,
    ) -> Tuple[
        RoutingActionResult,
        Optional[DecisionSnapshot],
    ]:
        """
        执行一次 PettingZoo Routing action，
        并把已经真实发生的 Routing 因果事实写入
        PendingJobTraceStore。

        第二十一步以后，本函数不再生成任何正式训练 Transition。

        正式 RoutingTransition 只能在：

            Job terminal
                ↓
            FinalizedJobTrace
                ↓
            _build_finalized_routing_transitions()

        阶段生成。

        本函数只负责：

            DecisionSnapshot
                ↓
            Routing action
                ↓
            Environment.step()
                ↓
            immediate reward
                ↓
            PendingRoutingStep

        如果是 forced drop：

            mark terminal
                ↓
            finalize complete Job trace

        返回的 next_decision 只是 Trainer 环境时间线上的
        下一条 Routing Decision。

        它不一定属于当前 Job，
        因而绝不在本函数中被当作当前 Job 的 Replay next_state。
        """

        # ==========================================================
        # 1. 基础检查
        # ==========================================================

        self._require_reset()

        self._validate_decision_is_current(
            decision
        )

        action = int(
            action
        )

        action_source = str(
            action_source
        )

        if action_source not in {
            "forced",
            "random",
            "policy",
            "orchestrator",
        }:
            raise ValueError(
                "TransitionCollector 收到未知 action_source："
                f"{action_source}"
            )

        # ==========================================================
        # 2. Forced action 一致性检查
        #
        # 如果 DecisionSnapshot 已经明确告诉 Trainer：
        #
        #     forced_action = DROP_ACTION
        #
        # Trainer 就不能继续走 random / policy。
        # ==========================================================

        if decision.forced_action is not None:

            if action_source != "forced":
                raise RuntimeError(
                    "当前 Routing Decision 必须执行 forced action，"
                    "但 Trainer 给出的 action_source 不是 forced："
                    f"job={decision.job_id}, "
                    f"action_source={action_source}"
                )

            if (
                    int(decision.forced_action)
                    != action
            ):
                raise RuntimeError(
                    "Trainer 执行的 forced action "
                    "与 DecisionSnapshot 不一致："
                    f"job={decision.job_id}, "
                    f"expected={decision.forced_action}, "
                    f"actual={action}"
                )

        elif action_source == "forced":

            raise RuntimeError(
                "DecisionSnapshot 没有 forced action，"
                "但 Trainer 将当前动作标记为 forced："
                f"job={decision.job_id}, "
                f"action={action}"
            )

        # ==========================================================
        # 3. 使用 Environment 唯一 action semantics 解码。
        # ==========================================================

        (
            action_type,
            target_dc_id,
        ) = self._decode_routing_action(
            agent_id=(
                decision.agent_id
            ),
            action=action,
        )

        # ==========================================================
        # 4. 执行真正的 PettingZoo Routing action。
        #
        # Host action 永远不会进入本函数。
        # ==========================================================

        self.env.step(
            action
        )

        # ==============================================================
        # 5. 从 Environment 获取真实 Routing Action Facts。
        #
        # Environment 不再提供 H-MASAC training reward。
        # ==============================================================

        routing_action_facts = (
            self.env
                .pop_last_routing_action_facts()
        )

        # ==========================================================
        # 6. 保存本次 Routing action 的即时 reward。
        #
        # 必须按动作执行前的 decision.agent_id 获取，
        # 因为 env.step() 后当前 Agent / Job 都可能变化。
        # ==========================================================

        immediate_reward = float(
            self.training_reward_model
                .calculate_routing_immediate_reward(
                routing_action_facts
            )
        )

        # ==========================================================
        # 7. 写入 Pending Job Causal Trace。
        #
        # 这里只记录事实：
        #
        #     state
        #     action
        #     source / target
        #     immediate_reward
        #
        # Edge action 的 same-job next_state 仍然未知。
        # 后续真正到达目标 DC 时，
        # _build_current_decision_snapshot()
        # 会负责 resolve。
        # ==========================================================

        self.pending_trace_store.record_routing_step(
            job_id=(
                decision.job_id
            ),

            agent_id=(
                decision.agent_id
            ),

            agent_index=(
                decision.agent_index
            ),

            env_time=(
                decision.env_time
            ),

            local_obs=(
                decision.local_obs
            ),

            global_state=(
                decision.global_state
            ),

            action=action,

            action_type=(
                action_type
            ),

            action_source=(
                action_source
            ),

            immediate_reward=(
                immediate_reward
            ),

            target_dc_id=(
                target_dc_id
            ),
        )

        # ==========================================================
        # 8. Forced Drop：
        #
        # Drop penalty 已经保存在当前 Routing Step 的
        # immediate_reward 中。
        #
        # 因此：
        #   - 不额外创建 RewardEvent；
        #   - 只 mark terminal；
        #   - 然后立即 Finalize。
        # ==========================================================

        if action_type == "drop":
            self.pending_trace_store.mark_terminal(
                job_id=(
                    decision.job_id
                ),

                env_time=float(
                    self.env.current_time
                ),

                reason="forced_drop",
            )

            self.pending_trace_store.finalize_terminal_trace(
                job_id=(
                    decision.job_id
                )
            )

        # ==========================================================
        # 9. 创建 Trainer 环境时间线上的下一 Routing Decision。
        #
        # 重要：
        #
        #     next_decision
        #
        # 不是当前 Job 的 Replay next_state。
        #
        # 如果它恰好是当前 Job 的真实 Edge successor，
        # _build_current_decision_snapshot() 会通过 job_id
        # 正确 resolve 前一跳。
        # ==========================================================

        environment_done = (
            self._is_episode_done()
        )

        next_decision: Optional[
            DecisionSnapshot
        ] = None

        if (
                not environment_done
                and self.env.agent_selection
                is not None
        ):
            next_decision = (
                self._build_current_decision_snapshot()
            )

        # ==========================================================
        # 10. 只返回即时执行结果。
        #
        # 不返回：
        #     next_state
        #     done
        #     terminated
        #     truncated
        #
        # 这些属于 Job Finalize 后的 RoutingTransition。
        # ==========================================================

        routing_result = (
            RoutingActionResult(
                agent_id=str(
                    decision.agent_id
                ),

                agent_index=int(
                    decision.agent_index
                ),

                job_id=str(
                    decision.job_id
                ),

                decision_time=float(
                    decision.env_time
                ),

                result_time=float(
                    self.env.current_time
                ),

                action=action,

                action_type=(
                    action_type
                ),

                action_source=(
                    action_source
                ),

                immediate_reward=(
                    immediate_reward
                ),

                target_dc_id=(
                    target_dc_id
                ),

                job_finalized=(
                    self.pending_trace_store
                        .has_finalized_trace(
                        decision.job_id
                    )
                ),
            )
        )

        # ==========================================================
        # 11. Collector 计数。
        # ==========================================================

        self.total_routing_action_count += 1
        self.episode_routing_action_count += 1

        return (
            routing_result,
            next_decision,
        )

    # 按 PettingZoo AEC 约定清理一个已经终止的智能体
    def drain_one_dead_agent(
            self,
    ) -> bool:
        """
        按 PettingZoo AEC 约定，
        对当前已经 terminal / truncated 的 Routing Agent
        执行一次 step(None)。

        Host Phase 下 agent_selection=None，
        此时本函数什么都不做。
        """

        if not self.env.agents:
            return False

        if self.env.agent_selection is None:
            return False

        agent_id = str(
            self.env.agent_selection
        )

        is_dead = bool(
            self.env.terminations.get(
                agent_id,
                False,
            )
            or self.env.truncations.get(
                agent_id,
                False,
            )
        )

        if not is_dead:
            return False

        self.env.step(
            None
        )

        return True

    # 清理 episode 结束后的全部 dead agent，返回清理次数
    def drain_all_dead_agents(self) -> int:
        count = 0
        while self.env.agents:
            if not self.drain_one_dead_agent():
                break
            count += 1
        return count

    ############################## 辅助函数 ############################################

    # 构造阶段检查环境是否提供采集器需要的接口
    def _validate_environment_interface(self) -> None:
        required_attributes = (
            "possible_agents",
            "agents",
            "agent_selection",
            # "current_agent_id",
            "current_job_id",
            "current_time",
            # "local_obs_dim",
            # "global_state_dim",
            "action_dim",
            "drop_action",
            "has_reset",
            "agent_name_mapping",
            "rewards",
            "terminations",
            "truncations",
        )

        for name in required_attributes:
            if not hasattr(self.env, name):
                raise AttributeError(
                    f"环境缺少 TransitionCollector 所需属性：{name}。"
                )

        # observe、state 和 step 不仅要存在，还必须可以调用。
        for method_name in (
                "step",
                "_decode_action",
                "_should_drop_arrival_job",
        ):
            method = getattr(
                self.env,
                method_name,
                None,
            )

            if not callable(
                    method
            ):
                raise AttributeError(
                    "Environment 缺少 TransitionCollector "
                    "所需可调用方法："
                    f"{method_name}()"
                )


        if int(self.env.action_dim) <= 0:
            raise ValueError("action_dim 必须大于 0。")
        if (
                int(
                    self.routing_observation_builder
                            .obs_dim
                )
                <= 0
        ):
            raise ValueError(
                "routing_obs_dim 必须大于 0。"
            )

        if (
                int(
                    self.routing_state_builder
                            .state_dim
                )
                <= 0
        ):
            raise ValueError(
                "routing_global_state_dim "
                "必须大于 0。"
            )
    # 所有采集动作都要求环境先完成reset
    def _require_reset(self) -> None:
        if not bool(self.env.has_reset):
            raise RuntimeError(
                "环境尚未 reset，不能采集 Transition。"
            )

    # 返回活着的、当前决策的agent
    def _get_live_selected_agent(
            self,
    ) -> str:
        """
        返回当前真正处于 PettingZoo Routing Phase 的 Agent。

        Host Phase 下：
            agent_selection == None

        因此必须立即拒绝，
        不能把 None 转换成字符串 "None"。
        """

        if self.env.agent_selection is None:
            raise RuntimeError(
                "当前不存在 PettingZoo Routing Agent。"
                "如果存在 pending Host decision，"
                "应由 Local Host SAC 分支处理，"
                "不能调用 Routing TransitionCollector。"
            )

        agent_id = str(
            self.env.agent_selection
        )

        if agent_id not in self.env.possible_agents:
            raise RuntimeError(
                "当前 agent_selection 不是合法 Routing Agent："
                f"agent={agent_id}, "
                f"possible_agents={self.env.possible_agents}"
            )

        if agent_id not in self.env.agents:
            raise RuntimeError(
                "当前 Routing Agent 已不在活跃 agents 中："
                f"agent={agent_id}, "
                f"agents={self.env.agents}"
            )

        return agent_id

    # 判断整个 episode 结束
    def _is_episode_done(self) -> bool:
        if len(self.env.possible_agents) == 0:
            return True
        return all(
            bool(
                self.env.terminations.get(agent_id, False)
                or self.env.truncations.get(agent_id, False)
            )
            for agent_id in self.env.possible_agents
        )

    # 返回当前决策动作
    def _get_forced_action(self, job_id: str) -> Optional[int]:
        should_drop_fn = getattr(
            self.env, "_should_drop_arrival_job", None
        )
        if bool(should_drop_fn(job_id)):
            return int(self.env.drop_action)
        return None

    def _decode_routing_action(
            self,
            *,
            agent_id: str,
            action: int,
    ) -> Tuple[
        str,
        Optional[str],
    ]:
        """
        使用 Environment 的唯一动作语义定义解析 Routing action。

        Collector 不复制：
            Self index
            Edge index
            Cloud index

        DROP_ACTION=-1 同样由 Environment 解码。
        """

        decode_action = getattr(
            self.env,
            "_decode_action",
            None,
        )

        if not callable(
                decode_action
        ):
            raise AttributeError(
                "TransitionCollector 需要 Environment 提供 "
                "_decode_action()。"
            )

        decoded_action = decode_action(
            agent_id=str(
                agent_id
            ),
            action=int(
                action
            ),
        )

        action_type = str(
            decoded_action.get(
                "action_type",
                "",
            )
        )

        if action_type not in {
            "self",
            "edge_dc",
            "cloud",
            "drop",
        }:
            raise RuntimeError(
                "Environment 返回未知 Routing action_type："
                f"agent={agent_id}, "
                f"action={action}, "
                f"action_type={action_type}"
            )

        raw_target_dc_id = (
            decoded_action.get(
                "target_dc_id",
                None,
            )
        )

        target_dc_id = (
            None
            if raw_target_dc_id is None
            else str(
                raw_target_dc_id
            )
        )

        return (
            action_type,
            target_dc_id,
        )






