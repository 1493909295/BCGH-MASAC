from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float32]
MaskArray = NDArray[np.int8]

# 接口清单类 TransitionCollector 的对象，都应该至少具备它描述的这些接口
# CloudEdgeEnvLike 类不负责实现逻辑
class CloudEdgeEnvLike(Protocol):

    # pettingzoo 需要区分全部可能智能体与现在活着的智能体，在我的环境中这俩个没区别
    possible_agents: Sequence[str]
    agents: Sequence[str]

    # 当前执行step的智能体与当前做决策的智能体，在我环境中这两个也相同
    agent_selection: Optional[str]
    current_agent_id: Optional[str]

    current_job_id: Optional[str]
    current_time: float
    local_obs_dim: int
    global_state_dim: int
    action_dim: int
    drop_action: int
    has_reset: bool

    # 把str类型的agent id映射成数字
    agent_name_mapping: Mapping[str, int]
    rewards: Mapping[str, float]

    # 正常停止与异常停止标记，但我其实没实现异常停止 ·_·
    terminations: Mapping[str, bool]
    truncations: Mapping[str, bool]

    def observe(self, agent: str) -> Dict[str, np.ndarray]:
        ...
    def state(self) -> np.ndarray:
        ...
    def step(self, action: Optional[int]) -> None:
        ...

# 动作执行前快照捕捉，True代表不可重新赋值
@dataclass(frozen=True)
class DecisionSnapshot:
    agent_id: str
    agent_index: int
    job_id: str
    env_time: float
    local_obs: FloatArray
    action_mask: MaskArray
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
            "action_mask": self.action_mask.copy(),
            "global_state": self.global_state.copy(),
            "forced_action": self.forced_action,
        }

# 表示 replay buffer 中的一条训练经验
@dataclass(frozen=True)
class Transition:
    agent_id: str
    agent_index: int
    job_id: str
    env_time: float
    local_obs: FloatArray
    global_state: FloatArray
    action_mask: MaskArray
    action: int
    action_type: str
    reward: float
    next_agent_id: Optional[str]

    # next_agent_index 在终止状态下使用 -1 表示“不存在下一智能体”
    next_agent_index: int
    next_job_id: Optional[str]
    next_env_time: float
    next_local_obs: FloatArray
    next_global_state: FloatArray
    next_action_mask: MaskArray
    terminated: bool
    truncated: bool
    done: bool

    # 将 Transition 转成字典，ReplayBuffer 可以按键读取各字段
    def as_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_index": self.agent_index,
            "job_id": self.job_id,
            "env_time": self.env_time,
            "local_obs": self.local_obs.copy(),
            "global_state": self.global_state.copy(),
            "action_mask": self.action_mask.copy(),
            "action": self.action,
            "action_type": self.action_type,
            "reward": self.reward,
            "next_agent_id": self.next_agent_id,
            "next_agent_index": self.next_agent_index,
            "next_job_id": self.next_job_id,
            "next_env_time": self.next_env_time,
            "next_local_obs": self.next_local_obs.copy(),
            "next_global_state": self.next_global_state.copy(),
            "next_action_mask": self.next_action_mask.copy(),
            "terminated": self.terminated,
            "truncated": self.truncated,
            "done": self.done,
        }

# 采集器，从环境中抽取经验
class TransitionCollector:
    def __init__(self, env: CloudEdgeEnvLike, validate_actions: bool = True,) -> None:
        self.env = env
        self.validate_actions = bool(validate_actions)
        # 全局计数，不随episode清空
        self.total_transition_count = 0
        # 当前episode生成多少条经验
        self.episode_transition_count = 0
        # 在采集器实例化时就用环境接口检查
        self._validate_environment_interface()

    # 重置计数器
    def reset_episode(self) -> None:
        self.episode_transition_count = 0

    # 获取决策状态快照
    # def capture_decision(self) -> DecisionSnapshot:
    #     self._require_reset()
    #     agent_id = self._get_live_selected_agent()
    #     job_id = str(self.env.current_job_id)
    #     observation = self.env.observe(agent_id)
    #
    #     # 把局部观测转成独立 float32 数组，避免对同一块内存的引用
    #     local_obs = np.asarray(observation["observation"], dtype=np.float32).copy()
    #
    #     # 把动作掩码转成独立 int8 数组
    #     action_mask = np.asarray(observation["action_mask"],dtype=np.int8).copy()
    #
    #     # 获取同一决策时刻的集中式全局状态
    #     global_state = np.asarray(self.env.state(), dtype=np.float32).copy()
    #
    #     return DecisionSnapshot(
    #         agent_id=agent_id,
    #         agent_index=int(self.env.agent_name_mapping[agent_id]),
    #         job_id=job_id,
    #         env_time=float(self.env.current_time),
    #         local_obs=local_obs,
    #         action_mask=action_mask,
    #         global_state=global_state,
    #         forced_action=self._get_forced_action(job_id),
    #     )
    # 执行一次动作并生成完整的经验

    # 获取当前环境所对应的决策状态快照
    def _build_current_decision_snapshot(self) -> DecisionSnapshot:
        """
            根据“环境当前时刻”的状态构造一个 DecisionSnapshot。

            这个函数既可以：
            1. 在 episode 第一个决策点使用；
            2. 也可以在执行 action 后，直接构造下一决策点 next_decision。

            这样 state_(t+1) 只需要计算一次：
                transition_t.next_state
            和
                decision_(t+1).state
            可以直接复用同一时刻得到的数据，
            不需要下一轮训练循环再次调用 env.state() / env.observe()。
            """

        agent_id = self._get_live_selected_agent()
        job_id = str(self.env.current_job_id)
        observation = self.env.observe(agent_id)

        local_obs = np.asarray(observation["observation"], dtype=np.float32).copy()
        action_mask = np.asarray(observation["action_mask"], dtype=np.int8,).copy()
        global_state = np.asarray(self.env.state(), dtype=np.float32,)

        # 把当前环境状态完整封装成决策快照
        return DecisionSnapshot(
            agent_id=agent_id,
            # 将字符串形式的数据中心 ID 转换成网络使用的整数 ID。
            agent_index=int(
                self.env.agent_name_mapping[agent_id]
            ),
            job_id=job_id,
            # 当前仿真时间。
            env_time=float(self.env.current_time),
            local_obs=local_obs,
            action_mask=action_mask,
            global_state=global_state,
            # 如果当前任务由于超时等原因必须执行指定动作，
            # 则这里记录 forced_action；
            # 普通任务返回 None，由 Actor 决策。
            forced_action=self._get_forced_action(job_id),
        )

    def capture_decision(self) -> DecisionSnapshot:
        self._require_reset()
        return self._build_current_decision_snapshot()
    # 这里相当于把原来的 capture_decision() 拆成 _build_current_decision_snapshot()->capture_decision(),这样后面 execute_and_collect() 也可以调用同一个函数

    def execute_and_collect(self, decision: DecisionSnapshot, action: int,action_type: str,) -> Tuple[Transition, Optional[DecisionSnapshot]]:

        action_type = str(action_type)
        # 例行检查
        self._require_reset()

        # 执行动作、计算奖励、推进事件队列，并寻找下一决策点
        self.env.step(int(action))

        # 使用动作前智能体 ID 读取本次即时奖励
        reward = float(self.env.rewards[decision.agent_id])

        # 记录终止原因
        terminated = bool(self.env.terminations.get(decision.agent_id, False))
        truncated = bool(self.env.truncations.get(decision.agent_id, False))

        # 获取动作执行后的全局状态
        # next_global_state = np.asarray(self.env.state(), dtype=np.float32).copy()

        episode_done = self._is_episode_done()
        if episode_done:
            next_decision: Optional[DecisionSnapshot] = None
            next_agent_id: Optional[str] = None
            next_agent_index = -1
            next_job_id: Optional[str] = None
            next_local_obs = np.zeros(int(self.env.local_obs_dim), dtype=np.float32)
            next_action_mask = np.zeros(int(self.env.action_dim), dtype=np.int8)
            next_global_state = np.asarray(self.env.state(), dtype=np.float32)

        else:
            next_decision = self._build_current_decision_snapshot()
            next_agent_id = next_decision.agent_id
            next_job_id = next_decision.job_id
            next_agent_index = next_decision.agent_index
            # next_observation = self.env.observe(next_agent_id)
            next_local_obs = next_decision.local_obs.copy()
            next_action_mask = next_decision.action_mask.copy()
            next_global_state = next_decision.global_state.copy()

        transition = Transition(
            agent_id=decision.agent_id,
            agent_index=decision.agent_index,
            job_id=decision.job_id,
            env_time=decision.env_time,
            local_obs=decision.local_obs.copy(),
            global_state=decision.global_state.copy(),
            action_mask=decision.action_mask.copy(),
            action=int(action),
            action_type=action_type,
            reward=reward,
            next_agent_id=next_agent_id,
            next_agent_index=next_agent_index,
            next_job_id=next_job_id,
            next_env_time=float(self.env.current_time),
            next_local_obs=next_local_obs,
            next_global_state=next_global_state,
            next_action_mask=next_action_mask,
            terminated=terminated,
            truncated=truncated,
            done=episode_done,
        )
        self.total_transition_count += 1
        self.episode_transition_count += 1
        return transition, next_decision

    # 按 PettingZoo AEC 约定清理一个已经终止的智能体
    def drain_one_dead_agent(self) -> bool:
        if not self.env.agents:
            return False
        agent_id = str(self.env.agent_selection)
        is_dead = bool(
            self.env.terminations.get(agent_id, False)
            or self.env.truncations.get(agent_id, False)
        )
        if not is_dead:
            return False

        # 按 PettingZoo AEC 约定，对已经结束的智能体执行一次 None 动作。
        self.env.step(None)
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
            "current_agent_id",
            "current_job_id",
            "current_time",
            "local_obs_dim",
            "global_state_dim",
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
        for method_name in ("observe", "state", "step"):
            method = getattr(self.env, method_name, None)
            if not callable(method):
                raise AttributeError(
                    f"环境缺少可调用方法：{method_name}()。"
                )

        if int(self.env.local_obs_dim) <= 0:
            raise ValueError("local_obs_dim 必须大于 0。")

        if int(self.env.global_state_dim) <= 0:
            raise ValueError("global_state_dim 必须大于 0。")

        if int(self.env.action_dim) <= 0:
            raise ValueError("action_dim 必须大于 0。")

    # 所有采集动作都要求环境先完成reset
    def _require_reset(self) -> None:
        if not bool(self.env.has_reset):
            raise RuntimeError(
                "环境尚未 reset，不能采集 Transition。"
            )

    # 返回活着的、当前决策的agent
    def _get_live_selected_agent(self) -> str:
        agent_id = str(self.env.agent_selection)
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






