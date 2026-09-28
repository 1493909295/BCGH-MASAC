from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]
MaskArray = NDArray[np.int8]
BoolArray = NDArray[np.bool_]

# ReplayBuffer.add() 所要求的最小 Transition 接口
class TransitionLike(Protocol):
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
    next_agent_index: int
    next_job_id: Optional[str]
    next_env_time: float
    next_local_obs: FloatArray
    next_global_state: FloatArray
    next_action_mask: MaskArray
    terminated: bool
    truncated: bool
    done: bool

@dataclass(frozen=True)
class ReplayBatch:
    agent_indices: IntArray
    local_obs: FloatArray
    global_states: FloatArray
    action_masks: MaskArray
    actions: IntArray
    rewards: FloatArray
    next_agent_indices: IntArray
    next_local_obs: FloatArray
    next_global_states: FloatArray
    next_action_masks: MaskArray
    terminated: BoolArray
    truncated: BoolArray
    done: BoolArray
    is_forced_action: BoolArray
    buffer_indices: IntArray
    agent_ids: NDArray[np.object_]
    job_ids: NDArray[np.object_]
    next_agent_ids: NDArray[np.object_]
    next_job_ids: NDArray[np.object_]
    env_times: NDArray[np.float64]
    next_env_times: NDArray[np.float64]

    # 把 ReplayBatch 转成普通字典
    def as_dict(self) -> Dict[str, np.ndarray]:
        return {
            "agent_indices": self.agent_indices.copy(),
            "local_obs": self.local_obs.copy(),
            "global_states": self.global_states.copy(),
            "action_masks": self.action_masks.copy(),
            "actions": self.actions.copy(),
            "rewards": self.rewards.copy(),
            "next_agent_indices": self.next_agent_indices.copy(),
            "next_local_obs": self.next_local_obs.copy(),
            "next_global_states": self.next_global_states.copy(),
            "next_action_masks": self.next_action_masks.copy(),
            "terminated": self.terminated.copy(),
            "truncated": self.truncated.copy(),
            "done": self.done.copy(),
            "is_forced_action": self.is_forced_action.copy(),
            "buffer_indices": self.buffer_indices.copy(),
            "agent_ids": self.agent_ids.copy(),
            "job_ids": self.job_ids.copy(),
            "next_agent_ids": self.next_agent_ids.copy(),
            "next_job_ids": self.next_job_ids.copy(),
            "env_times": self.env_times.copy(),
            "next_env_times": self.next_env_times.copy(),
        }

class ReplayBuffer:
    def __init__(
            self,
            capacity: int,
            local_obs_dim: int,
            global_state_dim: int,
            action_dim: int,
            seed: Optional[int] = None,
            forced_action_value: int = -1,
    ) -> None:

        # 将输入统一转换成int
        capacity = int(capacity)
        local_obs_dim = int(local_obs_dim)
        global_state_dim = int(global_state_dim)
        action_dim = int(action_dim)
        forced_action_value = int(forced_action_value)

        self.capacity = capacity
        self.local_obs_dim = local_obs_dim
        self.global_state_dim = global_state_dim
        self.action_dim = action_dim
        self.forced_action_value = forced_action_value

        self.rng = np.random.default_rng(seed)
        self.seed = seed
        # 下一条经验写入哪个槽位
        self._position = 0
        # 有覆盖经验数，判断当前能否采样
        self._size = 0
        # 无覆盖，没啥用其实，可以统计训练进度
        self.total_added = 0

        self.agent_indices = np.full(
            (capacity,),
            -1,
            dtype=np.int64,
        )

        self.local_obs = np.zeros(
            (capacity, local_obs_dim),
            dtype=np.float32,
        )

        self.global_states = np.zeros(
            (capacity, global_state_dim),
            dtype=np.float32,
        )

        self.action_masks = np.zeros(
            (capacity, action_dim),
            dtype=np.int8,
        )

        self.actions = np.full(
            (capacity,),
            forced_action_value,
            dtype=np.int64,
        )

        self.actions = np.full(
            (capacity,),
            forced_action_value,
            dtype=np.int64,
        )

        self.rewards = np.zeros(
            (capacity,),
            dtype=np.float32,
        )

        self.next_agent_indices = np.full(
            (capacity,),
            -1,
            dtype=np.int64,
        )

        self.next_local_obs = np.zeros(
            (capacity, local_obs_dim),
            dtype=np.float32,
        )

        self.next_global_states = np.zeros(
            (capacity, global_state_dim),
            dtype=np.float32,
        )

        self.next_action_masks = np.zeros(
            (capacity, action_dim),
            dtype=np.int8,
        )

        self.terminated = np.zeros(
            (capacity,),
            dtype=np.bool_,
        )

        self.truncated = np.zeros(
            (capacity,),
            dtype=np.bool_,
        )

        self.done = np.zeros(
            (capacity,),
            dtype=np.bool_,
        )

        self.is_forced_action = np.zeros(
            (capacity,),
            dtype=np.bool_,
        )
        # 这三条用于性能优化
        self._trainable_indices = np.full(
            (capacity,),
            -1,
            dtype=np.int64,
        )
        self._trainable_positions = np.full(
            (capacity,),
            -1,
            dtype=np.int64,
        )
        self._num_trainable = 0
        self._num_forced_actions = 0

        self.env_times = np.zeros(
            (capacity,),
            dtype=np.float64,
        )

        self.next_env_times = np.zeros(
            (capacity,),
            dtype=np.float64,
        )

        # 四个固定长度的列表，用来保存每条经验对应的字符串 ID 信息
        self.agent_ids: list[Optional[str]] = [None] * capacity
        self.job_ids: list[Optional[str]] = [None] * capacity
        self.next_agent_ids: list[Optional[str]] = [None] * capacity
        self.next_job_ids: list[Optional[str]] = [None] * capacity
        self.latest_job_transition_index: Dict[str, int,] = {}
        # 记录调度链用于奖励分配
        self.job_transition_indices: Dict[str, list[int]] = {}
        # 保存未获得后续决策状态的边边调度转移
        self.pending_edge_transition_index: Dict[str, int] = {}
        self.action_types: list[Optional[str]] = [None] * capacity

    def __len__(self) -> int:
        return self._size

    # 重置当前 episode 的任务级信用分配索引
    def reset_episode_job_tracking(self) -> None:
        self.job_transition_indices.clear()
        self.latest_job_transition_index.clear()
        self.pending_edge_transition_index.clear()



    # 返回下一条经验将写入的槽位
    @property
    def position(self) -> int:
        return self._position

    # 维护索引关系
    def _add_trainable_index(self, buffer_index: int,) -> None:
        buffer_index = int(buffer_index)
        if int(self._trainable_positions[buffer_index]) != -1:
            return
        trainable_position = self._num_trainable
        self._trainable_indices[trainable_position] = buffer_index
        self._trainable_positions[buffer_index] = trainable_position
        self._num_trainable += 1

    def _remove_trainable_index(self,buffer_index: int,) -> None:
        buffer_index = int(buffer_index)
        remove_position = int(self._trainable_positions[buffer_index])
        if remove_position == -1:
            return
        last_position = self._num_trainable - 1
        last_buffer_index = int(self._trainable_indices[last_position])
        self._trainable_indices[remove_position] = last_buffer_index
        self._trainable_positions[last_buffer_index] = remove_position
        self._trainable_indices[last_position] = -1
        self._trainable_positions[buffer_index] = -1
        self._num_trainable -= 1

    # 回填 Edge 后继状态
    def _resolve_pending_edge_with_same_job_successor(
            self,
            job_id: str,
            successor: TransitionLike,
    ) -> bool:

        job_id = str(job_id)

        pending_index = self.pending_edge_transition_index.get(
            job_id
        )

        # 当前 Job 没有等待回填的 Edge transition。
        if pending_index is None:
            return False

        pending_index = int(pending_index)

        # ============================================================
        # 防御检查 1：
        # ReplayBuffer 是环形结构。
        # 如果 pending_index 对应的槽位已经被其他 Job 覆盖，
        # 就必须删除失效映射，绝不能修改别的 Job 的经验。
        # ============================================================
        if self.job_ids[pending_index] != job_id:
            self.pending_edge_transition_index.pop(
                job_id,
                None,
            )
            return False

        # ============================================================
        # 防御检查 2：
        # Edge transition 的 next_state 必须来自“同一个 Job”
        # 真正到达目标 Edge 后形成的下一次决策状态。
        #
        # 不允许把环境时间线上另一个 Job 的状态作为 next_state。
        # ============================================================
        if str(successor.job_id) != job_id:
            return False

        # ============================================================
        # 核心修改：
        # 使用同一 Job 在目标 Edge 上的决策状态，
        # 回填上一条 Edge transition 的真实 next_state。
        # ============================================================

        self.next_agent_indices[pending_index] = int(
            successor.agent_index
        )

        self.next_local_obs[pending_index] = np.asarray(
            successor.local_obs,
            dtype=np.float32,
        )

        self.next_global_states[pending_index] = np.asarray(
            successor.global_state,
            dtype=np.float32,
        )

        self.next_action_masks[pending_index] = np.asarray(
            successor.action_mask,
            dtype=np.int8,
        )

        self.next_agent_ids[pending_index] = str(
            successor.agent_id
        )

        self.next_job_ids[pending_index] = job_id

        self.next_env_times[pending_index] = float(
            successor.env_time
        )

        # 当前 Edge transition 的后继仍然是正常决策状态，
        # 所以不能被视为 terminal。
        self.terminated[pending_index] = False
        self.truncated[pending_index] = False
        self.done[pending_index] = False

        # Edge transition 现在已经拥有真实后继状态，
        # 可以正式加入 SAC 可训练经验集合。
        if (
                not bool(self.is_forced_action[pending_index])
                and int(self._trainable_positions[pending_index]) == -1
        ):
            self._add_trainable_index(
                pending_index
            )

        # 当前 pending Edge 已经成功解析。
        self.pending_edge_transition_index.pop(
            job_id,
            None,
        )

        return True

    # 当 Edge 转发后的 Job 在产生下一次正常 Actor 决策前已经直接失败时，把 pending Edge Transition 标记成终态
    def _finalize_pending_edge_as_terminal(self,job_id: str,) -> None:
        job_id = str(job_id)

        pending_index = self.pending_edge_transition_index.pop(
            job_id,
            None,
        )

        if pending_index is None:
            return

        pending_index = int(pending_index)

        if self.job_ids[pending_index] != job_id:
            return

        # ======================== 新增 ========================
        # 该 Edge 行为直接走向任务终止，不允许继续 bootstrap。
        self.next_agent_indices[pending_index] = -1
        self.next_local_obs[pending_index].fill(0.0)
        self.next_global_states[pending_index].fill(0.0)
        self.next_action_masks[pending_index].fill(0)

        self.next_agent_ids[pending_index] = None
        self.next_job_ids[pending_index] = None

        self.next_env_times[pending_index] = self.env_times[
            pending_index
        ]

        # MASAC 当前使用 done 控制是否 bootstrap。
        self.done[pending_index] = True
        # ====================================================

        # 此时 terminal failure reward 会直接写入该 Job 的因果链，
        # 该 Edge Transition 已经完整，可以参与训练。
        if (
                not bool(self.is_forced_action[pending_index])
                and int(self._trainable_positions[pending_index]) == -1
        ):
            self._add_trainable_index(
                pending_index
            )

    # 返回当前保存的环境强制动作经验数量
    @property
    def num_forced_actions(self) -> int:
        return int(
            self._num_forced_actions
        )

    # 返回当前保存的普通 Actor 动作经验数量
    @property
    def num_trainable_actions(self) -> int:
        return int(
            self._num_trainable
        )

    # 当前所有尚未进入 trainable 集合的 transition 数量
    @property
    def num_untrainable_actions(self) -> int:
        return int(
            self._size
            - self._num_trainable
        )

    # 非 forced、但仍不能训练的 transition 数量
    @property
    def num_nonforced_untrainable_actions(self,) -> int:
        return max(
            int(
                self.num_untrainable_actions
                - self.num_forced_actions
            ),
            0,
        )

    # 当前仍等待同 Job 后继状态回填的 Edge transition 数量
    @property
    def num_pending_edge_actions(self,) -> int:
        return int(
            len(
                self.pending_edge_transition_index
            )
        )

    # 保存一条 Transition
    # 当回放池已满时，本次写入会覆盖最旧经验
    def add(self, transition: TransitionLike) -> None:
        agent_index = int(transition.agent_index)
        action = int(transition.action)
        is_forced_action = (action == self.forced_action_value)
        current_job_id = str(transition.job_id)
        current_action_type = str(transition.action_type)
        local_obs = np.asarray(transition.local_obs, dtype=np.float32).copy()
        global_state = np.asarray(transition.global_state, dtype=np.float32).copy()
        action_mask = np.asarray(transition.action_mask, dtype=np.int8).copy()
        next_local_obs = np.asarray(transition.next_local_obs, dtype=np.float32).copy()
        next_global_state = np.asarray(transition.next_global_state, dtype=np.float32).copy()
        next_action_mask = np.asarray(transition.next_action_mask,dtype=np.int8).copy()

        reward = float(transition.reward)
        env_time = float(transition.env_time)
        next_env_time = float(transition.next_env_time)
        terminated = bool(transition.terminated)
        truncated = bool(transition.truncated)
        done = bool(transition.done)
        next_agent_index = int(transition.next_agent_index)
        index = self._position

        # 如果当前槽位即将覆盖旧经验，
        # 先清理旧 Job 对该槽位的映射。
        if self._size == self.capacity:
            self._remove_trainable_index(index)

            if bool(
                    self.is_forced_action[
                        index
                    ]
            ):
                self._num_forced_actions -= 1

            old_job_id = self.job_ids[
                index
            ]

            if old_job_id is not None:
                old_job_id = str(old_job_id)
                old_job_chain = self.job_transition_indices.get(old_job_id)

                if old_job_chain is not None:
                    try:
                        old_job_chain.remove(index)
                    except ValueError:
                        # 当前 index 已经不在映射中时无需处理。
                        pass

                        # 该 Job 已经没有任何仍被跟踪的 Transition，
                        # 删除空映射，避免字典不断积累空列表。
                    if len(old_job_chain) == 0:
                        self.job_transition_indices.pop(
                            old_job_id,
                            None,
                        )
                if (self.latest_job_transition_index.get(old_job_id) == index):
                    self.latest_job_transition_index.pop(
                        old_job_id,
                        None,
                    )
                if (self.pending_edge_transition_index.get(old_job_id) == index):
                    self.pending_edge_transition_index.pop(
                        old_job_id,
                        None,
                    )

        if not is_forced_action:
            self._resolve_pending_edge_with_same_job_successor(
                job_id=current_job_id,
                successor=transition,
            )




        # 将经过检查的数据写入对应数组
        self.agent_indices[index] = agent_index
        self.local_obs[index] = local_obs
        self.global_states[index] = global_state
        self.action_masks[index] = action_mask
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_agent_indices[index] = next_agent_index
        self.next_local_obs[index] = next_local_obs
        self.next_global_states[index] = next_global_state
        self.next_action_masks[index] = next_action_mask
        self.terminated[index] = terminated
        self.truncated[index] = truncated
        self.done[index] = done
        self.is_forced_action[index] = is_forced_action
        self.env_times[index] = env_time
        self.next_env_times[index] = next_env_time

        self.agent_ids[index] = str(transition.agent_id)

        # current_job_id = str(transition.job_id)
        self.job_ids[index] = current_job_id
        self.latest_job_transition_index[str(transition.job_id)] = index
        self.job_transition_indices.setdefault(current_job_id, [],).append(index)

        self.action_types[index] = current_action_type

        self.next_agent_ids[index] = (
            None
            if transition.next_agent_id is None
            else str(transition.next_agent_id)
        )
        self.next_job_ids[index] = (
            None
            if transition.next_job_id is None
            else str(transition.next_job_id)
        )

        if not is_forced_action:
            # e2e经验暂时不入池
            if current_action_type == "edge_dc":
                self.pending_edge_transition_index[current_job_id] = index
            else:
                self._add_trainable_index(index)

        # 写入后向后移一位
        self._position = (
            self._position + 1
        ) % self.capacity

        # 经验数量上限
        self._size = min(
            self._size + 1,
            self.capacity,
        )

        self.total_added += 1

    # 把一个延迟出现的任务结果奖励，修正到该 Job 最后一次调度 Transition 上。
    def apply_reward_correction(self, job_id: str, reward_delta: float,) -> Optional[str]:
        job_id = str(job_id)
        reward_delta = float(reward_delta)

        # O(1) 找到这个 Job 最新经验的位置。
        buffer_index = (self.latest_job_transition_index.get(job_id))

        # 经验可能已经被 ReplayBuffer 覆盖。
        if buffer_index is None:
            return None

        buffer_index = int(buffer_index)

        # 真正修正 SAC 将来采样到的 reward
        self.rewards[buffer_index] = np.float32(float(self.rewards[buffer_index]) + reward_delta)

        agent_id = self.agent_ids[buffer_index]

        if agent_id is None:
            return None

        return str(agent_id)

    # 反向折扣奖励分配机制
    def apply_discounted_terminal_reward(
            self,
            job_id: str,
            reward_delta: float,
            credit_decay: float,
    ) -> list[tuple[str, float]]:

        job_id = str(job_id)
        reward_delta = float(reward_delta)
        credit_decay = float(credit_decay)

        # ============================================================
        # 核心：
        # 只要进入 terminal reward 分配函数，
        # 就说明当前 Job 生命周期已经结束。
        #
        # 正常情况下：
        # Edge transition 应该早已通过同 Job successor 完成回填，
        # 此处不会找到 pending Edge。
        #
        # 如果此时仍存在 pending Edge，
        # 说明任务在产生下一次普通 Actor 决策前就终止了。
        # 这条 Edge transition 必须：
        #   1. 标记 done=True；
        #   2. 禁止 bootstrap；
        #   3. 加入 trainable replay。
        #
        # 这样无论成功还是失败，都不会留下永久悬挂的 Edge experience。
        # ============================================================
        self._finalize_pending_edge_as_terminal(
            job_id=job_id
        )

        # 找出当前 Job 的完整调度链。
        candidate_indices = list(
            self.job_transition_indices.get(
                job_id,
                [],
            )
        )

        if not candidate_indices:
            # 当前 Job 已经 terminal，
            # 即使没有普通 Actor transition，也必须清理任务级索引。
            self.job_transition_indices.pop(
                job_id,
                None,
            )
            self.latest_job_transition_index.pop(
                job_id,
                None,
            )

            return []

        valid_indices: list[int] = []

        for buffer_index in candidate_indices:

            buffer_index = int(buffer_index)

            if self.job_ids[buffer_index] != job_id:
                continue

            # forced action 不由 Actor 选择，
            # 不直接拿来训练 Actor/Critic。
            if bool(
                    self.is_forced_action[buffer_index]
            ):
                continue

            if self.agent_ids[buffer_index] is None:
                continue

            valid_indices.append(
                buffer_index
            )

        if not valid_indices:
            # 即使整个 Job 只有 forced transition，
            # terminal 后也不能继续保留任务级映射。
            self.job_transition_indices.pop(
                job_id,
                None,
            )

            self.latest_job_transition_index.pop(
                job_id,
                None,
            )

            return []

        chain_length = len(
            valid_indices
        )

        raw_weights = np.asarray(
            [
                credit_decay ** (
                        chain_length - 1 - position
                )
                for position in range(
                chain_length
            )
            ],
            dtype=np.float64,
        )

        normalized_weights = (
                raw_weights
                / float(raw_weights.sum())
        )

        applied_credits: list[
            tuple[str, float]
        ] = []

        for (
                buffer_index,
                weight,
        ) in zip(
            valid_indices,
            normalized_weights,
        ):
            terminal_credit = (
                    reward_delta
                    * float(weight)
            )

            self.rewards[
                buffer_index
            ] = np.float32(
                float(
                    self.rewards[
                        buffer_index
                    ]
                )
                + terminal_credit
            )

            applied_credits.append(
                (
                    str(
                        self.agent_ids[
                            buffer_index
                        ]
                    ),
                    terminal_credit,
                )
            )

        # Job 生命周期已经结束，
        # 删除本 Job 的临时追踪结构。
        self.job_transition_indices.pop(
            job_id,
            None,
        )

        self.latest_job_transition_index.pop(
            job_id,
            None,
        )

        # pending Edge 已在函数开头处理，
        # 理论上这里应该不存在。
        self.pending_edge_transition_index.pop(
            job_id,
            None,
        )

        return applied_credits



    # 判断当前是否有足够经验采样一个 batch
    def can_sample(self, batch_size: int, include_forced_actions: bool = False,) -> bool:
        batch_size = int(batch_size)
        if batch_size <= 0:
            return False
        if include_forced_actions:
            available_count = self._size
        else:
            available_count = self._num_trainable

        return int(available_count) >= batch_size

    # 随机采样一个ReplayBatch
    def sample(self,batch_size: int,include_forced_actions: bool = False,replace: bool = False,) -> ReplayBatch:

        # # 生成全部有效槽位下标
        # candidate_indices = np.arange(self._size,dtype=np.int64,)
        #
        # # 默认排除强制动作经验
        # if not include_forced_actions:
        #     forced_flags = self.is_forced_action[
        #         candidate_indices
        #     ]
        #     candidate_indices = candidate_indices[
        #         ~forced_flags
        #     ]
        #
        # # 当前满足条件的经验数量
        # available_count = int(candidate_indices.shape[0])
        #
        # # 随机选择经验槽位
        # sampled_indices = self.rng.choice(
        #     candidate_indices,
        #     size=batch_size,
        #     replace=bool(replace),
        # )

        batch_size = int(batch_size)
        replace = bool(replace)
        if include_forced_actions:
            available_count = self._size
            sampled_indices = self.rng.choice(
                available_count,
                size=batch_size,
                replace=replace,
            )
        else:
            available_count = (self._num_trainable)
            sampled_positions = self.rng.choice(
                available_count,
                size=batch_size,
                replace=replace,
            )
            sampled_indices = (
                self._trainable_indices[
                    sampled_positions
                ]
            )
        sampled_indices = np.asarray(
            sampled_indices,
            dtype=np.int64,
        )
        # 统一转换成 int64 数组
        sampled_indices = np.asarray(
            sampled_indices,
            dtype=np.int64,
        )

        # 根据槽位下标复制出一个独立 batch
        return ReplayBatch(
            agent_indices=self.agent_indices[sampled_indices].copy(),
            local_obs=self.local_obs[sampled_indices].copy(),
            global_states=self.global_states[sampled_indices].copy(),
            action_masks=self.action_masks[sampled_indices].copy(),
            actions=self.actions[sampled_indices].copy(),
            rewards=self.rewards[sampled_indices].copy(),
            next_agent_indices=self.next_agent_indices[sampled_indices].copy(),
            next_local_obs=self.next_local_obs[sampled_indices].copy(),
            next_global_states=self.next_global_states[sampled_indices].copy(),
            next_action_masks=self.next_action_masks[sampled_indices].copy(),
            terminated=self.terminated[sampled_indices].copy(),
            truncated=self.truncated[sampled_indices].copy(),
            done=self.done[sampled_indices].copy(),
            is_forced_action=self.is_forced_action[sampled_indices].copy(),
            buffer_indices=sampled_indices.copy(),
            agent_ids=np.asarray(
                [self.agent_ids[index] for index in sampled_indices],
                dtype=object,
            ),
            job_ids=np.asarray(
                [self.job_ids[index] for index in sampled_indices],
                dtype=object,
            ),
            next_agent_ids=np.asarray(
                [self.next_agent_ids[index] for index in sampled_indices],
                dtype=object,
            ),
            next_job_ids=np.asarray(
                [self.next_job_ids[index] for index in sampled_indices],
                dtype=object,
            ),
            env_times=self.env_times[sampled_indices].copy(),
            next_env_times=self.next_env_times[sampled_indices].copy(),
        )

    # 清空全部经验
    def clear(self) -> None:
        self._position = 0
        self._size = 0
        self.total_added = 0
        self._num_trainable = 0
        self._num_forced_actions = 0
        self._trainable_indices.fill(-1)
        self._trainable_positions.fill(-1)
        self.agent_indices.fill(-1)
        self.local_obs.fill(0.0)
        self.global_states.fill(0.0)
        self.action_masks.fill(0)
        self.actions.fill(self.forced_action_value)
        self.rewards.fill(0.0)
        self.next_agent_indices.fill(-1)
        self.next_local_obs.fill(0.0)
        self.next_global_states.fill(0.0)
        self.next_action_masks.fill(0)
        self.terminated.fill(False)
        self.truncated.fill(False)
        self.done.fill(False)
        self.is_forced_action.fill(False)
        self.env_times.fill(0.0)
        self.next_env_times.fill(0.0)
        self.agent_ids = [None] * self.capacity
        self.job_ids = [None] * self.capacity
        self.next_agent_ids = [None] * self.capacity
        self.next_job_ids = [None] * self.capacity
        self.latest_job_transition_index.clear()
        self.job_transition_indices.clear()
        self.action_types = [None] * self.capacity
        self.latest_job_transition_index.clear()
        self.job_transition_indices.clear()
        self.pending_edge_transition_index.clear()




