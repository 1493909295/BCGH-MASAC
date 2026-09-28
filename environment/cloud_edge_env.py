
from gymnasium import spaces
from pettingzoo import AECEnv
# from pettingzoo.utils import AgentSelector
import heapq
import numpy as np
import functools

import math
import copy
from typing import Any, Dict, List, Optional, Tuple,Union
import networkx as nx
from environment.datacenter import (Host, DataCenter, )
from environment.job import (Job, jobs_generate,load_job_dataset,)
from environment.env_generate import (EnvGenerator, UseOldEnv)
from environment.energy_model import (
    calculate_edge_host_power_components_w,
    calculate_cloud_attributable_power_w,
    calculate_edge_task_attributable_compute_energy_j,
    calculate_cloud_task_attributable_compute_energy_j,
    calculate_transfer_energy_j,)

import config as conf

JOB_ARRIVAL = "JOB_ARRIVAL"
JOB_FINISH = "JOB_FINISH"
DROP_ACTION = -1

class CloudEdgeEnv(AECEnv):
    #  环境元信息
    metadata = {
        "name": "cloud_edge_masac_v0",
        "render_modes": [],
        "is_parallelizable": False,
    }

    JOB_FEAT_DIM = 4
    DC_FEAT_DIM = 8
    HOST_FEAT_DIM = 9
    TIME_EPS = 1e-9

    def __init__(
        self,
        env_source: Optional[Union[EnvGenerator, UseOldEnv]] = None,
        old_env_path: Optional[str] = None,
        seed: Optional[int] = None,
        # invalid_action_mode: str = "raise",
    ):
        super().__init__()

        # 随机数种子生成器用于任务分配
        self.rng = np.random.default_rng(seed)
        self.seed_value = seed

########################################### 环境准备与验证模块 ############################################################
        # 物理环境选择模块，判断是新生成环境还是用旧的环境
        if env_source is not None:
            self.env_source = env_source
        elif old_env_path is not None:
            self.env_source = UseOldEnv(old_env_path)
        else:
            self.env_source = EnvGenerator()
            self.env_source.generate_environment(
                # lambda_rate=self.env_source.lambda_rate,
                # job_dataset_path=self.env_source.job_dataset_path,
                cloud_latency_range=self.env_source.cloud_latency_range,
                edge_latency_range=self.env_source.edge_latency_range,
            )

        # 环境完备性检查
        required_attrs = ["global_dc_list","datacenter_graph",]
        for attr in required_attrs:
            if not hasattr(self.env_source, attr):
                raise AttributeError(
                    f"env_source 缺少必要属性 {attr}，"
                    f"请检查 EnvGenerator 或 UseOldEnv 是否正确生成/加载环境。"
                )

        if len(self.env_source.global_dc_list) == 0:
            raise ValueError("环境中的 global_dc_list 为空，无法进行训练。")
        if self.env_source.datacenter_graph is None:
            raise ValueError("环境中的 datacenter_graph 为空，无法计算节点间时延。")

        # 保存一份干净的基础环境副本,避免一个 episode 修改 host 队列、job 状态后污染下一个 episode
        self.base_datacenters = copy.deepcopy(self.env_source.global_dc_list)
        self.base_graph = copy.deepcopy(self.env_source.datacenter_graph)

        source_job_num = int(getattr(self.env_source, "job_num",conf.NUM_JOBS,))
        self.job_num = (
            source_job_num
            if source_job_num > 0
            else int(conf.NUM_JOBS)
        )
        source_lambda_rate = float(getattr(self.env_source, "lambda_rate", conf.LAMBDA_RATE,))
        self.lambda_rate = (
            source_lambda_rate
            if source_lambda_rate > 0.0
            else float(conf.LAMBDA_RATE)
        )
        self.job_dataset_path = str(
            getattr(
                self.env_source,
                "job_dataset_path",
                conf.JOB_DATASET_PATH,
            )
            or conf.JOB_DATASET_PATH
        )


        # 云节点定义
        self.cloud_id = "cloud"
        self.enable_cloud_action = bool(conf.ENABLE_CLOUD_ACTION)
        dc_ids = [dc.dc_id for dc in self.base_datacenters]

        # 提取边缘dc id
        self.edge_dc_ids: List[str] = [
            dc.dc_id
            for dc in self.base_datacenters
            if dc.dc_id != self.cloud_id
        ]
        if len(self.edge_dc_ids) == 0:
            raise ValueError("环境中没有可作为智能体的边缘数据中心。")

        # 根据 config 中的单一模式开关构造每个 Edge DC 的初始任务到达分布。
        self._configure_dc_arrival_profile()


        # 边缘 DC 在前，cloud 放在最后
        self.all_dc_ids = self.edge_dc_ids + [self.cloud_id]

        # 记录数据中心数量
        self.num_edge_dc = len(self.edge_dc_ids)
        self.num_all_dc = len(self.all_dc_ids)

        # 计算边缘数据中心中最大的 host 数量，不同 DC 的 host 数可能不同。为了构造统一动作空间，需要用最大 host 数量对齐。
        self.max_host_num = max(
            len(dc.host_list)
            for dc in self.base_datacenters
            if dc.dc_id != self.cloud_id
        )

        self.possible_agents = (
            self.edge_dc_ids[:]
        )

        self.agents: List[str] = []

        self.agent_ids = (
            self.possible_agents[:]
        )

        self.num_edge_agents = len(
            self.edge_dc_ids
        )

        self.agent_name_mapping = {
            agent: i
            for i, agent
            in enumerate(
                self.possible_agents
            )
        }

        self.routing_action_target_dc_ids = (
                list(self.edge_dc_ids)
                + (
                    [self.cloud_id]
                    if self.enable_cloud_action
                    else []
                )
        )

        # Routing Action 编码
        self.routing_action_to_dc_id = {
            action_idx: str(dc_id)
            for action_idx, dc_id
            in enumerate(
                self.routing_action_target_dc_ids
            )
        }

        self.routing_dc_id_to_action = {
            str(dc_id): action_idx
            for action_idx, dc_id
            in self.routing_action_to_dc_id.items()
        }

        # cloud开关
        self.cloud_action_index = (
            self.routing_dc_id_to_action.get(
                self.cloud_id
            )
        )

        # Action dimension 只由真实结构动作数量决定。
        self.action_dim = len(
            self.routing_action_target_dc_ids
        )

        if self.action_dim <= 0:
            raise ValueError(
                "Routing action space 为空。"
            )

        self.action_spaces = {
            agent_id:
                spaces.Discrete(
                    self.action_dim
                )
            for agent_id
            in self.agent_ids
        }

        self.single_action_space = (
            spaces.Discrete(
                self.action_dim
            )
        )

        ######## pettingzoo 要求这俩种获取智能体方法都得有，所以都得写
        # 自己定义的方法，获取边缘数据中心智能体总数
        self.num_edge_agents = len(self.edge_dc_ids)

        # 获取环境中所有坑智能体数量
        self.possible_agents = self.edge_dc_ids[:]
        # 获取当前活跃智能体数量
        self.agents: List[str] = []
        self.agent_ids = self.possible_agents[:]




        self.agent_name_mapping = {
            agent: i
            for i, agent in enumerate(self.possible_agents)
        }

        ########################################################################################################################


        # 事件队列
        # 队内元素固定为 (event_time, event_type, job_id)
        self.event_queue: List[Tuple[float, str, str]] = []

        # 当前时间
        self.current_time: float = 0.0

        # 能耗上次结算时间点
        self.last_energy_update_time: float = 0.0

        # 能耗数据
        self.edge_idle_energy_j: float = 0.0
        self.edge_cpu_dynamic_energy_j: float = 0.0
        self.edge_gpu_dynamic_energy_j: float = 0.0
        self.cloud_compute_energy_j: float = 0.0
        self.transfer_energy_j: float = 0.0
        self.edge_edge_transfer_energy_j: float = 0.0
        self.edge_cloud_transfer_energy_j: float = 0.0

        # 当前需要决策的智能体、需要调度的job
        self.current_agent_id = None
        self.current_job_id = None
        self.current_dc_id = None
        self.pending_host_job_id = None
        self.pending_host_dc_id = None
        self.running_job_location = {}

        # 丢弃机制
        self.sla_deadline_ratio = float(conf.SLA_DEADLINE_RATIO)
        self.drop_deadline_ratio = float(conf.DROP_DEADLINE_RATIO)
        self.drop_action = DROP_ACTION

        self.dropped_jobs_info: List[Dict[str, Any]] = []

        # 等待队列统计
        self.queued_jobs: int = 0
        self.started_from_waiting_jobs: int = 0
        self.waiting_timeout_drops: int = 0
        self.max_waiting_queue_length: int = 0
        self.queue_length_scale = float(conf.QUEUE_LENGTH_SCALE)

        # 先暂存每个智能体重做，然后再统一执行，我这里貌似不需要这样，我这里来一个任务，选个智能体做个决策，更新下奖励就行
        # self.pending_actions: Dict[str, Optional[int]] = {
        #     agent_id: None
        #     for agent_id in self.possible_agents
        # }


        # 局部观测空间
        # 任务特征维度4，两个资源需求，一个执行时间，一个等待时间
        self.job_feat_dim = self.JOB_FEAT_DIM
        # dc特征维度8，两个资源，两个负载，两个可用，一个等待队列一个执行队列
        self.dc_feat_dim = self.DC_FEAT_DIM
        # host特征维度8，和dc一样
        self.host_feat_dim = self.HOST_FEAT_DIM
        # 链路特征
        self.local_link_target_dc_ids = self.edge_dc_ids + [self.cloud_id]
        self.link_feat_dim = self.num_edge_dc + 1

        # 单个智能体局部观测维度
        self.local_obs_dim = (
                self.job_feat_dim
                + self.dc_feat_dim
                + self.max_host_num * self.host_feat_dim
                + self.link_feat_dim
        )

        # 每个智能体的局部观测空间
        self.local_observation_spaces = {
            agent_id: spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.local_obs_dim,),
                dtype=np.float32,
            )
            for agent_id in self.possible_agents
        }

        # critic 的全局状态空间
        self.global_state_dim = (
                self.job_feat_dim
                +self.num_all_dc * self.dc_feat_dim
                + self.num_all_dc * self.max_host_num * self.host_feat_dim
                + self.num_all_dc * self.num_all_dc
        )
        self.state_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.global_state_dim,),
            dtype=np.float32,
        )
        self.global_state_space = self.state_space

        # 缓存当前状态，性能优化用的
        self._cached_global_state = np.zeros(
            self.global_state_dim,
            dtype=np.float32,
        )

        # 第一套动作掩码，修复 host 维度使用最大host数可能造成的问题
        self.action_mask_spaces = {
            agent_id: spaces.MultiBinary(self.action_dim)
            for agent_id in self.possible_agents
        }

        self.observation_spaces = {
            agent_id: spaces.Dict(
                {
                    "observation": self.local_observation_spaces[agent_id],
                    "action_mask": self.action_mask_spaces[agent_id],
                }
            )
            for agent_id in self.possible_agents
        }
        self.single_observation_space = self.observation_spaces[self.possible_agents[0]]

        self.rewards = {
            agent_id: 0.0
            for agent_id in self.possible_agents
        }

        self._cumulative_rewards = {
            agent_id: 0.0
            for agent_id in self.possible_agents
        }

        # pettingzoo要求记录智能体是否已经自然停止
        self.terminations = {
            agent_id: False
            for agent_id in self.possible_agents
        }

        # pettingzoo要求记录智能体是否因外部限制被强行结束
        self.truncations = {
            agent_id: False
            for agent_id in self.possible_agents
        }

        # 智能体保存的额外信息
        self.infos = {
            agent_id: {
                "agent_index": self.agent_name_mapping[agent_id],
                "dc_id": agent_id,
                "global_state": np.zeros(self.global_state_dim, dtype=np.float32),
                "action_mask": np.ones(self.action_dim, dtype=np.int8),
            }
            for agent_id in self.possible_agents
        }

        # 选择执行动作的智能体（任务到达位置）
        self.agent_selection = None

        # 为当前episode准备运行状态
        self.jobs: List[Job] = []
        self.datacenters: List[DataCenter] = []
        self.graph: Optional[nx.Graph] = None
        self.job_map: Dict[str, Job] = {}
        self.dc_map: Dict[str, DataCenter] = {}

        # 环境重置标记
        self.has_reset = False

        # self.energy_normalization_j = float(conf.ENERGY_NORMALIZATION_J)

        self.pending_job_outcome_events: List[
            Dict[str, Any]
        ] = []

        # 最近一次 Routing action 真实产生的物理事实。
        #
        # env.step() 不返回值，因此 Collector 在 step() 结束后
        # 通过 pop_last_routing_action_facts() 获取。
        self.last_routing_action_facts: Optional[
            Dict[str, Any]
        ] = None

        self._init_normalization_stats()
        
    # PettingZoo 标准接口，返回指定 agent 的观测空间。
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    # PettingZoo 标准接口，返回指定 agent 的动作空间。
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    # MASAC centralized critic 使用的全局状态接口
    # def state(self):
    #     if hasattr(self, "_get_global_state"):
    #         return self._get_global_state()
    #     return np.zeros(self.global_state_dim, dtype=np.float32)
    def state(self) -> np.ndarray:
        return self._cached_global_state.copy()

    # 清空当前 Episode 的系统级物理能源账本
    def _reset_episode_energy_counters(self,) -> None:
        self.edge_idle_energy_j = 0.0
        self.edge_cpu_dynamic_energy_j = 0.0
        self.edge_gpu_dynamic_energy_j = 0.0
        self.cloud_compute_energy_j = 0.0
        self.transfer_energy_j = 0.0
        self.edge_edge_transfer_energy_j = 0.0
        self.edge_cloud_transfer_energy_j = 0.0

    # 返回当前 Episode 已累计的全部 Edge IT Energy
    def get_edge_total_energy_j(self,) -> float:
        return float(self.edge_idle_energy_j + self.edge_cpu_dynamic_energy_j + self.edge_gpu_dynamic_energy_j)

    # 返回当前 Episode 已累计的整个 Cloud-Edge 系统物理能耗
    def get_total_system_energy_j(self,) -> float:
        return float(self.get_edge_total_energy_j() + self.cloud_compute_energy_j + self.transfer_energy_j)

    # 系统计算能耗积分
    def _update_compute_energy_to(self, new_time: float,) -> None:

        new_time = float(new_time)

        # 当前尚未完成能耗结算的时间区间长度。
        delta_t = (new_time - float(self.last_energy_update_time))

        if delta_t <= self.TIME_EPS:
            return

        edge_idle_power_w = 0.0
        edge_cpu_dynamic_power_w = 0.0
        edge_gpu_dynamic_power_w = 0.0

        # 统计当前时刻整个 Edge 系统
        for edge_dc_id in self.edge_dc_ids:
            edge_dc = self.dc_map[edge_dc_id]
            for host in edge_dc.host_list:
                power_components = (calculate_edge_host_power_components_w( host=host,))

                edge_idle_power_w += float(power_components["idle_power_w"])
                edge_cpu_dynamic_power_w += float(power_components["cpu_dynamic_power_w"])
                edge_gpu_dynamic_power_w += float(power_components["gpu_dynamic_power_w"])

        self.edge_idle_energy_j += (edge_idle_power_w * delta_t)
        self.edge_cpu_dynamic_energy_j += (edge_cpu_dynamic_power_w * delta_t)
        self.edge_gpu_dynamic_energy_j += (edge_gpu_dynamic_power_w * delta_t)

        # 统计云的
        cloud_dc = self.dc_map[self.cloud_id]
        cloud_used_cpu = 0.0
        cloud_used_gpu = 0.0

        for cloud_host in cloud_dc.host_list:
            cloud_used_cpu += float(cloud_host.used_cpu)
            cloud_used_gpu += float(cloud_host.used_gpu)

        cloud_power_w = (
            calculate_cloud_attributable_power_w(
                used_cpu=cloud_used_cpu,
                used_gpu=cloud_used_gpu,
            )
        )
        self.cloud_compute_energy_j += (float(cloud_power_w) * delta_t)

        # 更新系统能耗计算时间
        self.last_energy_update_time = (new_time)

    # 时间推进接口
    def _advance_simulation_time(self, new_time: float,) -> None:
        new_time = float(new_time)
        self._update_compute_energy_to(new_time)
        self.current_time = new_time

    def _configure_dc_arrival_profile(self) -> None:
        """
        根据 config.DC_ARRIVAL_MODE 构造 Edge DC 初始到达概率。

        uniform 保留原有等概率分配语义；heterogeneous 使用配置中的
        DC 角色与权重。两种模式都保持全局 self.lambda_rate 不变。
        """
        mode = str(
            getattr(
                conf,
                "DC_ARRIVAL_MODE",
                "uniform",
            )
        ).strip().lower()

        valid_modes = {
            "uniform",
            "heterogeneous",
        }
        if mode not in valid_modes:
            raise ValueError(
                "DC_ARRIVAL_MODE 只支持 'uniform' 或 'heterogeneous'，"
                f"当前值为 {mode!r}。"
            )

        if mode == "uniform":
            roles = {
                dc_id: "uniform"
                for dc_id in self.edge_dc_ids
            }
            weights = {
                dc_id: 1.0
                for dc_id in self.edge_dc_ids
            }

        else:
            role_by_id_raw = getattr(
                conf,
                "DC_ARRIVAL_ROLE_BY_ID",
                None,
            )
            role_weights_raw = getattr(
                conf,
                "DC_ARRIVAL_ROLE_WEIGHTS",
                None,
            )

            if not isinstance(role_by_id_raw, dict):
                raise TypeError(
                    "heterogeneous 模式要求 "
                    "DC_ARRIVAL_ROLE_BY_ID 为字典。"
                )
            if not isinstance(role_weights_raw, dict):
                raise TypeError(
                    "heterogeneous 模式要求 "
                    "DC_ARRIVAL_ROLE_WEIGHTS 为字典。"
                )

            configured_dc_ids = {
                str(dc_id)
                for dc_id in role_by_id_raw
            }
            actual_dc_ids = set(self.edge_dc_ids)
            missing_dc_ids = sorted(
                actual_dc_ids - configured_dc_ids
            )
            unknown_dc_ids = sorted(
                configured_dc_ids - actual_dc_ids
            )
            if missing_dc_ids or unknown_dc_ids:
                raise ValueError(
                    "DC_ARRIVAL_ROLE_BY_ID 必须与当前 Edge DC 完全一致；"
                    f"缺失={missing_dc_ids}，多余={unknown_dc_ids}。"
                )

            roles = {}
            weights = {}
            for dc_id in self.edge_dc_ids:
                role = str(
                    role_by_id_raw[dc_id]
                ).strip().lower()
                if role not in role_weights_raw:
                    raise ValueError(
                        f"DC {dc_id} 的到达角色 {role!r} "
                        "未在 DC_ARRIVAL_ROLE_WEIGHTS 中配置。"
                    )

                weight = float(
                    role_weights_raw[role]
                )
                if not np.isfinite(weight) or weight <= 0.0:
                    raise ValueError(
                        f"DC {dc_id} 的到达权重必须为有限正数，"
                        f"当前值为 {weight!r}。"
                    )

                roles[dc_id] = role
                weights[dc_id] = weight

        total_weight = float(sum(weights.values()))
        if not np.isfinite(total_weight) or total_weight <= 0.0:
            raise ValueError("所有 DC 到达权重之和必须为有限正数。")

        probabilities = {
            dc_id: float(weights[dc_id] / total_weight)
            for dc_id in self.edge_dc_ids
        }
        configured_rates = {
            dc_id: float(
                self.lambda_rate * probabilities[dc_id]
            )
            for dc_id in self.edge_dc_ids
        }

        probability_vector = np.asarray(
            [
                probabilities[dc_id]
                for dc_id in self.edge_dc_ids
            ],
            dtype=np.float64,
        )
        if not np.isclose(
                float(np.sum(probability_vector)),
                1.0,
                rtol=0.0,
                atol=1e-12,
        ):
            raise RuntimeError("DC 到达概率归一化失败。")

        self.dc_arrival_mode = mode
        self.dc_arrival_roles = roles
        self.dc_arrival_weights = weights
        self.dc_arrival_probabilities = probabilities
        self.dc_configured_arrival_rates = configured_rates
        self.dc_arrival_profile = {
            dc_id: {
                "role": roles[dc_id],
                "weight": weights[dc_id],
                "probability": probabilities[dc_id],
                "configured_arrival_rate": configured_rates[dc_id],
            }
            for dc_id in self.edge_dc_ids
        }
        self._dc_arrival_probability_vector = probability_vector

    # 为当前 Episode 生成一套全新的任务 workload
    def _generate_episode_workload(self,) -> List[Job]:
        seed_tag = (
            str(self.seed_value)
            if self.seed_value is not None
            else "none"
        )

        jobs = jobs_generate(
            job_num=self.job_num,
            lambda_rate=self.lambda_rate,
            job_dataset_path=self.job_dataset_path,
            wait_assign_jobs_list=[],
            rng=self.rng,
            job_id_prefix=(
                f"episode_seed_{seed_tag}"
            ),
        )

        # 按照到达时间排序
        jobs.sort(key=lambda job: float(job.arrive_time))

        exogenous_arrival_counts = {
            dc_id: 0
            for dc_id in self.edge_dc_ids
        }

        for job in jobs:
            # uniform 模式保留原有不传 p 的随机选择路径，尽量维持既有
            # seed 下的基线行为；heterogeneous 模式按配置权重进行泊松分流。
            if self.dc_arrival_mode == "uniform":
                arrived_dc_id = str(
                    self.rng.choice(
                        self.edge_dc_ids
                    )
                )
            else:
                arrived_dc_id = str(
                    self.rng.choice(
                        self.edge_dc_ids,
                        p=(
                            self._dc_arrival_probability_vector
                        ),
                    )
                )

            job.set_origin_datacenter(arrived_dc_id)
            job.set_target_datacenter(arrived_dc_id)
            exogenous_arrival_counts[arrived_dc_id] += 1

        self.episode_exogenous_arrival_counts = (
            exogenous_arrival_counts
        )

        return jobs

    # 每轮新训练重启环境
    def reset(self,seed: Optional[int] = None, options: Optional[dict] = None):

        # 如果传入新seed，更新随机数生成器
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.seed_value = seed

        self.agents = self.possible_agents[:]

        # 从base_jobs模板恢复当前 episode 的任务、数据中心和拓扑图。
        # self.jobs = copy.deepcopy(self.base_jobs)
        self.datacenters = copy.deepcopy(self.base_datacenters)
        self.graph = copy.deepcopy(self.base_graph)
        self.jobs = (self._generate_episode_workload())

        # 归一化统一不同特征的尺度
        # self._init_normalization_stats()

        # 每次episode 开始时，重新随机指定每个 job 到达哪个边缘数据中心
        # for job in self.jobs:
        #     arrived_dc_id = str(self.rng.choice(self.edge_dc_ids))
        #     job.set_target_datacenter(arrived_dc_id)

        # 构建快速查询映射。
        self.job_map = {
            str(job.job_id): job
            for job in self.jobs
        }
        self.dc_map = {
            dc.dc_id: dc
            for dc in self.datacenters
        }

        # 重置事件队列
        self.event_queue = []
        for job in self.jobs:
            heapq.heappush(
                self.event_queue,
                (
                    float(job.arrive_time),
                    JOB_ARRIVAL,
                    str(job.job_id),
                )
            )

        # 重置仿真时间
        self.current_time = 0.0
        self.last_energy_update_time = 0.0

        # 重置能耗统计
        self._reset_episode_energy_counters()

        # 重置当前决策相关状态
        self.current_agent_id = None
        self.current_job_id = None
        self.current_dc_id = None

        # 重置当前 Host 层待决策状态
        self.pending_host_job_id = None
        self.pending_host_dc_id = None

        # 记录运行中任务的位置，reset 时清空
        self.running_job_location = {}
        self.dropped_jobs_info = []

        self.queued_jobs = 0
        self.started_from_waiting_jobs = 0
        self.waiting_timeout_drops = 0
        self.max_waiting_queue_length = 0
        # 每个 Episode 都重新清空 Environment Fact Queue。
        self.pending_job_outcome_events = []

        # 当前还没有执行 Routing action。
        self.last_routing_action_facts = None

        # 重置 PettingZoo AEC 必需状态
        self.rewards = {
            agent_id: 0.0
            for agent_id in self.possible_agents
        }
        self._cumulative_rewards = {
            agent_id: 0.0
            for agent_id in self.possible_agents
        }
        self.terminations = {
            agent_id: False
            for agent_id in self.possible_agents
        }
        self.truncations = {
            agent_id: False
            for agent_id in self.possible_agents
        }

        # 从事件队列中找到第一个任务到达事件
        first_decision_found = False
        while self.event_queue:
            event_time, event_type, job_id = heapq.heappop(self.event_queue)
            self._advance_simulation_time(event_time)
            if event_type != JOB_ARRIVAL:
                continue
            job = self.job_map[job_id]

        # job.target_datacenter 应该已经在 __init__ 中固定，每个 episode 使用的是同一种
            arrived_dc_id = job.target_datacenter
            if arrived_dc_id is None:
                raise ValueError(
                    f"任务 {job.job_id} 没有预先分配 target_datacenter，"
                    f"请检查 __init__ 中是否已经完成 job 到边缘数据中心的固定随机分配。"
                )
            arrived_dc_id = str(arrived_dc_id)

            if arrived_dc_id not in self.possible_agents:
                raise ValueError(
                    f"任务 {job.job_id} 到达的数据中心 {arrived_dc_id} "
                    f"不是合法边缘智能体。"
                )

            self.current_job_id = job_id
            self.current_dc_id = arrived_dc_id
            self.current_agent_id = arrived_dc_id
            self.agent_selection = arrived_dc_id
            first_decision_found = True
            break

        # 如果没有找到任何任务到达事件，则直接结束 episode
        if not first_decision_found:
            for agent_id in self.possible_agents:
                self.terminations[agent_id] = True
            self.agent_selection = self.possible_agents[0]

        # global_state = self._get_global_state()
        global_state = np.asarray(self._get_global_state(), dtype=np.float32)
        self._cached_global_state = global_state.copy()

        self.infos = {
            agent_id: {
                "agent_index": self.agent_name_mapping[agent_id],
                "dc_id": agent_id,
                "global_state": self._cached_global_state,
                # "action_mask": self._get_action_mask(agent_id),
            }
            for agent_id in self.possible_agents
        }

        self.has_reset = True

    # 局部观测
    def observe(self, agent: str) -> Dict[str, np.ndarray]:
        # 合法性检查，和_get_local_observation刚开始一样，检查agent合法性和是否reset
        if agent not in self.possible_agents:
            raise ValueError(
                f"非法 agent: {agent}。"
                f"合法智能体应为: {self.possible_agents}。"
            )
        if not self.has_reset:
            raise RuntimeError(
                "环境尚未 reset，不能调用 observe(agent)。"
                "请先调用 reset()。"
            )

        local_observation = self._get_local_observation(agent)
        action_mask = self._get_action_mask(agent)

        # 维度检查
        expected_obs_shape = self.local_observation_spaces[agent].shape
        if local_observation.shape != expected_obs_shape:
            raise ValueError(
                f"observe({agent}) 中 observation 维度错误，"
                f"期望 {expected_obs_shape}，实际 {local_observation.shape}。"
            )
        expected_mask_shape = self.action_mask_spaces[agent].shape
        if action_mask.shape != expected_mask_shape:
            raise ValueError(
                f"observe({agent}) 中 action_mask 维度错误，"
                f"期望 {expected_mask_shape}，实际 {action_mask.shape}。"
            )

        return {
            "observation": local_observation,
            # "action_mask": action_mask,
        }

    #
    def step(self, action: Optional[int]) -> None:

        ############################ 基础状态检查 #################################
        if not self.has_reset:
            raise RuntimeError(
                "环境尚未 reset，不能调用 step()。请先调用 reset()。"
            )

        if self.agent_selection is None:
            raise RuntimeError(
                "agent_selection 为空，当前没有可执行动作的智能体。"
            )

        # pettingzoo有控制智能体死活的功能，必须把智能体注册为act状态才行
        acting_agent = str(self.agent_selection)
        if acting_agent not in self.agents:
            raise RuntimeError(
                f"当前 agent_selection={acting_agent} 不在活跃智能体列表 "
                f"agents={self.agents} 中。"
            )

        # PettingZoo AEC 约定：已经终止的智能体只能执行 step(None)，并交由 _was_dead_step() 从 agents 中依次清理。
        if (
                self.terminations.get(acting_agent, False)
                or self.truncations.get(acting_agent, False)
        ):
            self._was_dead_step(action)
            return

        if action is None:
            raise ValueError(
                f"智能体 {acting_agent} 尚未终止，action 不能为 None。"
            )

        if self.current_job_id is None:
            raise RuntimeError(
                "当前没有等待调度的任务，却调用了 step()。"
            )

        if self.current_agent_id != acting_agent:
            raise RuntimeError(
                "当前决策智能体状态不一致："
                f"agent_selection={acting_agent}，"
                f"current_agent_id={self.current_agent_id}。"
            )

        if self.current_dc_id != acting_agent:
            raise RuntimeError(
                "当前任务所在数据中心与决策智能体不一致："
                f"current_dc_id={self.current_dc_id}，"
                f"acting_agent={acting_agent}。"
            )

        acting_job_id = str(self.current_job_id)
        if acting_job_id not in self.job_map:
            raise KeyError(
                f"当前任务 {acting_job_id} 不存在于 job_map 中。"
            )

        acting_job = self.job_map[acting_job_id]

        # PettingZoo 的标准奖励清空处理
        self._cumulative_rewards[acting_agent] = 0.0
        self._clear_rewards()

        ##################### 处理当前任务动作 ######################################

        action_value = int(
            action
        )

        should_drop = (
            self._should_drop_arrival_job(
                acting_job_id
            )
        )

        # 每次 step 必须产生且只能产生一份
        # Routing Action Fact。
        self.last_routing_action_facts = None

        if should_drop:
            # if action_value != self.drop_action:
            #     raise ValueError(
            #         f"任务 {acting_job_id} 已满足超时丢弃条件，"
            #         f"此时只能传入 DROP_ACTION={self.drop_action}，"
            #         f"实际收到 action={action_value}。"
            #     )

            self._drop_arrival_job(
                job_id=acting_job_id,
                drop_reasion="等待超时",
            )

            self.last_routing_action_facts = (
                self._build_routing_action_facts(
                    job=acting_job,

                    source_dc_id=(
                        acting_agent
                    ),

                    action_type="drop",

                    target_dc_id=None,

                    transfer_latency_s=0.0,

                    transfer_energy_j=0.0,

                    failure_reason=(
                        "waiting_timeout"
                    ),
                )
            )

        else:
            # 解码动作
            decoded_action = self._decode_action(
                agent_id=acting_agent,
                action=action_value,
            )
            action_type = str(decoded_action["action_type"])

            if action_type == "self":
                self.pending_host_job_id = acting_job_id
                self.pending_host_dc_id = acting_agent

                self.last_routing_action_facts = (
                    self._build_routing_action_facts(
                        job=acting_job,

                        source_dc_id=(
                            acting_agent
                        ),

                        action_type="self",

                        target_dc_id=(
                            acting_agent
                        ),
                    )
                )

                # PettingZoo reward 这里只作为接口兼容字段。
                # H-MASAC 不再从 env.rewards 获取训练 Reward。
                self.rewards[
                    acting_agent
                ] = 0.0
                self._clear_current_decision()

                self.agent_selection = None

                self._accumulate_rewards()

                return


            elif action_type in {"edge_dc", "cloud"}:
                target_dc_id = decoded_action["target_dc_id"]
                transfer_energy_before_j = float(
                    acting_job.transfer_energy_j
                )

                arrival_event_time = (
                    self._enqueue_transfer_arrival_event(
                        job=acting_job,

                        source_dc_id=(
                            acting_agent
                        ),

                        target_dc_id=str(
                            target_dc_id
                        ),
                    )
                )

                transfer_latency_s = (
                        float(
                            arrival_event_time
                        )
                        - float(
                    self.current_time
                )
                )

                transfer_energy_j = max(
                    float(
                        acting_job.transfer_energy_j
                    )
                    - transfer_energy_before_j,
                    0.0,
                )

                self.last_routing_action_facts = (
                    self._build_routing_action_facts(
                        job=acting_job,

                        source_dc_id=(
                            acting_agent
                        ),

                        action_type=(
                            action_type
                        ),

                        target_dc_id=str(
                            target_dc_id
                        ),

                        transfer_latency_s=(
                            transfer_latency_s
                        ),

                        transfer_energy_j=(
                            transfer_energy_j
                        ),
                    )
                )

        # 完成动作的收尾
        self.rewards[
            acting_agent
        ] = 0.0
        self._clear_current_decision()






        global_state = np.asarray(self._get_global_state(), dtype=np.float32)
        self._cached_global_state = global_state.copy()

        for agent_id in self.possible_agents:
            # episode 结束后不应再提供普通合法动作，终止智能体 mask 全置零。
            # if (
            #         self.terminations.get(agent_id, False)
            #         or self.truncations.get(agent_id, False)
            # ):
            #     updated_action_mask = np.zeros(
            #         self.action_dim,
            #         dtype=np.int8,
            #     )
            # else:
            #     updated_action_mask = self._get_action_mask(agent_id)

            self.infos[agent_id] = {
                "agent_index": self.agent_name_mapping[agent_id],
                "dc_id": agent_id,
                "global_state": self._cached_global_state,
                # "action_mask": updated_action_mask,
            }

        # 将本次即时奖励累积到 PettingZoo 的 _cumulative_rewards 中。
        self._advance_until_next_routing_decision_or_episode_end()
        self._refresh_pettingzoo_routing_infos()
        self._accumulate_rewards()

    def has_pending_host_decision(self) -> bool:
        return (
                self.pending_host_job_id is not None
                and self.pending_host_dc_id is not None
        )

    def get_pending_host_decision(self) -> Dict[str, str]:
        """
        返回当前等待 Local Host SAC 处理的 Host-level decision。

        Host 层不是 PettingZoo Agent，因此这里不修改：
            possible_agents
            agents
            agent_selection
        """

        if not self.has_pending_host_decision():
            raise RuntimeError(
                "当前不存在等待处理的 Host decision。"
            )

        return {
            "job_id": str(self.pending_host_job_id),
            "dc_id": str(self.pending_host_dc_id),
        }

    def execute_pending_host_action(self, host_action: int,) -> Dict[str, Any]:
        """
        执行 Local Host SAC 给出的 Host action。

        重要：
            本函数不是 PettingZoo step()。
            Host action 永远不能进入 env.step()。
        """

        if not self.has_pending_host_decision():
            raise RuntimeError(
                "当前不存在 pending Host decision，"
                "不能执行 Host action。"
            )

        if self.agent_selection is not None:
            raise RuntimeError(
                "执行 Host action 时 agent_selection 必须为 None。"
                "Host 层不属于 PettingZoo。"
            )

        job_id = str(self.pending_host_job_id)
        dc_id = str(self.pending_host_dc_id)
        host_action = int(host_action)

        if dc_id not in self.dc_map:
            raise KeyError(
                f"Host decision 找不到 DC：{dc_id}"
            )

        target_dc = self.dc_map[dc_id]

        if not (
                0 <= host_action < len(target_dc.host_list)
        ):
            raise ValueError(
                f"非法 Host action："
                f"dc={dc_id}, "
                f"action={host_action}, "
                f"host_count={len(target_dc.host_list)}"
            )

        target_host = target_dc.host_list[host_action]
        host_id = str(target_host.host_id)

        # ==========================================================
        # 在真正执行 Host placement 前先判断：
        #
        # 如果当前直接执行该任务，
        # 是否已经超过 Arrival/Execution Drop Budget。
        #
        # _execute_job_on_host() 内部同样会做这个判断，
        # 这里提前保存只是为了区分：
        #
        #   timeout
        #   resource failure
        #
        # 两种 terminal 原因。
        # ==========================================================

        host_arrival_timeout = (
            self._should_drop_arrival_job(
                job_id
            )
        )

        execution_result = self._execute_job_on_host(
            job_id=job_id,
            dc_id=dc_id,
            host_idx=host_action,
        )

        # ==========================================================
        # Local Host immediate drop 必须产生 Terminal Reward Event。
        #
        # 当前 _execute_job_on_host() 返回 "dropped" 的原因只有：
        #
        #   1. 已超过允许执行时间；
        #   2. 所选 Host 总资源永远无法容纳该 Job。
        #
        # 之前这里只记录 dropped_jobs_info，
        # 没有产生 reward correction，
        # 会导致 Pending Job Trace 永远无法 terminal。
        # ==========================================================

        if execution_result == "dropped":

            if host_arrival_timeout:

                reason = (
                    "local_host_arrival_timeout"
                )

            else:

                reason = (
                    "local_host_resource_failure"
                )

            # ==========================================================
            # Environment 只记录 terminal outcome。
            #
            # Timeout / Resource / Energy penalty
            # 全部由 TrainingRewardModel 计算。
            # ==============================================================

            self._record_job_outcome_event(
                job_id=(
                    job_id
                ),

                reason=(
                    reason
                ),

                terminal=True,
            )







        # Host decision 已消费。
        self.pending_host_job_id = None
        self.pending_host_dc_id = None

        # Host placement 完成以后，
        # 继续推进事件系统，直到：
        #
        #   1. 出现新的 Routing decision；
        #   2. 或 Episode 结束。
        self._advance_until_next_routing_decision_or_episode_end()

        return {
            "job_id": job_id,
            "dc_id": dc_id,
            "host_action": host_action,
            "host_id": host_id,
            "execution_result": str(execution_result),
            "env_time": float(self.current_time),
        }

    ################################### 辅助函数部分 ####################################

    def _advance_until_next_routing_decision_or_episode_end(self,) -> None:

        self.agent_selection = None

        next_decision_found = False

        # 推进事件队列
        next_decision_found = False
        while self.event_queue:
            current_event_time = float(self.event_queue[0][0])
            self._advance_simulation_time(current_event_time)
            pending_arrival_events = []
            while (
                    self.event_queue
                    and self._is_same_event_time(
                        self.event_queue[0][0],
                        current_event_time,
                    )
            ):
                event_time, event_type, event_job_id = heapq.heappop(self.event_queue)

                event_time = float(event_time)
                event_job_id = str(event_job_id)

                if event_type == JOB_FINISH:
                    self._process_job_finish_event(
                        event_job_id
                    )
                    continue

                if event_type == JOB_ARRIVAL:
                    pending_arrival_events.append(
                        (
                            event_time,
                            event_type,
                            event_job_id,
                        )
                    )
                    continue

            arrival_index = 0

            while arrival_index < len(pending_arrival_events):
                while (
                        self.event_queue
                        and self._is_same_event_time(
                            self.event_queue[0][0],
                            current_event_time,
                        )
                ):
                    new_event_time, new_event_type, new_event_job_id = (
                        heapq.heappop(self.event_queue)
                    )

                    new_event_time = float(new_event_time)
                    new_event_job_id = str(new_event_job_id)

                    if new_event_type == JOB_FINISH:
                        self._process_job_finish_event(
                            new_event_job_id
                        )
                        continue

                    if new_event_type == JOB_ARRIVAL:
                        pending_arrival_events.append(
                            (
                                new_event_time,
                                new_event_type,
                                new_event_job_id,
                            )
                        )
                        continue

                (
                    event_time,
                    event_type,
                    event_job_id,
                ) = pending_arrival_events[arrival_index]

                arrival_index += 1

                arrived_job = self.job_map[event_job_id]
                arrived_dc_id = str(
                    arrived_job.target_datacenter
                )

                if arrived_dc_id == self.cloud_id:
                    cloud_arrival_timeout = (
                        self._should_drop_arrival_job(
                            event_job_id
                        )
                    )
                    cloud_result = (
                        self._execute_job_on_host(
                            job_id=event_job_id,
                            dc_id=self.cloud_id,
                            host_idx=0,
                        )
                    )
                    if cloud_result == "dropped":

                        if cloud_arrival_timeout:

                            reason = (
                                "cloud_arrival_timeout"
                            )

                        else:

                            reason = (
                                "cloud_resource_failure"
                            )

                        self._record_job_outcome_event(
                            job_id=(
                                event_job_id
                            ),

                            reason=(
                                reason
                            ),

                            terminal=True,
                        )

                    # Cloud arrival 不需要暂停 AEC 环境，
                    # 继续处理同一时间点剩余事件。
                    continue

                if arrived_dc_id in self.possible_agents:
                    # 该任务需要交给对应边缘智能体做一次调度决策。
                    self.current_job_id = event_job_id
                    self.current_dc_id = arrived_dc_id
                    self.current_agent_id = arrived_dc_id
                    self.agent_selection = arrived_dc_id

                    next_decision_found = True

                    for remaining_event in (
                            pending_arrival_events[arrival_index:]
                    ):
                        heapq.heappush(
                            self.event_queue,
                            remaining_event,
                        )

                    break
            if next_decision_found:
                break

            #
            # event_time, event_type, event_job_id = heapq.heappop(self.event_queue)
            # event_time = float(event_time)
            # event_job_id = str(event_job_id)
            # self.current_time = event_time

            # if event_type == JOB_FINISH:
            #     self._process_job_finish_event(event_job_id)
            #     continue

            # if event_type == JOB_ARRIVAL:
            #     arrived_job = self.job_map[event_job_id]
            #     arrived_dc_id = str(arrived_job.target_datacenter)
            #
            #     # 任务到来事件来自云
            #     if arrived_dc_id == self.cloud_id:
            #         cloud_dc = self.dc_map[self.cloud_id]
            #         self._execute_job_on_host(
            #             job_id=event_job_id,
            #             dc_id=self.cloud_id,
            #             host_idx=0,
            #         )
            #         continue
            #
            #     # 任务到来事件来自边
            #     if arrived_dc_id in self.possible_agents:
            #         self.current_job_id = event_job_id
            #         self.current_dc_id = arrived_dc_id
            #         self.current_agent_id = arrived_dc_id
            #         self.agent_selection = arrived_dc_id
            #         next_decision_found = True
            #         break

        # episode 结束处理
        if not next_decision_found:

            self._drain_remaining_jobs_at_episode_tail()

            # if len(self.event_queue) == 0:
            #     self._drain_remaining_jobs_at_episode_tail()
            if self._check_episode_finished():
                self._terminate_episode()

    def _refresh_pettingzoo_routing_infos(self,) -> None:
        """
        刷新 PettingZoo Routing Agent 的辅助信息。

        Host 层不使用 PettingZoo infos。
        """

        global_state = np.asarray(
            self._get_global_state(),
            dtype=np.float32,
        )

        self._cached_global_state = global_state.copy()

        for agent_id in self.possible_agents:
            self.infos[agent_id] = {
                "agent_index": self.agent_name_mapping[agent_id],
                "dc_id": agent_id,
                "global_state": self._cached_global_state,
            }

    # 不需要掩码了


    def _get_action_mask(self, agent_id: str) -> np.ndarray:
        agent_id = str(agent_id)

        return np.ones(
            self.action_dim,
            dtype=np.int8,
        )

    # 将一个 DataCenter 编码成长度为 self.dc_feat_dim 的特征向量
    def _encode_dc_features(self, dc: DataCenter) -> List[float]:

        # 先更新负载情况
        dc.calculate_dc_loads()

        # 计算总资源容量
        total_cpu = sum(host.cpu_num for host in dc.host_list)
        total_gpu = sum(host.gpu_capacity_num for host in dc.host_list)
        # 计算总负载情况
        running_cpu = sum(
            float(host.used_cpu)
            for host in dc.host_list
        )
        running_gpu = sum(
            float(host.used_gpu)
            for host in dc.host_list
        )
        # 计算资源剩余情况
        available_cpu = max(total_cpu - running_cpu, 0.0)
        available_gpu = max(total_gpu - running_gpu, 0.0)
        # 计算队列情况
        waiting_jobs = sum(len(host.waiting_queue) for host in dc.host_list)
        running_jobs = sum(len(host.running_queue) for host in dc.host_list)
        dc_queue_length_scale = (self.queue_length_scale * max(len(dc.host_list),1,))


        # return [
        #     float(total_cpu),
        #     float(total_gpu),
        #     float(dc.dc_cpu_load),
        #     float(dc.dc_gpu_load),
        #     float(available_cpu),
        #     float(available_gpu),
        #     float(waiting_jobs),
        #     float(running_jobs),
        # ]
        # 归一化return
        return [
            self._normalize(total_cpu, self.max_dc_cpu),
            self._normalize(total_gpu, self.max_dc_gpu),
            float(np.clip(dc.dc_cpu_load, 0.0, 1.0)),
            float(np.clip(dc.dc_gpu_load, 0.0, 1.0)),
            self._normalize(available_cpu, self.max_dc_cpu),
            self._normalize(available_gpu, self.max_dc_gpu),
            # 整个本地 DC 的 Waiting Queue 拥塞程度
            self._saturating_ratio(value=float(waiting_jobs), scale=dc_queue_length_scale,),
            # 整个本地 DC 的 Running Queue 拥塞程度
            self._saturating_ratio(value=float(running_jobs), scale=dc_queue_length_scale,),
        ]

    #  将一个 Host 编码成长度为 self.host_feat_dim 的特征向量
    #  逻辑与_encode_dc_features基本相同
    def _encode_host_features(self, host: Host) -> List[float]:
        host.calculate_load()
        running_cpu = float(host.used_cpu)
        running_gpu = float(host.used_gpu)
        available_cpu = host.get_available_cpu()
        available_gpu = host.get_available_gpu()
        waiting_jobs = len(host.waiting_queue)
        waiting_workload = float(host.waiting_queue.get_total_duration())
        running_jobs = len(host.running_queue)
        waiting_queue_congestion = (self._saturating_ratio(value=float(waiting_jobs), scale=self.queue_length_scale,))
        running_queue_congestion = (self._saturating_ratio(value=float(running_jobs), scale=self.queue_length_scale,))
        waiting_workload_ratio = (self._saturating_ratio(value=waiting_workload, scale=self.queue_workload_scale,))


        # return [
        #     float(host.cpu_num),
        #     float(host.gpu_capacity_num),
        #     float(host.cpu_load),
        #     float(host.gpu_load),
        #     float(available_cpu),
        #     float(available_gpu),
        #     float(waiting_jobs),
        #     float(running_jobs),
        # ]
        return [
            self._normalize(host.cpu_num, self.max_host_cpu),# CPU 总容量
            self._normalize(host.gpu_capacity_num, self.max_host_gpu),
            float(np.clip(host.cpu_load, 0.0, 1.0)),# CPU 当前负载
            float(np.clip(host.gpu_load, 0.0, 1.0)),
            self._normalize(available_cpu, self.max_host_cpu),# CPU 当前可用比例
            self._normalize(available_gpu, self.max_host_gpu),
            waiting_queue_congestion,# Waiting Queue 拥塞程度
            waiting_workload_ratio,# Waiting Queue 总工作量
            running_queue_congestion,# 当前 Running Queue 拥塞程度
        ]

    # 把当前等待调度的任务编码成长度为 self.job_feat_dim 的特征向量
    def _encode_job_features(self, job: Optional[Job]) -> List[float]:
        if job is None:
            return [0.0] * self.job_feat_dim
        # 当前 Job 已经等待了多久
        elapsed_time = max(self.current_time - float(job.arrive_time), 0.0)
        # 当前 Job 最多允许等待多久
        pre_execution_drop_budget = max((self.drop_deadline_ratio - 1.0) * float(job.duration), self.norm_eps,)
        # 当前 Job 已经消耗了多少等待预算
        deadline_consumed_ratio = float(np.clip(elapsed_time / pre_execution_drop_budget, 0.0, 1.0,))



        # return [
        #     float(job.cpu_request),
        #     float(job.gpu_request),
        #     float(job.duration),
        #     float(waiting_time),
        # ]
        # 新的return是归一化结果
        return [
            self._normalize(job.cpu_request, self.max_job_cpu),
            self._normalize(job.gpu_request, self.max_job_gpu),
            self._normalize(job.duration, self.max_job_duration),
            deadline_consumed_ratio,
        ]

    def _encode_local_link_features(self, agent_id: str) -> List[float]:
        # 用于临时保存链路特征
        link_features: List[float] = []

        # 遍历固定的链路观测目标列表
        for target_dc_id in self.local_link_target_dc_ids:
            # 到自身时延是0
            if target_dc_id == agent_id:
                latency = 0.0
            elif self.graph is not None and self.graph.has_edge(agent_id, target_dc_id):
                latency = float(self.graph[agent_id][target_dc_id].get("weight", 0.0))
            else:
                latency = 0.0
            # 将链路时延归一化后加入局部观测
            link_features.append(self._normalize(latency, self.max_latency))

        if len(link_features) != self.link_feat_dim:
            raise ValueError(
                f"局部链路特征维度错误，期望 {self.link_feat_dim}，"
                f"实际 {len(link_features)}。"
            )
        return link_features

    # 将动作编码解码成动作，动作编码必须是int类型
    def _decode_action(
            self,
            agent_id: str,
            action: int,
    ) -> Dict[str, Any]:
        """
        将 Routing action index 解码成实际目标。

        Policy action space 中只包含当前结构上真实存在的动作。

        DROP_ACTION=-1 属于环境生命周期 forced action，
        不属于 Actor 输出空间。
        """

        agent_id = str(
            agent_id
        )

        action = int(
            action
        )

        # ==========================================================
        # Environment-forced Drop
        #
        # 不属于 Routing policy action space。
        # ==========================================================
        if action == self.drop_action:
            return {
                "action_type": "drop",
                "source_dc_id": agent_id,
                "target_dc_id": None,
                "host_idx": None,
            }

        # ==========================================================
        # Structural action range validation
        #
        # Cloud OFF 后旧 Cloud index 会直接在这里失败。
        #
        # 不存在：
        #     mask cloud
        #     silently remap cloud
        # ==============================================================

        if (
                action < 0
                or action >= self.action_dim
        ):
            raise ValueError(
                "非法 Routing action："
                f"action={action}, "
                f"valid_range="
                f"[0, {self.action_dim - 1}], "
                f"cloud_enabled="
                f"{self.enable_cloud_action}"
            )

        if (
                action
                not in self.routing_action_to_dc_id
        ):
            raise RuntimeError(
                "Routing action mapping 不完整："
                f"action={action}, "
                f"mapping="
                f"{self.routing_action_to_dc_id}"
            )

        target_dc_id = str(
            self.routing_action_to_dc_id[
                action
            ]
        )

        # ==========================================================
        # Cloud
        # ==============================================================

        if target_dc_id == self.cloud_id:

            # 理论上 Cloud OFF 时 mapping 中根本不存在 Cloud。
            # 这里是防止配置不变量被破坏。
            if not self.enable_cloud_action:
                raise RuntimeError(
                    "Cloud action 在关闭状态下"
                    "仍然出现在 Routing action mapping 中。"
                )

            return {
                "action_type": "cloud",
                "source_dc_id": agent_id,
                "target_dc_id": self.cloud_id,
                "host_idx": None,
            }

        # ==========================================================
        # Self
        # ==============================================================

        if target_dc_id == agent_id:
            return {
                "action_type": "self",
                "source_dc_id": agent_id,
                "target_dc_id": agent_id,
                "host_idx": None,
            }

        # ==========================================================
        # Edge -> Edge
        # ==============================================================

        return {
            "action_type": "edge_dc",
            "source_dc_id": agent_id,
            "target_dc_id": target_dc_id,
            "host_idx": None,
        }

    # 构造 MASAC centralized critic 使用的全局状态
    def _get_global_state(self) -> np.ndarray:
        state = []

        # 当前任务的特征
        if self.current_job_id is not None and self.current_job_id in self.job_map:
            current_job = self.job_map[self.current_job_id]
        else:
            current_job = None
        state.extend(self._encode_job_features(current_job))

        # 所有数据中心特征
        for dc_id in self.all_dc_ids:
            dc = self.dc_map[dc_id]
            state.extend(self._encode_dc_features(dc))

        # 所有 host 特征
        for dc_id in self.all_dc_ids:
            dc = self.dc_map[dc_id]
            for host_idx in range(self.max_host_num):
                if host_idx < len(dc.host_list):
                    host = dc.host_list[host_idx]
                    state.extend(self._encode_host_features(host))
                else:
                    # 不足 max_host_num 的 host 用 0 padding。
                    state.extend([0.0] * self.host_feat_dim)

        # 数据中心之间的时延矩阵
        for src_dc_id in self.all_dc_ids:
            for dst_dc_id in self.all_dc_ids:
                if src_dc_id == dst_dc_id:
                    latency = 0.0
                elif self.graph is not None and self.graph.has_edge(src_dc_id, dst_dc_id):
                    latency = float(self.graph[src_dc_id][dst_dc_id].get("weight", 0.0))
                else:
                    latency = 0.0

                state.append(self._normalize(latency, self.max_latency))

        state = np.asarray(state, dtype=np.float32)
        if state.shape != self.state_space.shape:
            raise ValueError(
                f"global_state 维度错误，期望 {self.state_space.shape}，"
                f"实际 {state.shape}。"
            )

        return state

    # 初始化 observation/state 归一化所需的尺度参数
    def _init_normalization_stats(self):
        # 避免出现0设定的极小正数
        eps = 1e-8
        self.norm_eps = eps

        # # 统计任务相关特征的最大值，后续可以用这些最大值对 job 特征做 max-scale 归一化
        # self.max_job_cpu = max(
        #     max(float(job.cpu_request) for job in self.base_jobs),
        #     eps,
        # )
        # self.max_job_gpu = max(
        #     max(float(job.gpu_request) for job in self.base_jobs),
        #     eps,
        # )
        # self.max_job_duration = max(
        #     max(float(job.duration) for job in self.base_jobs),
        #     eps,
        # )
        # self.max_arrive_time = max(
        #     max(float(job.arrive_time) for job in self.base_jobs),
        #     eps,
        # )

        job_dataset_df = load_job_dataset(self.job_dataset_path)

        job_cpu_values = np.asarray(job_dataset_df["cpu_request"], dtype=np.float64,)
        job_gpu_values = np.asarray(job_dataset_df["gpu_request"],dtype=np.float64,)
        job_duration_values = np.asarray(job_dataset_df["duration"],dtype=np.float64,)

        self.max_job_cpu = max(float(np.max(job_cpu_values)),eps,)
        self.max_job_gpu = max(float(np.max(job_gpu_values)),eps,)
        self.max_job_duration = max(float(np.max(job_duration_values)),eps,)

        # 统计 host 级别和 datacenter 级别的资源容量最大值
        all_hosts = [
            host
            for dc in self.base_datacenters
            for host in dc.host_list
        ]
        self.max_host_cpu = max(
            max(float(host.cpu_num) for host in all_hosts),
            eps,
        )
        self.max_host_gpu = max(
            max(float(host.gpu_capacity_num) for host in all_hosts),
            eps,
        )
        self.max_dc_cpu = max(
            max(float(sum(host.cpu_num for host in dc.host_list)) for dc in self.base_datacenters),
            eps,
        )
        self.max_dc_gpu = max(
            max(float(sum(host.gpu_capacity_num for host in dc.host_list)) for dc in self.base_datacenters),
            eps,
        )

        # # Waiting Queue Workload 的归一化参考尺度
        # job_durations = np.asarray(
        #     [
        #         float(job.duration)
        #         for job in self.base_jobs
        #     ],
        #     dtype=np.float64,
        # )
        # self.queue_workload_scale = max(
        #     float(
        #         np.percentile(
        #             job_durations,
        #             75.0,       # 这里可能是个坑，留着以后填
        #         )                 # 2026-8-20 1:43，填坑成功
        #     ),
        #     self.norm_eps,
        # )

        self.queue_workload_scale = max(float(np.percentile(job_duration_values, 75.0,)),self.norm_eps,)

        # 统计拓扑图中最大的链路时延
        # latencies = []
        all_latencies = []
        # edge_latencies = []
        # cloud_latencies = []
        #
        # 如果基础拓扑图存在，就遍历图中的所有边。
        if self.base_graph is not None:
            for _, _, data in self.base_graph.edges(data=True):
                latency = float(
                    data.get("weight", 0.0)
                )
                all_latencies.append(latency)

        self.max_latency = max(
            max(all_latencies)
            if all_latencies
            else 1.0,
            eps,
        )
        # self.max_edge_latency = max(
        #     max(edge_latencies)
        #     if edge_latencies
        #     else 1.0,
        #     eps,
        # )
        # self.max_cloud_latency = max(
        #     max(cloud_latencies)
        #     if cloud_latencies
        #     else 1.0,
        #     eps,
        # )

    # 安全除法，避免 scale 为 0
    def _safe_div(self, value: float, scale: float) -> float:
        return float(value) / max(float(scale), self.norm_eps)

    # 执行归一化
    def _normalize(self, value: float, scale: float) -> float:
        return float(np.clip(float(value) / max(float(scale), self.norm_eps), 0.0, 1.0))

    # 等待队列专用归一化
    def _saturating_ratio(self, value: float, scale: float,) -> float:
        value = max(float(value), 0.0,)
        scale = max(float(scale), self.norm_eps,)
        return float(value / (value + scale))

    # 构造单个边缘智能体的局部观测
    def _get_local_observation(self, agent_id: str) -> np.ndarray:

        # 先合法性检查好，主要怕把cloud传进来
        if agent_id not in self.possible_agents:
            raise ValueError(
                f"非法 agent_id: {agent_id}。"
                f"合法智能体应为边缘数据中心: {self.possible_agents}。"
            )

        # 确保环境经过初始化
        if self.graph is None or len(self.dc_map) == 0:
            raise RuntimeError(
                "当前 episode 尚未初始化，无法构造局部观测。"
                "请先调用 reset()。"
            )

        # 确保向量归一化，其实reset里写过归一化，上一条通过了这个一定会过
        if not hasattr(self, "norm_eps") or not hasattr(self, "max_job_cpu"):
            self._init_normalization_stats()

        # 获取当前等待调度的任务
        if self.current_job_id is not None and self.current_job_id in self.job_map:
            current_job = self.job_map[self.current_job_id]
        else:
            current_job = None

        # 获取当前智能体对应的dc
        if agent_id not in self.dc_map:
            raise KeyError(
                f"agent_id={agent_id} 不在当前 episode 的 dc_map 中，"
                f"请检查 reset() 是否正确构建数据中心映射。"
            )
        local_dc = self.dc_map[agent_id]

        # 拼接xiangli
        obs: List[float] = []
        obs.extend(self._encode_job_features(current_job))
        obs.extend(self._encode_dc_features(local_dc))
        # host向量维度要向最大的看齐，不足的用0补全
        for host_idx in range(self.max_host_num):
            if host_idx < len(local_dc.host_list):
                obs.extend(self._encode_host_features(local_dc.host_list[host_idx]))
            else:
                obs.extend([0.0] * self.host_feat_dim)
        obs.extend(self._encode_local_link_features(agent_id))

        obs_array = np.asarray(obs, dtype=np.float32)

        expected_shape = self.local_observation_spaces[agent_id].shape
        if obs_array.shape != expected_shape:
            raise ValueError(
                f"local observation 维度错误，期望 {expected_shape}，"
                f"实际 {obs_array.shape}。"
            )

        return obs_array

####################### 一些时间辅助函数 ###############
    # 返回任务满足 SLA 所允许的最大端到端完成时间
    def _get_sla_completion_limit(self, job: Job) -> float:
        return (self.sla_deadline_ratio  * float(job.duration))

    # 返回任务允许存在于系统中的最大端到端完成时间
    def _get_drop_completion_limit(self, job: Job) -> float:
        return (self.drop_deadline_ratio * float(job.duration))

    # 返回任务从最初到达到当前时刻已经消耗的时间
    def _get_elapsed_service_time(self, job: Job) -> float:
        return max(float(self.current_time) - float(job.arrive_time), 0.0,)

    # 返回预计周转时间
    def _predict_completion_time_if_start_now(self, job: Job, extra_latency: float = 0.0,) -> float:
        elapsed_time = self._get_elapsed_service_time(job)
        return (elapsed_time + max(float(extra_latency), 0.0) + float(job.duration))



    # 任务是否丢弃判断
    def _should_drop_arrival_job(self, job_id: str) -> bool:
        job_id = str(job_id)
        job = self.job_map[job_id]
        predicted_completion_time = (
            self._predict_completion_time_if_start_now(job=job, extra_latency=0.0,))
        drop_limit = (self._get_drop_completion_limit(job))
        return (predicted_completion_time > drop_limit + self.TIME_EPS)

    # 任务丢弃记录
    def _drop_arrival_job(self, job_id: str,drop_reasion: str) -> None:
        job_id = str(job_id)
        job = self.job_map[job_id]
        self.dropped_jobs_info.append(
            {
                "job_id": job_id,
                "drop_time": float(self.current_time),
                "drop_reasion": drop_reasion,
            }
        )

    # 打印丢弃任务信息
    def print_dropped_jobs(self) -> None:
        dropped_num = len(self.dropped_jobs_info)
        print("\n" + "=" * 60)
        print("任务丢弃统计")
        print("=" * 60)
        print(f"当前 episode 被丢弃的任务数量: {dropped_num}")

        if dropped_num == 0:
            print("当前 episode 暂无被丢弃任务。")
            print("=" * 60 + "\n")
            return

        print("\n被丢弃任务列表:")
        for idx, drop_info in enumerate(self.dropped_jobs_info, start=1):
            job_id = drop_info.get("job_id", "UNKNOWN")
            drop_time = drop_info.get("drop_time", None)
            drop_reasion = drop_info.get("drop_reasion", "UNKNOWN")
            print("-" * 60)
            print(f"{idx}. 任务 ID: {job_id}")
            print(f"丢弃原因：{drop_reasion}")
            if drop_time is not None:
                print(f"   丢弃时间 drop_time        : {drop_time:.4f}")

        print("=" * 60 + "\n")

    # 调度到其他地方计算的job，打包成新到达事件
    def _enqueue_transfer_arrival_event(self, job: Job, source_dc_id: str, target_dc_id: str,) -> float:

        job_id = str(job.job_id)
        source_dc_id = str(source_dc_id)
        target_dc_id = str(target_dc_id)

        latency_s = float(
            self.graph[source_dc_id][target_dc_id].get("weight", 0.0)
        )
        # 一次性计算本次 Transmission Energy
        transfer_energy_j = (calculate_transfer_energy_j(latency_s=latency_s,))
        self.transfer_energy_j += float(transfer_energy_j)
        if target_dc_id == self.cloud_id:
            self.edge_cloud_transfer_energy_j += float(transfer_energy_j)
            job.add_edge_cloud_transfer_energy(transfer_energy_j)
        else:
            self.edge_edge_transfer_energy_j += float(transfer_energy_j)
            job.add_edge_edge_transfer_energy(transfer_energy_j)
            job.record_edge_routing_hop(
                latency_s
            )

        arrival_event_time = float(self.current_time) + latency_s
        job.set_target_datacenter(target_dc_id)

        heapq.heappush(
            self.event_queue,
            (
                arrival_event_time,
                JOB_ARRIVAL,
                job_id,
            )
        )
        return arrival_event_time

    # 真正启动一个任务到host上运行
    def _start_job_on_host(self, job_id: str, dc_id: str, host_idx: int,) -> bool:
        job_id = str(job_id)
        dc_id = str(dc_id)
        host_idx = int(host_idx)
        job = self.job_map[job_id]
        target_dc = self.dc_map[dc_id]
        target_host = target_dc.host_list[host_idx]

        started = target_host.add_to_running_queue(job=job, current_time=float(self.current_time),)

        if not started:
            return False

        # actual_waiting_time = job.get_waiting_time()
        # if actual_waiting_time is None:
        #     actual_waiting_time = 0.0
        # actual_waiting_time = max(float(actual_waiting_time), 0.0,)
        # allowed_waiting_time = max(self.drop_deadline_ratio * float(job.duration),self.norm_eps,)
        # waiting_ratio = self._normalize(value=actual_waiting_time, scale=allowed_waiting_time,)
        # if waiting_ratio > 0.0:
        #     self._record_reward_correction(
        #         job_id=job_id,
        #         reward_delta=-float(self.waiting_time_cost_weight * waiting_ratio),
        #         reason="job_start_waiting_cost",
        #     )


        # 添加任务完成事件到队列
        target_dc.calculate_dc_loads()
        self.running_job_location[job_id] = {"dc_id": dc_id, "host_idx": host_idx,}
        finish_time = (float(self.current_time) + float(job.duration))
        heapq.heappush(self.event_queue,(finish_time, JOB_FINISH, job_id,))

        return True

    # 调度动作执行给环境
    def _execute_job_on_host(self, job_id: str, dc_id: str, host_idx: int,) -> str:
        job_id = str(job_id)
        dc_id = str(dc_id)
        host_idx = int(host_idx)

        job = self.job_map[job_id]
        target_dc = self.dc_map[dc_id]
        target_host = target_dc.host_list[host_idx]

        drop_reasion_1 = "等待超时"
        drop_reasion_2 = "资源不足"

        # 任务已经经过了太久的调度，被扔掉了
        if self._should_drop_arrival_job(job_id):
            self._drop_arrival_job(job_id,drop_reasion_1)
            # self.running_job_location.pop(job_id, None)
            return "dropped"

        job.set_target_datacenter(dc_id)

        # host总资源不够
        if not target_host.can_ever_accommodate(job):
            self._drop_arrival_job(job_id, drop_reasion_2,)
            return "dropped"

        # 等待队列有人物，新来的也等待
        if not target_host.waiting_queue.is_empty():
            target_host.add_to_waiting_queue(job)
            self.queued_jobs += 1
            self.max_waiting_queue_length = max(
                self.max_waiting_queue_length,
                len(target_host.waiting_queue),
            )
            return "queued"

        # 可接受
        if target_host.can_accommodate(job):
            started = self._start_job_on_host(
                job_id=job_id,
                dc_id=dc_id,
                host_idx=host_idx,
            )
            if started:
                return "started"
            target_host.add_to_waiting_queue(job)
            self.queued_jobs += 1
            self.max_waiting_queue_length = max(
                self.max_waiting_queue_length,
                len(target_host.waiting_queue),
            )

            return "queued"

        target_host.add_to_waiting_queue(job)
        self.queued_jobs += 1
        self.max_waiting_queue_length = max(
            self.max_waiting_queue_length,
            len(target_host.waiting_queue),
        )
        return "queued"
        # started = self._start_job_on_host(
        #     job_id=job_id,
        #     dc_id=dc_id,
        #     host_idx=host_idx,
        # )
        # 因资源不足被丢弃
        # if not started:
        #     self._drop_arrival_job(job_id,drop_reasion_2)
        #     # self.running_job_location.pop(job_id, None)
        #     return  False

        # 成功卸载后更新dc负载
        # target_dc.calculate_dc_loads()
        # self.running_job_location[job_id] = {
        #     "dc_id": dc_id,
        #     "host_idx": host_idx,
        # }
        #
        # # 创建任务完成事件
        # finish_time = (float(self.current_time)+ float(job.duration))
        # heapq.heappush(self.event_queue,(finish_time,JOB_FINISH,job_id,))
        # return True

    # Host 释放资源后，严格按照 FCFS 尝试启动 waiting_queue 中的任务
    def _drain_host_waiting_queue(self, dc_id: str, host_idx: int,) -> None:
        dc_id = str(dc_id)
        host_idx = int(host_idx)

        target_dc = self.dc_map[dc_id]
        target_host = target_dc.host_list[host_idx]

        # 得用while，因为可能一次资源释放能满足多个等待队列中的任务同时上
        while not target_host.waiting_queue.is_empty():
            waiting_job = target_host.waiting_queue._queue[0]
            waiting_job_id = str(waiting_job.job_id)

            if self._should_drop_arrival_job(waiting_job_id):
                dropped_job = target_host.remove_from_waiting_queue()
                self._drop_arrival_job(
                    job_id=str(dropped_job.job_id),
                    drop_reasion="等待超时",
                )

                self._record_job_outcome_event(
                    job_id=str(
                        dropped_job.job_id
                    ),

                    reason=(
                        "waiting_timeout"
                    ),

                    terminal=True,
                )
                continue

            if not target_host.can_accommodate(waiting_job):
                break

            started = self._start_job_on_host(
                job_id=waiting_job_id,
                dc_id=dc_id,
                host_idx=host_idx,
            )

            if not started:
                break

            removed_job = target_host.remove_from_waiting_queue()
            self.started_from_waiting_jobs += 1

    # 处理任务完成事件
    def _process_job_finish_event(self, job_id: str) -> Job:
        # 定位任务与执行位置
        job_id = str(job_id)
        location = self.running_job_location[job_id]
        dc_id = str(location["dc_id"])
        host_idx = int(location["host_idx"])
        target_dc = self.dc_map[dc_id]
        target_host = target_dc.host_list[host_idx]
        finished_job = target_host.remove_from_running_queue(job_id)

        # 放入完成队列
        target_host.add_to_completed_queue(
            job=finished_job,
            current_time=float(self.current_time),
        )

        if dc_id == self.cloud_id:
            attributable_compute_energy_j = (calculate_cloud_task_attributable_compute_energy_j(job=finished_job,))

        else:
            attributable_compute_energy_j = (calculate_edge_task_attributable_compute_energy_j(job=finished_job,host=target_host,))

        finished_job.set_compute_energy(attributable_compute_energy_j)

        self._record_job_outcome_event(
            job_id=(
                job_id
            ),

            reason=(
                "completed"
            ),

            terminal=True,
        )


        # 更新负载
        target_dc.calculate_dc_loads()

        # 删除记录
        self.running_job_location.pop(job_id)

        self._drain_host_waiting_queue(
            dc_id=dc_id,
            host_idx=host_idx,
        )

        return finished_job

    # 修复训练一半报错的bug
    def _drain_remaining_jobs_at_episode_tail(self) -> None:
        # 只有在当前已经没有 Actor 决策时才能进行 Episode 尾部收尾。
        if self.current_job_id is not None:
            return

        if self.current_agent_id is not None:
            return

        if self.current_dc_id is not None:
            return

        while True:

            ####################################################################
            # 第一步：
            # 如果某个 Host 当前没有正在运行的任务，
            # 但是 waiting_queue 中还有任务，
            # 主动按照已有 FCFS 规则尝试启动这些任务。
            ####################################################################
            for dc in self.datacenters:
                for host_idx, host in enumerate(dc.host_list):

                    if host.waiting_queue.is_empty():
                        continue

                    # 正在运行任务的 Host 不需要主动 drain。
                    #
                    # 正常情况下，该运行任务对应的 JOB_FINISH 事件
                    # 应该已经存在于 event_queue，
                    # 等它完成后 _process_job_finish_event()
                    # 会自动调用 _drain_host_waiting_queue()。
                    if not host.running_queue.is_empty():
                        continue

                    self._drain_host_waiting_queue(
                        dc_id=str(dc.dc_id),
                        host_idx=int(host_idx),
                    )

            ####################################################################
            # 第二步：
            # 上面的 drain 可能启动等待任务，
            # _start_job_on_host() 会为它们创建新的 JOB_FINISH 事件。
            #
            # Episode 尾部已经不存在新的调度决策，因此这里继续消费这些
            # JOB_FINISH 事件即可。
            ####################################################################
            if self.event_queue:

                event_time, event_type, event_job_id = heapq.heappop(
                    self.event_queue
                )

                event_time = float(event_time)
                event_job_id = str(event_job_id)

                self._advance_simulation_time(event_time)

                # Episode 尾部理论上只应该剩下 JOB_FINISH。
                #
                # 如果出现 JOB_ARRIVAL，说明调用这个收尾函数的时机有问题，
                # 不能静默吞掉事件，否则会遗漏 Actor 决策。
                if event_type != JOB_FINISH:
                    raise RuntimeError(
                        "Episode 尾部收尾阶段发现非 JOB_FINISH 事件："
                        f"time={event_time}, "
                        f"type={event_type}, "
                        f"job_id={event_job_id}"
                    )

                self._process_job_finish_event(
                    event_job_id
                )

                # _process_job_finish_event() 可能：
                # 1. 启动新的 waiting job；
                # 2. 创建新的 JOB_FINISH；
                # 因此重新进入循环。
                continue

            ####################################################################
            # 第三步：
            # event_queue 已空，检查是否还有任务残留。
            ####################################################################
            remaining_running_jobs = 0
            remaining_waiting_jobs = 0

            blocked_waiting_details = []

            for dc in self.datacenters:
                for host_idx, host in enumerate(dc.host_list):

                    running_count = len(host.running_queue)
                    waiting_count = len(host.waiting_queue)

                    remaining_running_jobs += running_count
                    remaining_waiting_jobs += waiting_count

                    if waiting_count > 0:
                        head_job = host.waiting_queue._queue[0]

                        blocked_waiting_details.append(
                            {
                                "dc_id": str(dc.dc_id),
                                "host_idx": int(host_idx),
                                "host_id": str(host.host_id),
                                "waiting_count": int(waiting_count),
                                "running_count": int(running_count),
                                "head_job_id": str(head_job.job_id),
                                "used_cpu": float(host.used_cpu),
                                "cpu_capacity": float(host.cpu_num),
                                "used_gpu": float(host.used_gpu),
                                "gpu_capacity": float(
                                    host.gpu_capacity_num
                                ),
                            }
                        )

            ####################################################################
            # 没有任何 running / waiting job：
            # Episode 中的后台任务已经全部处理完，可以正常结束。
            ####################################################################
            if (
                    remaining_running_jobs == 0
                    and remaining_waiting_jobs == 0
            ):
                self._update_compute_energy_to(float(self.current_time))
                return

            ####################################################################
            # event_queue 已空却还有 running job：
            # 说明 running job 的 JOB_FINISH 事件丢失。
            #
            # 这属于严重环境不变量错误，不能直接把 Episode 标为完成。
            ####################################################################
            if remaining_running_jobs > 0:
                raise RuntimeError(
                    "环境进入非法状态：event_queue 已空，"
                    "但仍存在 running job。"
                    f"running_jobs={remaining_running_jobs}, "
                    f"waiting_jobs={remaining_waiting_jobs}"
                )

            ####################################################################
            # 到这里意味着：
            #
            # event_queue == 空
            # running_queue == 空
            # waiting_queue != 空
            #
            # 对于一个空闲 Host 来说，等待队首如果满足 can_ever_accommodate，
            # _drain_host_waiting_queue() 理应能够启动它。
            #
            # 因此如果还残留 waiting job，说明存在其他环境逻辑错误，
            # 不允许继续静默运行。
            ####################################################################
            raise RuntimeError(
                "环境无法继续推进："
                "event_queue、running_queue 均为空，"
                "但 waiting_queue 仍有任务。"
                f"waiting_jobs={remaining_waiting_jobs}, "
                f"details={blocked_waiting_details}"
            )

    def _record_job_outcome_event(
            self,
            *,
            job_id: str,
            reason: str,
            terminal: bool = True,
    ) -> None:
        """
        记录已经真实发生的 Job outcome facts。

        Environment 在这里不计算任何 RL Reward。
        """

        job_id = str(
            job_id
        )

        job = self.job_map[
            job_id
        ]

        completion_time_s = None

        if (
                job.finish_time
                is not None
        ):
            completion_time_s = (
                job.get_turnaround_time()
            )

            if completion_time_s is not None:
                completion_time_s = max(
                    float(
                        completion_time_s
                    ),
                    0.0,
                )

        waiting_time_s = None

        if job.start_time is not None:
            waiting_time_s = max(
                float(
                    job.start_time
                )
                - float(
                    job.arrive_time
                ),
                0.0,
            )

        execution_time_s = None

        if job.start_time is not None:
            execution_end_time = (
                float(
                    job.finish_time
                )
                if job.finish_time is not None
                else float(
                    self.current_time
                )
            )

            execution_time_s = max(
                execution_end_time
                - float(
                    job.start_time
                ),
                0.0,
            )

        self.pending_job_outcome_events.append(
            {
                "job_id":
                    job_id,

                "reason":
                    str(
                        reason
                    ),

                "terminal":
                    bool(
                        terminal
                    ),

                "env_time":
                    float(
                        self.current_time
                    ),

                "job_duration_s":
                    float(
                        job.duration
                    ),

                "elapsed_service_time_s":
                    float(
                        self._get_elapsed_service_time(
                            job
                        )
                    ),

                "completion_time_s":
                    completion_time_s,

                "waiting_time_s":
                    waiting_time_s,

                "execution_time_s":
                    execution_time_s,

                "compute_energy_j":
                    float(
                        job.compute_energy_j
                    ),

                "transfer_energy_j":
                    float(
                        job.transfer_energy_j
                    ),

                "edge_edge_transfer_energy_j":
                    float(
                        job
                            .edge_edge_transfer_energy_j
                    ),

                "edge_cloud_transfer_energy_j":
                    float(
                        job
                            .edge_cloud_transfer_energy_j
                    ),

                "total_attributable_energy_j":
                    float(
                        job
                            .get_total_attributable_energy()
                    ),
            }
        )

    def pop_job_outcome_events(
            self,
    ) -> List[Dict[str, Any]]:
        """
        取走从上一次 Trainer 消费以后产生的所有
        Environment Job Outcome Facts。
        """

        events = list(
            self.pending_job_outcome_events
        )

        self.pending_job_outcome_events.clear()

        return events



    def _build_routing_action_facts(
            self,
            *,
            job: Job,
            source_dc_id: str,
            action_type: str,
            target_dc_id: Optional[str],
            transfer_latency_s: float = 0.0,
            transfer_energy_j: float = 0.0,
            failure_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        构造一次 Routing action 已经真实发生后的物理事实。

        本函数不计算 Reward。
        """

        return {
            "job_id":
                str(
                    job.job_id
                ),

            "source_dc_id":
                str(
                    source_dc_id
                ),

            "target_dc_id":
                (
                    None
                    if target_dc_id is None
                    else str(
                        target_dc_id
                    )
                ),

            "action_type":
                str(
                    action_type
                ),

            "env_time":
                float(
                    self.current_time
                ),

            "job_duration_s":
                float(
                    job.duration
                ),

            "elapsed_service_time_s":
                float(
                    self._get_elapsed_service_time(
                        job
                    )
                ),

            "transfer_latency_s":
                max(
                    float(
                        transfer_latency_s
                    ),
                    0.0,
                ),

            "transfer_energy_j":
                max(
                    float(
                        transfer_energy_j
                    ),
                    0.0,
                ),

            "total_attributable_energy_j":
                float(
                    job
                        .get_total_attributable_energy()
                ),

            "failure_reason":
                (
                    None
                    if failure_reason is None
                    else str(
                        failure_reason
                    )
                ),
        }

    def pop_last_routing_action_facts(
            self,
    ) -> Dict[str, Any]:
        """
        返回并清空最近一次 Routing action 的事实快照。
        """

        if self.last_routing_action_facts is None:
            raise RuntimeError(
                "当前没有可供 Trainer 消费的 "
                "Routing action facts。"
            )

        facts = dict(
            self.last_routing_action_facts
        )

        self.last_routing_action_facts = None

        return facts

    # 清除调度决策执行时的临时变量
    def _clear_current_decision(self) -> None:
        self.current_job_id = None
        self.current_dc_id = None
        self.current_agent_id = None



    # 检查好一个 episode 是否结束
    def _check_episode_finished(self) -> bool:

        # 当前还有任务等待边缘智能体决策
        if self.current_job_id is not None:
            return False
        if self.current_agent_id is not None:
            return False
        if self.current_dc_id is not None:
            return False
        if self.pending_host_job_id is not None:
            return False
        if self.pending_host_dc_id is not None:
            return False

        # 事件队列中仍有任务到达事件或任务完成事件
        if len(self.event_queue) > 0:
            return False

        for dc in self.datacenters:
            for host in dc.host_list:
                if not host.running_queue.is_empty():
                    return False
                if not host.waiting_queue.is_empty():
                    return False

        return True

    # 将当前 episode 标记为自然结束
    def _terminate_episode(self) -> None:
        for agent_id in self.possible_agents:
            self.terminations[agent_id] = True
        self.current_job_id = None
        self.current_dc_id = None
        self.current_agent_id = None
        self.pending_host_job_id = None
        self.pending_host_dc_id = None
        # Host direct path 可能在 agent_selection=None 的情况下
        # 触发 Episode terminal。
        #
        # PettingZoo AEC 后续仍需要一个 dead agent 作为
        # agent_selection，以便通过 step(None) 清理 agents。
        if (
                self.agents
                and self.agent_selection is None
        ):
            self.agent_selection = self.agents[0]

    # 判断两个浮点事件时间是否属于同一个仿真时刻
    @classmethod
    def _is_same_event_time(cls, time_a: float, time_b: float,) -> bool:
        return math.isclose(
            float(time_a),
            float(time_b),
            rel_tol=0.0,
            abs_tol=cls.TIME_EPS,
        )

