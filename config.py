import os

# 数据集路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JOB_DATASET_PATH = os.path.join(BASE_DIR, "./dataset/DC_dataset/new_small_job_info_df.csv")
HOST_DATASET_PATH = os.path.join(BASE_DIR, "./dataset/DC_dataset/node_info_df.csv")


# 环境基本参数
NUM_DATACENTERS = 5          # 数据中心数量
NUM_HOST = 100               # 全局生成的主机 (Host) 总数量
NUM_JOBS = 2000              # 全局生成的任务 (Job) 总数量
LAMBDA_RATE = 0.35           # LAMBDA_RATE 越大，任务到达越密集（时间间隔越短）

# DC 初始任务到达模式。只修改这一个参数即可切换：
#   "uniform"       ：所有 Edge DC 等概率接收初始任务（原有行为）。
#   "heterogeneous" ：按下方角色和权重形成异构到达率。
# LAMBDA_RATE 始终表示全局总到达率；切换模式不会改变系统任务总量。
DC_ARRIVAL_MODE = "heterogeneous"

# 5 个 Edge DC 的异构角色：2 个忙碌、1 个正常、2 个清闲。
# 角色仅影响任务的初始到达位置，不包含 Cloud，也不改变可选路由动作。
DC_ARRIVAL_ROLE_BY_ID = {
    "DC-1": "busy",
    "DC-2": "busy",
    "DC-3": "normal",
    "DC-4": "idle",
    "DC-5": "idle",
}

# 权重会自动归一化为单节点分配概率。当前权重下：
#   busy = 30%，normal = 20%，idle = 10%。
# 单节点到达率 = LAMBDA_RATE × 分配概率，由环境运行时自动计算。
DC_ARRIVAL_ROLE_WEIGHTS = {
    "busy": 1.5,
    "normal": 1.0,
    "idle": 0.5,
}

CLOUD_LATENCY_RANGE = (10, 20) # 边缘节点到云数据中心的时延范围
EDGE_LATENCY_RANGE = (2,5)    # 边缘节点之间的时延范围
# DROP_DEADLINE_RATE = 2.0      # 丢弃任务超时倍数
SLA_DEADLINE_RATIO = 2.0
DROP_DEADLINE_RATIO = 2.5
QUEUE_LENGTH_SCALE = 10.0      # 队列长度归一化使用参数（根据训练后最大等待队列的一半）
ENV_KEEP_PATH = os.path.join(BASE_DIR, "./environment/env_keep")

SIMULATION_TIME_UNIT = "s"     # 系统时间基本单位
ENERGY_UNIT = "J"              # 系统能量基本单位
EDGE_CPU_IDLE_POWER_W = 110.0       # edge host CPU 空闲功率
EDGE_CPU_FULL_POWER_W = 170.0       # edge host CPU 满载功率
EDGE_GPU_IDLE_POWER_W = 25.0        # edge host GPU 空闲功率
EDGE_GPU_FULL_POWER_W = 250.0       # edge host GPU 满载功率
# EDGE_EDGE_TRANSFER_ENERGY_RATIO = 0.04      # e2e传输能耗系数
# EDGE_CLOUD_TRANSFER_ENERGY_RATIO = 0.10     # e2c传输能耗系数

# 以下参数来源诡异，有待验证
ENERGY_NORMALIZATION_PERCENTILE = 95.0      # 能耗归一化参数
CLOUD_CPU_POWER_PER_UNIT_W = 0.4711         # cloud单位CPU功率
CLOUD_GPU_POWER_PER_UNIT_W = 226.9522       # cloud单位GPU功率
TRANSFER_SEND_FIXED_ENERGY_J = 866.1977     # 发送能耗
TRANSFER_RECEIVE_FIXED_ENERGY_J = 866.1977  # 接收能耗
TRANSFER_LATENCY_ENERGY_COEFFICIENT_W = 628.4384    # 传输功率


# 模型基本参数
ACTOR_HIDDEN_DIM = 256      # actor-net 隐藏层维度
Q_NET_HIDDEN_DIM = 256      # Q-net 隐藏层维度
ACTOR_GAIN = 0.01           # Actor 输出层初始化幅度
Q_NET_GAIN = 0.01           # Critic 输出层初始化幅度
GAMMA = 0.99                # 长期奖励折扣
TUA = 0.005                 # Target Critic 软更新速度
ACTOR_LR = 1e-4             # Actor 学习率
CRITIC_LR = 2e-4            # Critic 学习率
ALPHA_LR = 1e-4             # 温度参数 α 的学习率
INITIAL_ALPHA = 0.1         # 初始探索强度
TARGET_ENTROPY_RATIO = 0.2  # 随机性保留
MAX_GRAD_NORM = 3.0         # 梯度裁剪阈值
Policy_Updata_Interval = 2  # Actor 相对 Critic 的更新频率
Target_Update_Interval = 1  # Target Critic 更新频率

ROUTING_ACTOR_HIDDEN_DIM = 256
ROUTING_CRITIC_HIDDEN_DIM = 256
ROUTING_ACTOR_GAIN = 0.01
ROUTING_CRITIC_GAIN = 0.01
ROUTING_GAMMA = 0.99
ROUTING_TAU = 0.005
ROUTING_ACTOR_LR = 1e-4
ROUTING_CRITIC_LR = 2e-4
ROUTING_ALPHA_LR = 1e-4
ROUTING_INITIAL_ALPHA = 0.1
ROUTING_TARGET_ENTROPY_RATIO = 0.2
ROUTING_MAX_GRAD_NORM = 3.0
ROUTING_POLICY_UPDATE_INTERVAL = 2
ROUTING_TARGET_UPDATE_INTERVAL = 1

HOST_ACTOR_HIDDEN_DIM = 256
HOST_CRITIC_HIDDEN_DIM = 256
HOST_ACTOR_GAIN = 0.01
HOST_CRITIC_GAIN = 0.01
HOST_GAMMA = 0.99
HOST_TAU = 0.005
HOST_ACTOR_LR = 1e-4
HOST_CRITIC_LR = 2e-4
HOST_ALPHA_LR = 1e-4
HOST_INITIAL_ALPHA = 0.1
HOST_TARGET_ENTROPY_RATIO = 0.2
HOST_MAX_GRAD_NORM = 3.0
HOST_POLICY_UPDATE_INTERVAL = 2
HOST_TARGET_UPDATE_INTERVAL = 1
DEVICE = "cuda:0"





# 训练基本参数
Episodes = 1000        # 训练轮次
ReplyBuffer_Capacity = 200000           # 经验池容量
Batch_Size = 512            # 采样批量
Seed = 42           # 宇宙的终极答案
Random_warmup_step = 5000           # 前期随机预热启动
Learning_Starts = 5000          # 至少积累多少个普通动作后再开始更新网络
Train_Every = 4                 # 收集多少经验训一次网络
Updates_Per_Train = 2            # 每条经验最多执行多少次网络更新

ROUTING_REPLAY_CAPACITY = 200000
ROUTING_BATCH_SIZE = 512
ROUTING_RANDOM_WARMUP_STEPS = 5000
ROUTING_LEARNING_STARTS = 5000
ROUTING_TRAIN_EVERY = 4
ROUTING_UPDATES_PER_TRAIN = 2

HOST_REPLAY_CAPACITY = 200000
HOST_BATCH_SIZE = 512
HOST_RANDOM_WARMUP_STEPS = 5000
HOST_LEARNING_STARTS = 5000
HOST_TRAIN_EVERY = 4
HOST_UPDATES_PER_TRAIN = 2

Log_interval = 1            # 每多少episode打印一次统计信息
Checkpoint_Interval = 100        # 仅单层 MASAC 使用；H/BGH-MASAC 只在联合优化开始和训练结束时保存
Checkpoint_Dir = "model/H-MASAC/checkpoints"
Log_csv_Path = "result/H-MASAC/train_log.csv"
H_MASAC_EPISODE_LOG_CSV_PATH = "result/H-MASAC/episode_log.csv"
H_MASAC_DC_LOG_CSV_PATH = "result/H-MASAC/dc_log.csv"
BGH_MASAC_CHECKPOINT_DIR = ("model/BGH-MASAC/checkpoints")
BGH_MASAC_EPISODE_LOG_CSV_PATH = ("result/BGH-MASAC/episode_log.csv")
BGH_MASAC_DC_LOG_CSV_PATH = ("result/BGH-MASAC/dc_log.csv")

# OPT shares H-MASAC's SAC/reward/three-stage settings, with full DC-load observations.
OPT_CHECKPOINT_DIR = "model/OPT/checkpoints"
OPT_EPISODE_LOG_CSV_PATH = "result/OPT/episode_log.csv"
OPT_DC_LOG_CSV_PATH = "result/OPT/dc_log.csv"
OPT_RESUME_CHECKPOINT = None
# Optional two-layer checkpoint: import only its compatible Host weights.
OPT_HOST_INIT_CHECKPOINT = None


Old_Env_Path = None         #可选的旧环境文件路径,为 None 时，CloudEdgeEnv 会按自己的默认逻辑生成新环境
Resume_Checkpoint = None         #可选的断点模型路径,为 None 表示从头训练。
BGH_MASAC_RESUME_CHECKPOINT = None  # BGH-MASAC 只能从自己的 checkpoint 恢复。None 表示从头训练。
Vary_Episode_Seed: bool = True          # 是否在每个 episode 使用不同但可复现的 seed

HOST_PRETRAIN_EPISODES = 200    # host训练轮数
ROUTING_TRAIN_EPISODES = 500    # routing训练轮数
JOINT_FINETUNE_EPISODES = (Episodes - HOST_PRETRAIN_EPISODES - ROUTING_TRAIN_EPISODES)      #合并训练



# 奖励参数（先这样写着吧，目前没什么好办法
TASK_COMPLETION_REWARD = 2.5            # 一个任务真正完成获得

COMPLETION_TIME_COST_WEIGHT = 1.0       # 实际任务完成时间成本权重
SLA_VIOLATION_COST_WEIGHT = 1.5         # SLA 违约严重程度权重
QUEUE_ADMISSION_COST_WEIGHT = 0.5       # 排队风险参数，帮助模型认识到进入等待队列和立即执行是不同的

REMOTE_OFFLOAD_BASE_PENALTY = 0.05        # 远程调度成本
REMOTE_LATENCY_COST_WEIGHT = 1.5          # 远程时延成本权重
SLA_RISK_COST_WEIGHT = 1.0              # 违约风险参数
EDGE_DEADLINE_RISK_COST_WEIGHT = 1.0    # Edge 转发行为对任务剩余 deadline 的侵蚀
TIMEOUT_DROP_PENALTY = 4.0              # 超时惩罚
RESOURCE_DROP_PENALTY = 2.0             # 资源不足惩罚，其实已经不会触发了，因为后面搞了掩码
# COMPLETION_CREDIT_DECAY = 0.8           # 调度链奖励衰减参数
# FAILURE_CREDIT_DECAY = 0.8              # 调度链惩罚衰减参数

ENERGY_NORMALIZATION_J = 170000.0       # 将真实物理能耗 J 缩放到适合 Reward 学习的数值范围
ENERGY_COST_WEIGHT = 0.30               #  控制 Energy objective 在联合 Reward 中的相对权重


# cloud 开关，false为关闭云
ENABLE_CLOUD_ACTION = False

# Neighbor Historical Feedback：COLLECT 控制是否更新历史统计，
# USE 控制是否把历史反馈特征拼接进 Routing Actor observation。
COLLECT_NEIGHBOR_HISTORICAL_FEEDBACK = True
USE_NEIGHBOR_HISTORICAL_FEEDBACK = True
NEIGHBOR_FEEDBACK_EWMA_ALPHA = 0.10     # 参数越大越重视最近成果，控制新结果对历史的影响性
NEIGHBOR_FEEDBACK_AGE_SCALE_SAMPLES = 100.0 # 参数越大历史保留越久，判断历史多久没更新了
NEIGHBOR_FEEDBACK_CONFIDENCE_SCALE_SAMPLES = 20.0   # 判断历史可信度

BGH_ENABLE_BAYESIAN_GAME = True        # 启用贝叶斯后验与历史拥塞博弈
BGH_ENABLE_HEURISTIC_GUIDANCE = True   # 启用贝叶斯-拥塞启发式 Actor 引导

# 第 2 步：统一 Bayesian Congestion Belief 语义。
# alpha 表示拥塞证据，beta 表示非拥塞证据；当前使用无信息先验 Beta(1, 1)。
# 这些参数已接入终止任务 Evidence、拥塞博弈和 Actor 引导链路。
BGH_BAYESIAN_PRIOR_ALPHA = 1.0        # alpha_congested 的初始值
BGH_BAYESIAN_PRIOR_BETA = 1.0         # beta_non_congested 的初始值
BGH_BAYESIAN_CONFIDENCE_SCALE = 20.0  # 有效历史样本量映射到置信度的尺度

# 短时间窗口与资源竞争成本：分别按 CPU/GPU 已转发需求除以静态容量，
# 对当前任务计算 f(x+d)-f(x)，其中 f(x)=linear*x+quadratic*x*x。
BGH_SHORT_WINDOW_S = 1200.0         # 仿真秒；约两倍数据集任务时长中位数，可按实验调整
BGH_SOURCE_EVIDENCE_WEIGHT = 2.0    # 来源弱证据的最大伪样本量（非真实拥塞标签）
BGH_SOURCE_VOLUME_SCALE = 1.0       # 归一化来源资源量的饱和尺度
BGH_SOURCE_CONFIDENCE_SCALE = 5.0   # 来源独立接收事件的置信度尺度
BGH_RESOURCE_CPU_WEIGHT = 0.5
BGH_RESOURCE_GPU_WEIGHT = 0.5
BGH_RESOURCE_COST_SCALE = 1.0       # 原始增量成本以 c/(c+scale) 归一化
BGH_MAX_LOGIT_BIAS = 0.3           # 最终单动作偏置绝对值上限
BGH_PRESSURE_LINEAR_WEIGHT = 1.0     # 拥塞成本的一阶项权重
BGH_PRESSURE_QUADRATIC_WEIGHT = 1.0  # 拥塞成本的二阶项权重

# 目标 DC 短窗口吸收能力：区分本地接纳、继续 Edge 转发、转 Cloud 和丢弃。
# 仅作用于其他 Edge DC 候选；无样本时 confidence=0，先验不会产生惩罚。
BGH_ABSORPTION_ENABLED = True
BGH_ABSORPTION_PRIOR_SELF = 1.0
BGH_ABSORPTION_PRIOR_EDGE = 1.0
BGH_ABSORPTION_PRIOR_CLOUD = 1.0
BGH_ABSORPTION_PRIOR_DROP = 1.0
BGH_ABSORPTION_CONFIDENCE_SCALE = 8.0
BGH_ABSORPTION_EDGE_WEIGHT = 0.7
BGH_ABSORPTION_CLOUD_WEIGHT = 0.5
BGH_ABSORPTION_DROP_WEIGHT = 1.0
# 单任务需求相对于目标 DC 最大主机容量的主导占比，划分 low/medium/high。
BGH_ABSORPTION_DEMAND_THRESHOLDS = (0.25, 0.60)
BGH_ABSORPTION_CLASS_CONFIDENCE_SCALE = 4.0
# random/forced/orchestrator 路由仍进入行为日志，但不校准策略吸收信念。
BGH_ABSORPTION_POLICY_ACTIONS_ONLY = True

# 第 5 步：Benefit / Risk / Utility / Bias 参数。
BGH_BENEFIT_SUCCESS_WEIGHT = 0.4     # 历史成功质量权重
BGH_BENEFIT_SLA_WEIGHT = 0.4         # SLA 满足质量权重
BGH_BENEFIT_DELAY_WEIGHT = 0.2       # 历史延迟质量权重
BGH_RISK_CONGESTION_COST_WEIGHT = 1.0  # C_j 在 Risk 中的权重
BGH_GUIDANCE_SCALE = 0.3             # Utility 转换为 logit bias 的尺度

# 第 8 步：三阶段 Guidance λ 调度系数。
# 实际 Bias = BGH_GUIDANCE_SCALE × λ × centered_utility_bias。
BGH_GUIDANCE_LAMBDA_STAGE2_START = 0.3  # Routing Train 起始引导强度
BGH_GUIDANCE_LAMBDA_STAGE2_PEAK = 1.0   # Routing Train 结束/Joint 起始
BGH_GUIDANCE_LAMBDA_STAGE3_END = 0.3    # Joint Finetune 结束及评估引导强度
