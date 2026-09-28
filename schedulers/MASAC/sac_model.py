from __future__ import annotations
from typing import Optional, Tuple
import torch
from torch import nn
from torch.nn import functional as F
import config as conf


DEFAULT_EPS = 1e-8

# 把智能体整数编号转换成 one-hot 向量
def build_agent_one_hot(agent_indices: torch.Tensor, num_agents: int) -> torch.Tensor:

    num_agents = int(num_agents)
    agent_indices = agent_indices.to(dtype=torch.long)

    one_hot = F.one_hot(agent_indices, num_classes=num_agents,)
    one_hot = one_hot.to(dtype=torch.float32)
    return one_hot

# 初始化一个全连接层
def initialize_linear_layer(layer: nn.Linear, gain: float = 1.0) -> None:

    # 使用正交矩阵初始化权重
    nn.init.orthogonal_(layer.weight, gain=float(gain))

    # 如果该层存在偏置，则把偏置全部初始化为 0
    if layer.bias is not None:
        nn.init.constant_(
            layer.bias,
            0.0,
        )

# 参数共享的离散Actor网络
class MaskedDiscreteActor(nn.Module):
    """
       所有边缘智能体共用同一个 Actor。
       为了让共享 Actor 知道当前是哪一个智能体在决策，
       网络输入中除了 local_obs，还会拼接 agent one-hot。

       网络输入 shape：
           local_obs      -> (batch_size, local_obs_dim)
           agent_indices  -> (batch_size,)

       网络输出 shape：
           logits         -> (batch_size, action_dim)       网络原始偏好分数
           probabilities  -> (batch_size, action_dim)       动作概率
       """

    def __init__(
            self,
            local_obs_dim: int,
            action_dim: int,
            num_agents: int,
            hidden_dim: int = conf.ACTOR_HIDDEN_DIM,
    ) -> None:
        super().__init__()

        local_obs_dim = int(local_obs_dim)
        action_dim = int(action_dim)
        num_agents = int(num_agents)
        hidden_dim = int(hidden_dim)

        self.local_obs_dim = local_obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim

        # actor输入由局部观测 + 智能体one-hot拼接而成
        actor_input_dim = local_obs_dim + num_agents

        self.fc1 = nn.Linear(actor_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, action_dim)

        initialize_linear_layer(self.fc1, gain=nn.init.calculate_gain("relu"))
        initialize_linear_layer(self.fc2, gain=nn.init.calculate_gain("relu"))
        initialize_linear_layer(self.output_layer, gain=conf.ACTOR_GAIN)

    # 根据局部观测和智能体编号计算原始 logits
    # 在共享参数的actor中，每次actor收到的应该是局部观察+智能体编号的拼接
    def forward(self, local_obs: torch.Tensor, agent_indices: torch.Tensor) -> torch.Tensor:

        # 把智能体整数编号转换成 one-hot。
        agent_one_hot = build_agent_one_hot(agent_indices=agent_indices, num_agents=self.num_agents)

        # 确保 one-hot 与局部观测位于同一设备。
        agent_one_hot = agent_one_hot.to(device=local_obs.device)

        # 确保 one-hot 与局部观测数据类型一致。
        agent_one_hot = agent_one_hot.to(dtype=local_obs.dtype)

        # 在最后一维拼接局部观测和智能体 one-hot。
        actor_input = torch.cat([local_obs, agent_one_hot],dim=-1)

        hidden = F.relu(self.fc1(actor_input))
        hidden = F.relu(self.fc2(hidden))

        logits = self.output_layer(hidden)

        return logits

    # 计算应用动作掩码后的动作概率和对数概率
    def get_policy(
            self,
            local_obs: torch.Tensor,
            agent_indices: torch.Tensor,
            action_mask: torch.Tensor,
            eps: float = DEFAULT_EPS,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # 先通过 Actor 网络计算原始 logits
        logits = self.forward(local_obs=local_obs, agent_indices=agent_indices)

        # 把动作掩码统一转换为布尔 Tensor
        valid_action_mask = action_mask.to(dtype=torch.bool)

        # 统计每条样本有多少个合法动作。
        valid_action_count = valid_action_mask.sum(dim=-1)

        # 获取当前浮点类型能表示的最小有限值。
        very_negative = torch.finfo(logits.dtype).min

        # 合法动作保留原始 logits，非法动作替换成极小值。
        masked_logits = logits.masked_fill(~valid_action_mask, very_negative,)

        # 对应用掩码后的 logits 做 softmax，得到动作概率。
        action_probs = F.softmax(masked_logits,dim=-1,)

        # 再乘一次掩码，确保非法动作概率严格为 0。
        action_probs = action_probs * valid_action_mask.to(dtype=action_probs.dtype)

        # 计算每行概率之和。
        probability_sum = action_probs.sum(dim=-1,keepdim=True,)

        # 重新归一化，防止浮点误差导致概率和略微偏离 1。
        action_probs = action_probs / probability_sum.clamp_min(float(eps))

        # 对概率取对数。
        action_log_probs = torch.log(action_probs.clamp_min(float(eps)))

        # 非法动作位置的 log probability 设为 0。
        # 后续计算 sum(prob * log_prob) 时，这些位置贡献为 0。
        action_log_probs = torch.where(valid_action_mask, action_log_probs, torch.zeros_like(action_log_probs),)

        # 返回概率、对数概率和掩码后的 logits。
        return (
            action_probs,
            action_log_probs,
            masked_logits,
        )

    # 按照当前策略概率随机采样动作
    def sample_action(
            self,
            local_obs: torch.Tensor,
            agent_indices: torch.Tensor,
            action_mask: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:

        # 获取完整策略分布。
        action_probs, action_log_probs, _ = self.get_policy(
            local_obs=local_obs,
            agent_indices=agent_indices,
            action_mask=action_mask,
        )

        # 使用离散分类分布包装动作概率。
        distribution = torch.distributions.Categorical(probs=action_probs)

        # 为 batch 中每条样本随机采样一个动作。
        sampled_actions = distribution.sample()

        # 从完整 log probability 中取出被选动作的 log probability。
        selected_log_probs = action_log_probs.gather(
            dim=1,
            index=sampled_actions.unsqueeze(1),
        ).squeeze(1)

        # 返回采样结果和完整策略信息。
        return (
            sampled_actions,
            selected_log_probs,
            action_probs,
            action_log_probs,
        )

# 离散Q网络
class DiscreteQNetwork(nn.Module):
    def __init__(
            self,
            global_state_dim: int,
            action_dim: int,
            num_agents: int,
            hidden_dim: int = conf.Q_NET_HIDDEN_DIM,
    ) -> None:

        super().__init__()
        global_state_dim = int(global_state_dim)
        action_dim = int(action_dim)
        num_agents = int(num_agents)
        hidden_dim = int(hidden_dim)

        # 保存网络结构参数。
        self.global_state_dim = global_state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim

        critic_input_dim = global_state_dim + num_agents
        self.fc1 = nn.Linear(critic_input_dim, hidden_dim,)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim,)
        self.output_layer = nn.Linear(hidden_dim,action_dim,)

        # 初始化各层参数。
        initialize_linear_layer(self.fc1, gain=nn.init.calculate_gain("relu"),)
        initialize_linear_layer(self.fc2, gain=nn.init.calculate_gain("relu"),)
        initialize_linear_layer(self.output_layer, gain=conf.Q_NET_GAIN,)

    # 计算当前智能体所有动作的 Q 值
    def forward(
            self,
            global_state: torch.Tensor,
            agent_indices: torch.Tensor,
    ) -> torch.Tensor:

        # 智能体编号one-hot编码
        agent_one_hot = build_agent_one_hot(agent_indices=agent_indices, num_agents=self.num_agents,)

        # 把 one-hot 移动到全局状态所在设备。
        agent_one_hot = agent_one_hot.to(device=global_state.device,)

        # 统一数据类型。
        agent_one_hot = agent_one_hot.to(dtype=global_state.dtype,)

        # 拼接全局状态和当前智能体身份信息。
        critic_input = torch.cat([global_state, agent_one_hot],dim=-1,)

        # 第一层特征提取。
        hidden = F.relu(self.fc1(critic_input))

        # 第二层特征提取。
        hidden = F.relu(self.fc2(hidden))

        q_values = self.output_layer(hidden)
        return q_values

# 双Q Critic
class TwinDiscreteCritic(nn.Module):
    def __init__(
        self,
        global_state_dim: int,
        action_dim: int,
        num_agents: int,
        hidden_dim: int = 256,
    ) -> None:

        super().__init__()

        self.q1 = DiscreteQNetwork(
            global_state_dim=global_state_dim,
            action_dim=action_dim,
            num_agents=num_agents,
            hidden_dim=hidden_dim,
        )

        self.q2 = DiscreteQNetwork(
            global_state_dim=global_state_dim,
            action_dim=action_dim,
            num_agents=num_agents,
            hidden_dim=hidden_dim,
        )

# 同时返回Q1 Q2对全部动作的估计
    def forward(
        self,
        global_state: torch.Tensor,
        agent_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        q1_values = self.q1(global_state=global_state, agent_indices=agent_indices,)
        q2_values = self.q2(global_state=global_state, agent_indices=agent_indices,)

        return q1_values, q2_values

# 把 source_network 的全部参数完整复制给 target_network，一般在算法初始化时调用一次
def hard_update(target_network: nn.Module, source_network: nn.Module) -> None:
    # state_dict 中包含网络全部可训练参数和缓冲区
    target_network.load_state_dict(source_network.state_dict())

def soft_update(
    target_network: nn.Module,
    source_network: nn.Module,
    tau: float,
) -> None:

    tau = float(tau)
    with torch.no_grad():
        # 同时遍历目标网络和源网络的对应参数。
        for target_parameter, source_parameter in zip(
                target_network.parameters(),
                source_network.parameters(),
        ):
            # 原地执行加权平均。
            target_parameter.mul_(1.0 - tau)
            target_parameter.add_(
                source_parameter,
                alpha=tau,
            )


