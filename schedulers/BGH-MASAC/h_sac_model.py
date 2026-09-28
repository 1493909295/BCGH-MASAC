from __future__ import annotations
from typing import Tuple
import torch
from torch import nn
from torch.nn import functional as F
import config as conf


DEFAULT_EPS = 1e-8

__all__ = (
    # Routing MASAC
    "RoutingDiscreteActor",
    "RoutingDiscreteQNetwork",
    "RoutingTwinDiscreteCritic",

    # Local Host SAC
    "LocalHostDiscreteActor",
    "LocalHostQNetwork",
    "LocalHostTwinCritic",

    # Common network update helpers
    "hard_update",
    "soft_update",
)

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
class RoutingDiscreteActor(nn.Module):
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
            hidden_dim: int = conf.ROUTING_ACTOR_HIDDEN_DIM,
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
        initialize_linear_layer(self.output_layer, gain=conf.ROUTING_ACTOR_GAIN,)

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
            eps: float = DEFAULT_EPS,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        # Actor 原始偏好
        logits = self.forward(
            local_obs=local_obs,
            agent_indices=agent_indices,
        )

        # ==========================================================
        # 所有结构动作均参与策略分布。
        #
        # 不再：
        #     masked_fill()
        #     multiply action_mask
        #     masked renormalization
        # ==========================================================
        action_probs = F.softmax(
            logits,
            dim=-1,
        )

        action_log_probs = torch.log(
            action_probs.clamp_min(
                float(eps)
            )
        )

        return (
            action_probs,
            action_log_probs,
            logits,
        )

    # 按照当前策略概率随机采样动作
    def sample_action(
            self,
            local_obs: torch.Tensor,
            agent_indices: torch.Tensor,

    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:

        # 获取完整策略分布。
        action_probs, action_log_probs, _ = (
            self.get_policy(
                local_obs=local_obs,
                agent_indices=agent_indices,
            )
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
class RoutingDiscreteQNetwork(nn.Module):
    """
    Routing MASAC 的 centralized discrete Q network。

    输入：
        global_state
            CTDE 训练阶段的 Routing centralized state

        agent_indices
            当前正在做 Routing decision 的 Edge DC identity

    网络内部：
        global_state
            +
        agent one-hot

    输出：
        当前 Routing Agent 对全部 Routing actions 的 Q values。

    注意：
        1. 这是 Routing 层 Critic；
        2. 不属于 Local Host SAC；
        3. 不读取 Host Observation；
        4. 不与任何 LocalHostQNetwork 共享参数。
    """

    def __init__(
            self,
            global_state_dim: int,
            action_dim: int,
            num_agents: int,
            hidden_dim: int = conf.ROUTING_CRITIC_HIDDEN_DIM,
    ) -> None:
        super().__init__()

        self.global_state_dim = int(
            global_state_dim
        )

        self.action_dim = int(
            action_dim
        )

        self.num_agents = int(
            num_agents
        )

        self.hidden_dim = int(
            hidden_dim
        )

        if self.global_state_dim <= 0:
            raise ValueError(
                "Routing global_state_dim 必须 > 0"
            )

        if self.action_dim <= 0:
            raise ValueError(
                "Routing action_dim 必须 > 0"
            )

        if self.num_agents <= 0:
            raise ValueError(
                "Routing num_agents 必须 > 0"
            )

        # ==========================================================
        # CTDE Critic Input
        #
        # Centralized Routing State
        #       +
        # 当前 Routing Agent identity
        # ==========================================================

        critic_input_dim = (
            self.global_state_dim
            + self.num_agents
        )

        self.fc1 = nn.Linear(
            critic_input_dim,
            self.hidden_dim,
        )

        self.fc2 = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
        )

        self.output_layer = nn.Linear(
            self.hidden_dim,
            self.action_dim,
        )

        initialize_linear_layer(
            self.fc1,
            gain=nn.init.calculate_gain(
                "relu"
            ),
        )

        initialize_linear_layer(
            self.fc2,
            gain=nn.init.calculate_gain(
                "relu"
            ),
        )

        initialize_linear_layer(
            self.output_layer,
            gain=conf.ROUTING_CRITIC_GAIN,
        )

    def forward(
            self,
            global_state: torch.Tensor,
            agent_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        返回当前 Routing Agent 对全部 Routing actions
        的 centralized Q values。
        """

        agent_one_hot = build_agent_one_hot(
            agent_indices=(
                agent_indices
            ),

            num_agents=(
                self.num_agents
            ),
        )

        agent_one_hot = agent_one_hot.to(
            device=global_state.device,
            dtype=global_state.dtype,
        )

        critic_input = torch.cat(
            [
                global_state,
                agent_one_hot,
            ],
            dim=-1,
        )

        hidden = F.relu(
            self.fc1(
                critic_input
            )
        )

        hidden = F.relu(
            self.fc2(
                hidden
            )
        )

        return self.output_layer(
            hidden
        )

# 双Q Critic
class RoutingTwinDiscreteCritic(nn.Module):
    """
    Routing MASAC 的 centralized Twin-Q Critic。

    两个 Q 网络：

        Q1(global_state, agent_id)
        Q2(global_state, agent_id)

    完全服务于 Routing MASAC + CTDE。

    不与任何 Local Host SAC Critic 共享：
        - 参数
        - Optimizer
        - Target Critic
        - ReplayBuffer
    """

    def __init__(
            self,
            global_state_dim: int,
            action_dim: int,
            num_agents: int,
            hidden_dim: int = conf.ROUTING_CRITIC_HIDDEN_DIM,
    ) -> None:
        super().__init__()

        self.q1 = RoutingDiscreteQNetwork(
            global_state_dim=(
                global_state_dim
            ),

            action_dim=(
                action_dim
            ),

            num_agents=(
                num_agents
            ),

            hidden_dim=(
                hidden_dim
            ),
        )

        self.q2 = RoutingDiscreteQNetwork(
            global_state_dim=(
                global_state_dim
            ),

            action_dim=(
                action_dim
            ),

            num_agents=(
                num_agents
            ),

            hidden_dim=(
                hidden_dim
            ),
        )

    def forward(
            self,
            global_state: torch.Tensor,
            agent_indices: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        同时返回 Q1 / Q2 对全部 Routing actions 的估计。
        """

        q1_values = self.q1(
            global_state=(
                global_state
            ),

            agent_indices=(
                agent_indices
            ),
        )

        q2_values = self.q2(
            global_state=(
                global_state
            ),

            agent_indices=(
                agent_indices
            ),
        )

        return (
            q1_values,
            q2_values,
        )

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





class LocalHostDiscreteActor(nn.Module):
    """
    单个 Edge DC 的 Local Host SAC Actor。

    输入：
        当前 Job + 当前 Local DC + 当前 DC 全部真实 Host

    输出：
        当前 DC 内各 Host 的离散动作概率。

    注意：
        1. 不属于 PettingZoo；
        2. 不使用 agent one-hot；
        3. 不使用 global state；
        4. 不使用 action mask；
        5. 不使用 Host padding。
    """

    def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            hidden_dim: int = conf.HOST_ACTOR_HIDDEN_DIM,
    ) -> None:
        super().__init__()

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)

        self.fc1 = nn.Linear(
            self.obs_dim,
            self.hidden_dim,
        )

        self.fc2 = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
        )

        self.output_layer = nn.Linear(
            self.hidden_dim,
            self.action_dim,
        )

        initialize_linear_layer(
            self.fc1,
            gain=nn.init.calculate_gain("relu"),
        )

        initialize_linear_layer(
            self.fc2,
            gain=nn.init.calculate_gain("relu"),
        )

        initialize_linear_layer(
            self.output_layer,
            gain=conf.HOST_ACTOR_GAIN,
        )

    def forward(
            self,
            obs: torch.Tensor,
    ) -> torch.Tensor:

        hidden = F.relu(
            self.fc1(obs)
        )

        hidden = F.relu(
            self.fc2(hidden)
        )

        return self.output_layer(hidden)

    def get_policy(
            self,
            obs: torch.Tensor,
            eps: float = DEFAULT_EPS,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:

        logits = self.forward(obs)

        probabilities = F.softmax(
            logits,
            dim=-1,
        )

        log_probabilities = torch.log(
            probabilities.clamp_min(
                float(eps)
            )
        )

        return (
            probabilities,
            log_probabilities,
            logits,
        )

class LocalHostQNetwork(nn.Module):
    """
    Local Host SAC Critic。

    输入仅为当前 DC 的 Host Observation；
    输出当前 DC 所有 Host action 的 Q value。
    """

    def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            hidden_dim: int = conf.HOST_CRITIC_HIDDEN_DIM,
    ) -> None:
        super().__init__()

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)

        self.hidden_dim = int(
            hidden_dim
        )
        self.fc1 = nn.Linear(
            self.obs_dim,
            hidden_dim,
        )

        self.fc2 = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        self.output_layer = nn.Linear(
            hidden_dim,
            self.action_dim,
        )

        initialize_linear_layer(
            self.fc1,
            gain=nn.init.calculate_gain("relu"),
        )

        initialize_linear_layer(
            self.fc2,
            gain=nn.init.calculate_gain("relu"),
        )

        initialize_linear_layer(
            self.output_layer,
            gain=conf.Q_NET_GAIN,
        )

    def forward(
            self,
            obs: torch.Tensor,
    ) -> torch.Tensor:

        hidden = F.relu(
            self.fc1(obs)
        )

        hidden = F.relu(
            self.fc2(hidden)
        )

        return self.output_layer(hidden)

class LocalHostTwinCritic(nn.Module):
    """
    Local Host SAC 使用独立 Twin Q。

    不与 Routing MASAC Critic 共享任何参数。
    """

    def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            hidden_dim: int = conf.HOST_CRITIC_HIDDEN_DIM,
    ) -> None:
        super().__init__()

        self.q1 = LocalHostQNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        )

        self.q2 = LocalHostQNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        )

    def forward(
            self,
            obs: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        return (
            self.q1(obs),
            self.q2(obs),
        )
