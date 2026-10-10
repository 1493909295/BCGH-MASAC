# MR-BCGH-MASAC 使用说明

MR-BCGH-MASAC 在 BCGH-MASAC 的双层调度和贝叶斯拥塞引导基础上，为每个边缘 DC 创建完整、独立的 routing SAC。网络架构与超参数可以相同，Actor、双 Critic、目标双 Critic、温度参数、优化器及训练经验均不共享。

## 运行

在项目根目录执行：

```powershell
python -m schedulers.MR-BCGH-MASAC.train_mr_bcgh_masac
```

也支持直接执行 `schedulers/MR-BCGH-MASAC/train_mr_bcgh_masac.py`。沿用项目的 GPU 训练要求和 CUDA 运算预检。公共实验参数仍由 `config.py` 提供。

新调度器保持九个 Python 文件：`__init__.py`、`train_mr_bcgh_masac.py`、`h_masac_agent.py`、`experience.py`、`observations.py`、`guided_policy.py`、`bayesian_game.py`、`bayesian_congestion_game.py`、`training_support.py`。内部使用自身包的相对导入，不依赖其他调度器的实现。

## 网络与训练语义

`RoutingMASAC` 是无可训练参数的管理器，`agents[dc_id]` 是对应 DC 的 `LocalRoutingMASAC`。

- 每个 Actor 仅输入当前 DC 的局部 routing 观测。
- 每个 Critic 输入集中式全局状态，保持 CTDE。
- 网络入口不再拼接 DC one-hot；经验中的 DC ID/index 用于选择网络和校验归属。
- 每个 DC 使用不同的 routing 初始化种子、经验采样种子和预热动作随机流。
- 各 DC 的 α、Actor/Critic 优化器、更新计数及目标网络软更新分别独立。
- 云端继续作为可选路由目的地，不创建 routing 学习器。动作索引、Self/Edge/Cloud 及环境强制 Drop 的语义沿用原环境。
- Host 层、奖励计算、同一任务的因果链和短窗口贝叶斯机制沿用 BCGH-MASAC。

对当前决策主体为 DC_i、同一任务后继决策主体为 DC_j 的非终止经验，DC_i 的 Critic 目标为：

\[
y_i = r_i + \gamma \sum_a \pi_j(a\mid o'_j)
\left[\min(Q'_{j,1}(s',a),Q'_{j,2}(s',a))-\alpha_j\log\pi_j(a\mid o'_j)\right].
\]

管理器按后继 DC 分组计算价值，使用该 DC 的 Actor、目标双 Critic 和 α，整个过程不产生梯度。更新 DC_i 时只修改其自身网络与优化器。终止经验的目标直接为奖励，不访问后继 DC，也不将 `-1` 终止标识替换成其他 DC。

经验只在任务终止并完成轨迹整理后写入，按当前决策的来源 DC 分发。Forced 和 Host 预训练阶段的 orchestrator 决策不进入 routing 经验池。跨 DC 的后继信息仍完整保留。

## 配置与三阶段调度

新配置项：

| 配置 | 含义 |
|---|---|
| `MR_BCGH_MASAC_CHECKPOINT_DIR` | 专属模型目录，默认 `model/MR-BCGH-MASAC/checkpoints` |
| `MR_BCGH_MASAC_EPISODE_LOG_CSV_PATH` | 系统级日志路径 |
| `MR_BCGH_MASAC_DC_LOG_CSV_PATH` | DC 级日志路径 |
| `MR_BCGH_MASAC_RESUME_CHECKPOINT` | 恢复入口；`None` 表示从头训练 |
| `MR_BCGH_ROUTING_REPLAY_CAPACITY_PER_DC` | 每 DC routing 经验池容量，默认沿用 `ROUTING_REPLAY_CAPACITY` |

`TrainConfig.routing_replay_capacity` 也表示每 DC 容量。总经验容量等于边缘 DC 数量乘以该值；每个池都保存全局状态，因此内存占用需要按总容量估算。

Routing 的网络和优化参数沿用 `ROUTING_*`，引导参数沿用 `BCGH_*`。`ROUTING_RANDOM_WARMUP_STEPS`、`ROUTING_LEARNING_STARTS` 和 `ROUTING_TRAIN_EVERY` 均按每 DC 的普通动作步数判断。Actor 与目标网络的更新间隔按该 DC 的 Critic 更新次数判断。

管理器在事件循环边界和回合结束时检查所有 DC。若经验在更新间隔之后才完成写入，满足条件的 DC 仍能得到一次更新块；同一动作计数不会重复触发。多个已过去的间隔合并为一次更新块，不在任务集中终止时补做所有历史间隔。

三个阶段保持原有顺序：

1. Host 预训练：正常任务采用 Self routing，只更新 Host。
2. Routing 训练：冻结 Host 参数，各 DC 独立更新 routing。
3. 联合微调：按各层条件更新 routing 和 Host。

执行动作时仍使用当前 DC 的 `logits + λ × bias` 和原有引导可行性规则。引导只影响行为策略，SAC 更新继续使用基础 Actor 分布。

关闭贝叶斯和启发式引导后，运行模式为 `Independent-routing-no-guidance`。这不再表示与共享 routing 网络的 H-MASAC 完全等价。启发式引导仍要求启用贝叶斯博弈。

## 日志

系统级日志中的 `routing_update_step` 是各 DC 更新次数之和，`routing_alpha` 是各 DC α 的等权均值，损失和熵按本回合有效更新记录取平均。对应聚合口径也写入 CSV。

DC 级日志记录 routing 经验池大小与容量、累计普通动作步数、更新次数、当前 α、本回合损失与熵、随机预热完成状态和学习就绪状态。就绪状态表示已满足样本量与 learning-starts 条件；实际更新还受当前训练阶段和更新间隔控制。

系统级日志另有按 DC 保存的 α 和动作步数 JSON，便于识别忙碌与清闲 DC 的学习差异。特性开关记录本次 `TrainConfig` 的实际值。

## 检查点与恢复

沿用在联合微调开始及训练结束时保存的时机。以 `final.pt` 为例：

- `final.pt`：routing 管理器清单、DC 顺序、动作映射、配置及各 DC 训练计数。
- `final_routing/dc_0.pt` 等：按清单映射到 DC ID 的各套独立 routing 网络和优化器。
- `final_hosts/<dc_id>.pt`：各 DC 的 Host 模型。
- `final.trainer.json`：双层结构、实验配置及训练状态。

将 `MR_BCGH_MASAC_RESUME_CHECKPOINT` 设置为该清单的路径即可恢复。必须保留相邻 routing、Host 目录及 trainer JSON。

恢复先检查结构和配置，再将所有模型加载到临时学习器中；整套模型及计数校验成功后才替换运行中的状态。缺失或损坏最后一个 DC 文件也不会留下部分恢复的网络。

原 BCGH-MASAC 的共享 routing 检查点不能直接恢复到此调度器。DC 顺序、动作映射、云开关、维度或 routing 配置不一致时会拒绝恢复。

与原训练入口一样，检查点不保存 replay 内容和随机数生成器的运行状态。恢复后经验池重新积累，模型、优化器和各 DC 计数继续使用保存值，因此不承诺逐步复现未中断的训练。

## 评估与验证

`training_support.evaluate_episode(env, routing_agent, host_agents, settings, seed=42, deterministic=True)` 支持完整引导、仅贝叶斯和无引导模式，根据当前 DC 选择专属 Actor，评估不更新参数。

行为测试位于 `tests/test_mr_bcgh_masac.py`：

```powershell
python -m pytest tests/test_mr_bcgh_masac.py -q
```

测试覆盖参数隔离、后继 DC 的目标 Q 与 α、终止经验、经验归属、延迟写入后的训练触发、引导网络选择、检查点完整性、分 DC 指标及三阶段训练。完整训练测试需要 CUDA 和项目中的小规模历史环境夹具；不满足条件时会明确跳过该项。

对照实验应同时记录总经验容量、总更新次数、参数量、训练耗时及各 DC 样本量。网络独立性带来的性能变化需通过正式实验判断。
