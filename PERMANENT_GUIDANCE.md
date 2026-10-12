# 长期保留启发式引导：实现与验证

本次按长期保留方案修改 BCGH-MASAC 与 MR-BCGH-MASAC。默认训练与评估的 λ 均为 0.3，不随回合下降。既有三阶段参数接口仍可用于消融；正式比较应把三个系数设为同一非零值。没有改动启发式评分公式、奖励、折扣、Host 网络、环境或 H-MASAC 对照算法。

## 实际优化的策略

对决策时刻的局部观测 o、有效引导偏置 d=λb、静态可行性掩码 m，定义：

```
c = concat(d, not m)
z = Actor(o, agent identity, c)
μ = masked_softmax(z + detach(d), m)
Q = Critic(global_state, agent identity, c)
```

MR 每个 DC 使用独立网络，DC 身份由网络选择提供。下一状态 bootstrap 使用同任务下一次真实路由决策所属 DC 的 Actor、目标 Critic 和温度。

Actor 和双 Critic 的第一隐藏层新增 `GuidanceConditioner`，以线性分支编码 c，并与原状态分支相加。新增权重从零开始，不消耗原网络初始化的随机数序列。无引导时 c 全零，分支没有 bias，因此即使训练后，关闭引导时也不会附加隐藏常数。网络可学习增强、削弱或反向修正启发式偏置；这属于联合策略优化的正常行为。

这次采用紧凑上下文：有效 λb 和静态禁用标记。λ 已包含在偏置中；置信度已参与上游偏置计算，并另外写入决策审计日志，未作为独立神经网络输入或 Replay 列。它明确描述当前策略的直接干预，不能替代所有历史信念，也不声称使部分可观测环境变成完全 Markov 环境。

执行、Actor 损失、下一状态软价值和温度熵均使用相同复合分布。偏置及上下文输入 detach，Actor 和上下文编码权重正常接收梯度。温度的目标熵按同一掩码下的允许动作数计算。终止样本不运行下一状态网络。

## 回放和兼容性

采样前得到引导快照，Collector 在进入环境执行前复制数组。Pending Trace 保留该次决策的事实；同任务真实下一次决策的快照成为 next context，任务最终完成时不从实时信念重新计算旧偏置。Finalize 再次独立复制当前及下一次决策的偏置和掩码，避免定稿 Transition 与旧 Pending 数组共用可变存储。

随机预热动作仍允许来自均匀行为策略，但必须保存当时目标策略的快照。引导启用且 λ=0 的消融路径显式保存零偏置、全允许掩码，并保留原采样行为。

训练入口开启严格回放模式：缺失当前快照、非终止样本缺失 successor 快照、形状或版本不兼容均报错。一个 Replay 池不允许混合有引导和无引导记录，禁止通过补零掩盖历史数据缺失。关闭引导的独立对照运行仍可使用完整的无引导 Replay。

Routing 权重、MR manifest、训练元数据、Transition、ReplayBatch、回合/DC 日志和决策审计新增 `guidance_context_version=1`；原有日志统计 schema 仍为 5。旧 Routing 权重不自动迁移，应复用同一已验收 Host bundle，重新训练 Routing。回放原流程不跨运行持久化；恢复模型后重新积累当前版本的经验。

## 诊断与运行

原有本地动作概率变化、argmax 改变次数、原始/执行熵继续记录。新增以下回合和 DC 汇总字段，也写入抽样决策 JSONL：

| 字段 | 含义 |
| --- | --- |
| `guidance_mean_logit_span` | 条件 Actor 的最大最小 logit 差 |
| `guidance_mean_effective_bias_span` | 实际 λb 的最大最小差 |
| `guidance_mean_bias_logit_span_ratio` | 每次决策偏置差与 logit 差之比的均值，分母至少 1e-8 |
| `guidance_mean_masked_action_count` | 每次策略决策的静态禁用动作数均值 |
| `guidance_mean_snapshot_complete` | 被审计策略决策的快照完整比例；正常为 1 |
| `guidance_mean_outcome_confidence` | 候选终局质量置信度均值；手工 bias 无对应证据时为空 |

`base_probabilities` 表示**同一上下文、同一掩码**下，去掉显式 logit 偏置后的概率；它衡量加法偏置的直接影响。训练过的上下文分支仍影响这些 logits，故此项不能等同于完全关闭引导的消融。随机预热不计入这些策略决策汇总，快照完整性由 Replay 的严格校验保障。

训练入口不变，应使用新的输出路径和相同的已验收 Host bundle：

```powershell
python -B -m schedulers.BCGH-MASAC.train_bcgh_masac
python -B -m schedulers.MR-BCGH-MASAC.train_mr_bcgh_masac
```

`evaluate_guidance.py` 要求带 context-v1 标记的新检查点。评估期间不执行优化器更新，每个配置/任务种子使用独立环境及新建的因果信念统计。full 使用训练配置的 λ；no_guidance 去掉整个引导上下文与加法偏置。后者是评估时消融；若要比较算法学习能力，应额外从头训练相应对照。

## 验证范围

专门回归覆盖 Actor/Critic 上下文梯度、启发式 detach、上下文分支学习后执行/回放概率一致、当前和 successor Critic 上下文对应、缺失或混合快照拒绝、λ=0 等价采样、旧检查点在修改网络前拒载，以及中性分支不改变初始化随机数。

相关测试还覆盖跨 DC bootstrap、终止样本、拥塞评分、冻结 Host 的短训练、保存恢复及评估。短训练只验证数据和优化链路；没有执行完整训练，不能据此宣称完成率、SLA 或收敛速度改善。性能比较需要相同 Host、负载和训练预算，以及多个训练种子和独立评估种子。

2026-10-12 最终验证：下列两组分别 **92 passed**、**7 passed**，合计 99 项通过。Windows 沙箱默认临时目录不可写，验证使用已授权的正常临时目录权限。

```powershell
python -B -m pytest tests/test_permanent_guidance.py tests/test_congestion_guidance_v5.py tests/test_mr_bcgh_masac.py tests/test_bcgh_short_window.py tests/test_guidance_scoring.py -q
python -B -m pytest tests/test_shared_host.py -q
```
