# BCGH / MR-BCGH 对称终局评分

> 本文保留 v4 的设计说明。当前代码已升级为 v5：拥塞成本默认启用、首后继拆分 started/queued、训练使用因果引导快照。现行公式、默认参数与日志定义以 [CONGESTION_GUIDANCE_V5.md](CONGESTION_GUIDANCE_V5.md) 为准。

BCGH-MASAC 和 MR-BCGH-MASAC 的终局反馈、评分、策略诊断和审计日志分别整合在各自的 `guided_policy.py` 中。
两个调度器不依赖共享评分模块。
默认启用新版评分。网络结构、奖励、Actor/Critic 更新公式及现有引导强度保持原设置。
本改造修复评分口径，是否提高最终完成率和 SLA 满足率需要正式的多种子对照实验验证。

## 终局反馈口径

每个源 DC 维护 `(source, target)` 的终局历史，包含本地 `(i, i)`、远端 `(i, j)` 和开启时的云动作。
反馈只能由已闭合的 `FinalizedJobTrace` 写入。主机接纳或开始运行不算成功；
真正完成才记成功，主机拒绝、超时和最终失败都记失败。
被强制终止、没有实际 Host 步骤的 self 路由不伪造本地 Host 结果。
一个任务多次转发时，每个已执行的路由步骤各记一条同任务终局结果，使用 `job_id:sequence_index` 去重。
远端最终在其他 DC 或云完成时，原始转发动作同样获得该端到端结果；它衡量这次路由的后果，不表示目标 DC 一定本地吸收了任务。

所有动作使用相同规则：

- 成功分数：窗口内真正完成的任务比例。
- SLA 分数：窗口内真正完成且周转时间不超过 `sla_deadline_ratio × job.duration` 的比例，分母包含失败任务。
- 时间分数：仅对有完成时间的成功任务取平均周转时间 `T`，再计算 `D / (D + T)`；`D = max_job_duration × drop_deadline_ratio`，所有 DC 共用。
- 没有完成时间时，时间分数为中性值 0.5，并记录 `delay_known=false`。不把失败任务的缺失时间解释成零延迟。
- 无终局样本时，质量分数均为 0.5，置信度为 0，不产生质量偏置。
- 终局样本置信度为 `c = n / (n + bayesian_confidence_scale)`。
- 不良结果是“未完成或 SLA 未达标”，后验 `p = (alpha + n_bad) / (alpha + beta + n)`。默认 `alpha=beta=1`。

窗口按**终局结果可见的仿真时间**保留最近 `short_window_s` 秒，边界 `observed_at <= now-window` 过期。
因此等待很久才完成的任务在完成后仍能更新评分，不会因路由发生得早而直接丢失反馈。
每轮重置窗口、去重集合和诊断计数；时钟回退、未来结果和非法时间会报错。
旧的远端 Bayesian、资源压力和吸收统计继续保留自己的原有口径，供观察特征、诊断和旧模式使用。

## 同一公式

```text
benefit = w_success × success + w_sla × sla + w_delay × delay
p_effective = 0.5 + c × (p - 0.5)
U = 0.5 + c × (benefit - 0.5) - (p_effective - 0.5)
    - applied_resource_cost - applied_transit_penalty
bias = guidance_scale × (U - mean(U of feasible actions))
executed_logits = actor_logits + lambda × bias
```

最终 bias 统一缩放到 `max_logit_bias` 范围内，保持相对排序及中心化。
原有静态可行性约束保留；关闭引导或 `lambda=0` 时直接调用基础策略，作为无引导对照。

默认 `applied_resource_cost=0`、`applied_transit_penalty=0`。
远端历史资源成本、来源侧队列信号和转发吸收统计继续计算并记录，但默认不进入新版偏置。
资源和转发开关打开后，恢复对应旧惩罚作为消融；这两项仍是原先的远端惩罚，不能把这类消融称为完全对称的成本模型。
需要重新引入对称资源成本时，应先定义本地和远端共同可观测的成本来源，再单独验证。

## 配置与消融

`config.py` 新增：

```python
BCGH_SCORING_MODE = "symmetric"       # "legacy" 复现旧评分公式
BCGH_RESOURCE_PENALTY_ENABLED = False
BCGH_TRANSIT_PENALTY_ENABLED = False
BCGH_GUIDANCE_LOG_EVERY = 100         # 每 100 个策略决策记录 1 个详情；全部终局均记录
```

训练入口继续使用原模块，例如 `python -B -m schedulers.BCGH-MASAC.train_bcgh_masac`。
脚本调用时也可以通过 `dataclasses.replace(TrainConfig(), ...)` 独立设置各组，避免修改公共配置后混淆实验：

| 组 | enable_bayesian_game | enable_heuristic_guidance | scoring_mode | resource_penalty_enabled | transit_penalty_enabled |
| --- | --- | --- | --- | --- | --- |
| A 无引导 | False | False | symmetric | False | False |
| B 旧评分 | True | True | legacy | 忽略 | 忽略 |
| C 对称终局质量，保留旧额外惩罚 | True | True | symmetric | True | True |
| D 去除资源惩罚 | True | True | symmetric | False | True |
| E 新默认 | True | True | symmetric | False | False |

legacy 模式始终使用原资源项及原吸收惩罚配置，不受两个新开关控制。
为隔离评分因素，各组都保留相同的 `collect_neighbor_historical_feedback` 和 `use_neighbor_historical_feedback` 设置。
A 组可用于同一算法结构内部的无引导对照；与独立 H-MASAC 训练结果比较时，还应核对观察历史口径等设置。

固定同一个已验收 Host bundle、环境快照、负载、任务数量、训练预算和训练/评估种子。
BCGH 与 MR-BCGH 各自在自身架构内做 A–E 对照，使用独立输出目录且不加载其他组的训练 checkpoint。
评估函数支持关闭引导，不更新 Actor/Critic；每次评估建立独立的历史窗口，允许终局反馈随评估进程因果更新。
评估时不要把无反馈的 Actor 原始概率等同于关闭引导后重新运行整轮环境的结果。

## 日志与兼容性

Episode CSV 和 DC CSV 新增决策数、本地概率引导前后均值及差值、argmax 改变次数、原始/执行分布熵。
这些均值只统计实际调用引导的策略决策，随机预热和强制动作不混入；没有样本时均值留空。
Episode CSV 同时写入本轮实际评分模式、两个惩罚开关和实际 Bayesian/Guidance 开关。

同名 `.guidance.jsonl` 包含两类事件：

- `decision`：记录概率向量、动作映射、每个候选的质量、样本数、置信度、窗口年龄、效用、bias、可行性、原始成本和实际惩罚。
- `terminal`：记录每个任务的最终原因和时间、各路由步骤的终局反馈。

使用 `(episode, event_id)` 关联抽样决策与终局结果；`event_id=job_id:sequence_index`。
`base_probabilities` 是 Actor 原始 softmax；`guided_probabilities` 包含 bias 和原有可行性 mask。
解释概率变化时应结合候选的 `feasible` 字段，区分 mask 与评分的作用。
`congestion_probability`、`source_*`、`congestion_cost`、`transit_penalty` 保留旧诊断含义；
新版实际计算请看 `effective_probability`、`applied_resource_cost`、`applied_transit_penalty` 和 `guidance_utility`。

Guidance checkpoint 元数据升级到版本 4，并记录评分模式及开关。
旧版训练 checkpoint 不可直接续训为新版，避免在同一条曲线中混合两种评分语义；已有模型文件不会被转换或覆盖。
共享 Host bundle 的格式和权重不变。

## 验证

```powershell
python -B -m pytest tests/test_guidance_scoring.py tests/test_bcgh_short_window.py tests/test_mr_bcgh_masac.py -q --basetemp=tests/_pytest_guidance_run
```

覆盖本地/远端同数据同评分、本地失败降低偏置、缺失时间、终局时序、窗口过期、去重和重置、开关、旧公式、概率审计及终局关联。
已有短训练回归覆盖两种算法、云动作开关、实际 Routing 更新、共享 Host 冻结、checkpoint 保存/恢复和独立评估。
这些检查验证实现正确性；正式性能结论仍需足够训练预算和多种子统计。
