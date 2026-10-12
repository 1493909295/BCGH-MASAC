# 拥塞博弈与启发式引导改造（v5）

本版本覆盖 BCGH-MASAC 和 MR-BCGH-MASAC。BCGH 仍共享 Routing 参数，MR-BCGH 仍为每个 DC 保持独立网络。保留现有共享冻结 Host 流程，其他算法继续使用各自的 Routing 实现。

## 评分与拥塞博弈

本地、远端和云动作使用同一套终局质量定义。每个实际执行的路由步骤记录端到端完成、SLA 与周转时间；转发后在下一站完成仍可为原转发动作提供正反馈。另记录任务是否在该次目标访问中真正被接纳、是否在目标本地按时完成，两种成功不混用。

CPU/GPU 任务分为 low、medium、high 资源类（以目标的静态单主机容量为分母）。类样本不足时，以 `n_class/(n_class+4)` 混合类统计和 source→target 总统计。接纳条件下的终局统计单独提供；待完成任务不作为失败样本。

新版效用：

```text
B = 0.4 × completion + 0.4 × SLA + 0.2 × delay_quality
c = n_terminal / (n_terminal + confidence_scale)
U = 0.5 + c × (B - 0.5) - congestion_cost - residual_path_risk
b = centered(U) × guidance_scale
executed_policy = softmax(actor_logits + lambda × b), with static feasibility mask
```

质量只做一次显式置信度缩减。不良结果的 Beta 后验继续记录为诊断，不再在质量分数之外重复惩罚同一个终局失败。

拥塞博弈仍使用 CPU/GPU 凸势函数的边际增量 `Φ(L+d)-Φ(L)`，本地和远端对称计算并归一化。不同于旧窗口转发量，新 `L` 来自尚未释放的任务承诺：

- 已知任务需求按 `resource_request × estimated_remaining_service / residence_horizon` 折算。
- 尚未知晓开工时，使用目标驻留时间 EWMA 的指数生存估计；没有样本时初值 300 秒。
- 收到真实开工事件后，按已知任务时长扣除已运行时间。
- 获知任务再次转发、完成或丢弃后释放原目标承诺；重复反馈不会重复释放或重新加入。
- 未终结的承诺不因 1200 秒反馈窗口过期而直接消失。历史转发量另作诊断。
- 可通过 `BCGH_PRESSURE_BACKGROUND_CPU/GPU` 配置背景负载比例，默认 0。该估计不能代替不可见的远端本地到达负载；不会读取远端实时队列或剩余资源。
- 本地原生任务承诺只用于该源 DC 自己的 self 成本，不暴露给其他源 DC 的远端评分。远端原生负载仍按背景配置估计。

不再以“继续转发概率大”直接扣分。新版路径项只对已知链路传输时间超过剩余 SLA 预算、或再次访问已访问的 DC 加有限风险。时间和能耗的常规代价保留在既有终局质量/奖励中，不额外按转发次数重复计费。循环项是软约束，原有超时终止机制继续生效；没有新增硬跳数上限。

## 排队与有效本地完成

每次 Edge 转发只记录一个首个后继：`started / queued / edge / cloud / drop`。`queued` 不再记录为 `self` 或已完成。随后记录 queued→started→terminal 生命周期，终局区分按时完成、迟到完成、启动前丢弃、启动后丢弃。

环境新增独立的 `pop_job_service_events()` 开工事件接口，避免把开工通知当作奖励或完成事件。Host 接口可能推进仿真后才返回，因此区分实际接纳时间与结果可见时间。本地排队时间为 `start_time - admission_time`，不包含此前的传输等待。

Episode/DC CSV 新字段包含：

| 字段 | 定义 |
| --- | --- |
| `successor_started/queued/edge/cloud/drop_count` | 五类互斥首后继计数 |
| `successor_pending_count` | 首后继尚不可见 |
| `local_accepted_count` | started + queued |
| `local_terminal_count`, `local_pending_count` | 已接纳任务中已终结、未终结的数量 |
| `queue_started_count` | queued 中后来实际启动的数量 |
| `queue_completed_on_time_count`, `queue_terminal_count`, `queue_pending_count` | 排队组的按时完成数、终局分母、待终结数 |
| `effective_local_completion_rate` | 目标本地按时完成数 / 已接纳且已终结数 |
| `queue_sla_rate` | 排队后按时完成数 / 排队组已终结数 |
| `local_queue_wait_mean_s`, `local_queue_wait_samples` | 实际启动任务的本地排队均值与样本数 |

零分母留空。待终结任务单独记录，不强行算成功或失败。指标以“转发事件”为单位；一个多跳任务有多个转发事件。DC 行按转发目标归属。

旧的 `incoming_edge_absorbed_count`、`edge_target_absorption_*`、`first_edge_absorption_*` 保留历史的接纳口径（包含排队），供旧分析脚本读取；新增 `legacy_absorption_semantics=host_admitted_including_queued` 明示这一点。分析真正的本地有效完成请使用新字段。新日志 `guidance_schema_version=5`，不能与旧日志直接拼接后当作相同定义。

## 执行与学习一致

决策发生时捕获 `lambda × bias` 和可行性 mask。快照随 Pending Trace 进入 Replay，同任务真实下一跳的快照成为 next context；不会在任务最终完成后用更新过的缓存重新计算过去的偏置。

Actor、在线 Critic 和目标 Critic 都以同一有效偏置和静态不可行动作标记为上下文；新增分支零初始化。执行采样、Actor 更新、Critic 的下一状态 soft value、温度更新共用 `guided_distribution`。温度目标按相同 mask 的允许动作数量计算。终止经验不进入下一状态网络。随机预热经验也保存当时目标策略的引导上下文；关闭引导时仍为基础策略。

## 默认配置与复现

```python
BCGH_SCORING_MODE = 'symmetric'
BCGH_RESOURCE_PENALTY_ENABLED = True
BCGH_TRANSIT_PENALTY_ENABLED = True
BCGH_RISK_CONGESTION_COST_WEIGHT = 0.1
BCGH_GUIDANCE_LAMBDA_STAGE2_START = 0.3
BCGH_GUIDANCE_LAMBDA_STAGE2_PEAK = 0.3
BCGH_GUIDANCE_LAMBDA_STAGE3_END = 0.3
```

λ 默认在 Routing 训练及评估全程固定为 0.3，长期保留引导；与质量置信度承担不同作用。`max_logit_bias` 继续限幅。这些是保守起始参数，尚未经过完整训练调优。

训练入口不变：

```powershell
python -B -m schedulers.BCGH-MASAC.train_bcgh_masac
python -B -m schedulers.MR-BCGH-MASAC.train_mr_bcgh_masac
```

需要现有流程要求的同一已验收 Host bundle。检查点的 `bgh_guidance.version=5` 保留统计定义，新增 `guidance_context_version=1` 记录网络及回放上下文语义；缺少上下文版本的旧 Routing 权重需要重新训练。详见 [长期引导修改说明](PERMANENT_GUIDANCE.md)。不要覆盖原训练输出或把旧权重评价当成新版训练收益。

新增固定 Routing 权重、固定 Host、配对任务种子的评估入口：

```powershell
python -B evaluate_guidance.py --algorithm BCGH-MASAC --routing-checkpoint "新训练目录/final.pt" --host-checkpoint "model/shared_host/final" --environment "环境快照目录" --seeds 101,102,103,104,105 --output "analysis_outputs/guidance_v5_eval.csv"
```

对 MR 将算法参数改为 `MR-BCGH-MASAC`。脚本输出 full、no_guidance、no_congestion、no_path_risk、legacy_scoring 五组，拒绝覆盖已有文件。`--jobs 24` 可做小规模入口验证。

这属于**评估时消融**。要判断学习效果，需要各配置从头训练，固定相同 Host、拓扑、负载、预算及配对种子，再用未参与训练的多组种子比较完成率、SLA、延迟、能耗、本地有效完成及置信区间。当前改造和短训练测试不构成性能提升证明。

## 验证

新增 `tests/test_congestion_guidance_v5.py` 覆盖排队四种终局、未终结分母、重复通知、承诺过窗与释放、多跳成功、循环风险、资源类回退、Replay 快照复制，以及执行/Actor/温度/目标策略的一致性。既有测试覆盖冻结 Host 的训练、恢复、评估及云开关两条路径。

```powershell
python -B -m pytest tests -q --basetemp=_pytest_guidance_v5_check
```

每次检查使用新的 `--basetemp` 目录，避免影响历史测试产物。

2026-10-12 验证记录：全项目回归 123 项通过；补充远端信息边界后，相关回归 77 项通过；最终工作量/历史流量诊断专项 17 项复验通过。两条算法各完成 2 回合、每回合 24 任务的短训练，并各完成 5 组配对评估入口验证。产物位于 `analysis_outputs/guidance_v5_validation/`；任务规模和训练预算仅用于接口检查，不能用于判断算法性能。
