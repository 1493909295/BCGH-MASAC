# BCGH 短窗口引导 v3

贝叶斯结果、收益、Actor 的邻居反馈和竞争需求都只保留当前回合的
`(t - BCGH_SHORT_WINDOW_S, t]`。默认窗口为 1200 仿真秒，数据集任务时长中位数
约为 616 秒；该值是初始实验参数，不是最优窗口的结论。
较长任务的结果可能因转发时间过旧而被排除，因此窗口结果统计偏向较快闭合的任务。

## 信号与信息边界

- 结果：任务终止后提供唯一的 `job_id:sequence_index` 证据；同时保存转发与
  终止时间。迟到的旧转发结果不会重新成为当前证据，同一证据不会重复更新。
- 竞争：实际执行 Edge-to-Edge 转发后立即计入 CPU/GPU 需求；任务终止时
  不再计入。随机预热转发同样消耗资源，因此计入竞争。
- 来源：接收端真正出现路由决策时记录直接上一跳。查询只扫描当前源 DC
  的主机等待/执行队列，按接收时间线性衰减，在窗口外归零。
  最初来源字段 `origin_datacenter` 不作为上一跳。队列反复扫描不累加后验。
  随机预热卸载不作为主动卸载的来源证据；正常策略采样的卸载仍是弱证据。
- 容量：初始化时从 `base_datacenters` 提取安装的 CPU/GPU 和单主机容量。
  博弈核心不持有环境引用，不读取邻居的实时队列、使用量或可用资源。
  竞争统计沿用共享的已执行路由历史，并非完全分布式的私有信息实现。
- 吸收：每次 `source -> target` Edge 转发在 target 作出下一次真实路由动作时，
  分类为 SELF、EDGE、CLOUD 或 DROP。SELF 还要等待真实 Host 结果，只有
  started/queued 算本地吸收，Host 拒绝算 DROP。随机、强制或编排动作保留在
  行为统计中，默认不进入策略吸收后验。

## 计算

窗口内结果构成 Beta(alpha0 + bad, beta0 + good)。本地来自邻居 j 的任务
按 `max(cpu/local_cpu_capacity, gpu/local_gpu_capacity)` 汇总资源比例，再按
`max(0, 1 - age/T)` 加权。等待与执行任务分别统计，再相加为 V。
`s = V/(V + source_volume_scale)`，`pseudo = source_evidence_weight * s`。
融合风险为 `(alpha + pseudo)/(alpha + beta + pseudo)`；来源软证据是有界的
启发式近似，不是校准过的物理拥塞概率。无入站任务不构成空闲证据。

每类资源的压力为 `x = recent_dispatched_demand / installed_capacity`，当前任务
增量为 `d = request / installed_capacity`。原始竞争成本按 CPU/GPU 权重求和：
`a*d + b*(2*x*d + d*d)`，再以 `cost/(cost + resource_cost_scale)` 映射到 [0,1)。
不存在可容纳任务的单主机时，远端候选在引导策略中被静态掩码排除。
Self 和 Cloud 保留原有专门处理；这不是远端实时可执行性检测。

每个有向 Edge Pair 对后继动作维护四分类 Dirichlet 后验。吸收证据按任务
`CPU/GPU 型 × low/medium/high` 分类；类别样本不足时按
`n_class/(n_class+4)` 回退到 Pair 总体后验。总体吸收置信度为 `n/(n+8)`。
无样本时置信度为零，因此均匀先验不会惩罚尚未探索的邻居。

收益仍为 `0.4*success + 0.4*sla + 0.2*delay_quality`，全部来自短窗口，
不增加能耗项。中转原始成本为
`0.7*P(edge) + 0.5*P(cloud) + 1.0*P(drop)`，乘吸收置信度后仅作用于
其他 Edge DC 候选。原始效用为
`benefit - fused_probability - weight*resource_cost - transit_penalty`。
终局 Beta 风险只统计失败和 SLA 违约；`reforwarded` 仍可用于诊断和 Actor
历史反馈，但不再重复进入终局拥塞后验。

置信度按证据来源分别处理：结果为 `n/(n+20)`，来源为新鲜度加权的独立接收
事件数 `m/(m+5)`。为了避免空窗口的 0.5 先验成为真实惩罚，实际偏置使用：

```
effective_probability = 0.5 + outcome_conf*(outcome_probability - 0.5)
                            + source_conf*(fused_probability - outcome_probability)
guidance_utility = 0.5 + outcome_conf*(benefit - 0.5)
                       - (effective_probability - 0.5)
                       - cost_weight*normalized_cost - transit_penalty
```

已执行资源需求的竞争成本不受结果样本置信度屏蔽。否则没有出站结果时，
新接入的来源证据和竞争成本都可能不起作用。日志同时保存原始 utility 和
guidance_utility，便于区分效用与置信度调整。效用中心化后乘 guidance_scale，
必要时整体缩放使每项绝对值不超过 max_logit_bias，保持动作间排序及零均值。

## 调度、日志和检查点

Host 预训练 lambda=0；Routing 阶段 0→1；Joint 阶段 1→0.3。
单回合阶段直接使用该阶段的结束值。偏置尺度仍是 0.3，末期外部总缩放为
0.3×0.3；并非 30% 动作由启发式接管。Actor/Critic 的 SAC 更新和原有能耗奖励不变。

每次运行在 episode CSV 同目录输出 `*.guidance.jsonl`，记录动作、lambda、
各目标结果概率、来源贡献、当前样本量、资源压力、增量成本、四分类吸收概率、
资源类别、吸收置信度、中转成本、原始效用、实际引导效用和偏置。Episode/DC
CSV 另外记录首次 Edge 吸收率、Edge→Edge、Edge→Cloud、DROP、P95 跳数以及
每个目标 DC 的入站去向。反馈字段沿用原有七维字段名，`*_ewma` 是窗口均值。

检查点沿用兼容字段 `bgh_guidance.version=3` 和全部参数。回合边界保存，无需携带跨回合
窗口状态。开启 BCGH 时，旧累计信念检查点或不同参数的检查点不能静默续训。
MASAC/H-MASAC 基线文件未修改；关闭 BCGH 两开关仍走原来的等价路径。

## 评估

`schedulers.BCGH-MASAC.training_support.evaluate_episode(env, routing_agent, host_agents, settings)`
用于已加载模型；环境拓扑和主机顺序必须与训练检查点一致，settings 使用检查点
`bgh_guidance.parameters` 中的配置。函数在一次全新回合里更新相同的短窗口状态，
固定使用 `guidance_lambda_stage3_end`（默认 0.3），不更新网络、不写训练检查点。
仅调用基础 Actor 的 select_action 不代表完整 BCGH 方法。

## 验证

运行 `python -B -m unittest discover -s tests -p test_bcgh_short_window.py -v`。
行为测试不要求 GPU；三阶段训练与完整评估的集成测试沿用项目 CUDA 要求，
使用小网络和 24 个任务，在专用临时目录保存并清理检查点与日志。
这些测试验证接线和性质，不构成性能提升或均衡收敛的证明。
