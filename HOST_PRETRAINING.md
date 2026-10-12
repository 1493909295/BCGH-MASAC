# 共享 Host 预训练与 Routing 训练

BCGH-MASAC、BCGH2-MASAC、H-MASAC、MR-BCGH-MASAC、MR-H-MASAC 默认只训练 Routing。
五者加载同一个 Host bundle，按 DC 使用其中对应的 LocalHostSAC。
Host 使用确定性动作，无优化器、无训练经验池，不执行随机预热、网络更新或联合微调。
OPT 保留原训练方式。

## 1. 单独训练 Host

在项目根目录运行：

```powershell
python -B -m schedulers.pretrain_host
```

默认 GPU 为 `config.DEVICE`。训练参数可用 `--help` 查看。默认设置：

- 最少 1000 轮，最多 5000 轮；每个 DC 最少 20000 个终局样本、5000 次更新。
- 训练依次覆盖 0.5、1.0、1.5 倍到达率；主机、DC 和动作顺序保持固定。
- 每 50 轮使用 5 个独立固定验证种子，在三个负载下确定性评估。
- 每个 DC、每种验证负载都必须通过检查；不以全局均值代替。
- 最近 5 个评估窗口内，Host 平均任务回报相对波动不超过 3%；完成率/SLA 满足率绝对波动不超过 2%；完成时间与单位任务计算能耗相对波动不超过 5%。
- 最终回报至少比初始未训练策略提高 0.02，完成率至少 50%，SLA 满足率至少 30%；这两个服务指标也不能比初始策略下降超过 2 个百分点。
- 首次满足条件后，继续两个评估周期的训练；全部检查持续通过才发布最后一轮模型。

这些是工程验收标准，不是强化学习数学收敛证明。阈值需要结合实验负载解释；不能为了让实验显示“收敛”而事后静默放宽。
Host 指标按真正进入 Host 的任务统计；Routing 之前被强制终止的任务不进入 Host 指标分母。
验证使用独立环境，不写训练经验池、不更新网络。所有奖励继续使用现有任务因果链与终局奖励。

可指定固定环境快照：

```powershell
python -B -m schedulers.pretrain_host --old-env-path environment/env_keep/2026_10_10_09_40
```

也可指定独立实验输出目录及训练预算：

```powershell
python -B -m schedulers.pretrain_host --output-dir model/shared_host_experiment2 --min-episodes 1000 --max-episodes 8000
```

默认输出 `model/shared_host/`：

| 路径 | 内容 |
| --- | --- |
| `environment/` | 固定 DC、主机和拓扑快照 |
| `progress.json` | 最新训练进度与最近指标 |
| `train_log.jsonl` | 每轮训练指标及周期性验证结果 |
| `initial/` | 训练开始前、可续训的初始 checkpoint |
| `episode_XXXXXX/` | 每 50 轮的可续训 checkpoint |
| `unconverged_XXXXXX/` | 达到上限但未通过收敛验收的 checkpoint |
| `latest.json` | 最近一次完整保存的 checkpoint 路径 |
| `final/` | 自动收敛验收或明确人工验收后的共享模型 |

已有训练记录或 `final` 的目录不会被新训练覆盖。正式 `final` 采用临时目录完整写入、逐文件校验后再提交。
它包含每个 DC 的 Actor、Critic、Target Critic、温度、优化器和更新计数，以及结构、归一化、训练配置、最终验证结果和文件 SHA-256。

## 2. 恢复 Host 预训练

从 `latest.json` 读取 checkpoint 路径，例如：

```powershell
python -B -m schedulers.pretrain_host --resume model/shared_host/unconverged_005000 --max-episodes 8000
```

恢复会加载网络、优化器、计数、随机状态和收敛历史。默认不保存大容量经验池，恢复后重新积累可采样经验；因此不宣称逐步复现原轨迹。
需要恢复完整经验池时，从首次训练开始传入 `--save-replay`，这会明显增加磁盘占用。
续训不允许更改奖励、验证集、学习参数或收敛标准；可以延长最大轮数、改变设备及保存间隔。
意外中断后优先从最近完整 checkpoint 恢复，不把半轮状态伪装成已完成回合。

## 3. 五个调度器使用同一个 Host

本次用户指定使用第 **1550** 轮模型作为人工验收通过的共享 Host。原始
`episode_001550/` 保留，发布到 `model/shared_host/final/` 时不改变任何权重或验证指标。
manifest 使用 `status="accepted_manual"`，记录用户验收原因、来源轮次、来源文件摘要，
并明确记录自动收敛检查是否通过；人工验收不会将实际未通过的检查伪装为通过。

人工验收只能在用户明确授权后执行，例如本次：

```powershell
python -X utf8 -B -m schedulers.accept_host model/shared_host/episode_001550 --reason "用户指定第1550轮作为成熟Host模型，人工验收通过"
```

此操作拒绝覆盖已有 final，完整校验源文件，逐字节复制模型后发布。

`config.py` 中统一配置：

```python
SHARED_HOST_CHECKPOINT = "model/shared_host/final"
SCHEDULER_ROUTING_EPISODES = 500
```

随后运行原训练入口，例如：

```powershell
python -B -m schedulers.H-MASAC.train_h_masac
python -B -m schedulers.BCGH-MASAC.train_bcgh_masac
python -B -m schedulers.BCGH2-MASAC.train_bcgh2_masac
python -B -m schedulers.MR-BCGH-MASAC.train_mr_bcgh_masac
python -B -m schedulers.MR-H-MASAC.train_mr_h_masac
```

程序接口仍使用各调度器的 `TrainConfig`，新增 `host_checkpoint` 字段。自定义轮数时需让
`num_episodes == routing_train_episodes`，并将 `host_pretrain_episodes`、`joint_finetune_episodes` 设为 0。
Host 网络尺寸来自 bundle，`host_sac_config` 在调度器训练中仅提供推理设备；不会覆盖预训练结构或恢复优化器。

默认自动加载 bundle 的硬件快照，因此更换 Routing 训练种子不会重新随机分配主机。
显式提供 `old_env_path` 时，会检查 DC/主机顺序、容量和 Host 观察量归一化是否兼容。
缺失、损坏、既未自动收敛验收也未明确人工验收、或结构不兼容的 Host 都会报错，不自动训练替代模型。

BCGH 与 MR-BCGH 的 Guidance 强度按整个 Routing 预算计算：前半段从 start 到 peak，后半段从 peak 到 end；单轮训练使用 end。
沿用原 `guidance_lambda_stage2_start/stage2_peak/stage3_end` 配置名，但不再代表 Host/联合训练阶段。
BCGH2 的 Routing 奖励和折扣修正不进入 Host。

## 4. 调度器 checkpoint 与评估

调度器仍保存 `final.pt`、`final.trainer.json` 和 `final_hosts/`，便于兼容已有读取布局。
`final_hosts/` 是原始共享 Host 文件的逐字节副本。metadata 和 Episode/DC 日志记录 Host bundle ID 与 manifest SHA-256。
日志中 `host_training_update_count` 和 Host 训练动作计数为 0；原 `host_update_step` 展示预训练时已有的更新次数。
Host 终局事件、奖励和贝叶斯证据仍正常处理，零存储经验接收器不保留训练样本。

续训会在加载任何 Routing 权重前检查全部 Host 文件和共享模型身份。
旧三阶段 checkpoint 或使用其他 Host 的 checkpoint 不能直接作为新模式续训；需要另行显式迁移。
BCGH 系列评估接口检查所用 Host 与训练配置一致，即使随机评估 Routing，Host 也保持确定性。

## 验证

```powershell
python -B -m pytest tests -q --basetemp tests/_pytest_shared_host
```

回归测试包含五个调度器真实更新与续训、Host 全状态不变、坏文件在模型加载前被拒绝、主机顺序校验、未收敛拒绝加载、预训练上限/续训/最后权重发布，以及 OPT 原有训练路径。
测试中的合成 `converged` bundle 只用于检查接口，不能作为正式预训练结果。
