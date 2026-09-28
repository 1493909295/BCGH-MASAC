# OPT 全局负载可观测调度器

OPT 与 H-MASAC 使用相同的分层强化学习、奖励和环境执行规则。
Routing Actor 在训练及评估时都能读取全部 Edge DC 和 Cloud 的实时聚合负载，
通过离散 SAC 学习卸载策略。每个 Edge DC 的独立 LocalHostSAC 继续学习 Host 选择。
OPT 表示理想信息条件的学习基准，不保证数学意义上的全局最优。

## 观测与信息范围

Actor 输入顺序固定为：当前任务、所有 DC 的 13 维状态块、任务路由历史、
当前源 DC 的链路、原有邻居历史反馈。DC 顺序是 `edge_dc_ids + [cloud]`，
所有 Agent 使用相同顺序；当前 Agent 的 one-hot 身份由共享 Actor 内部拼接。

每个 DC 状态块保留 H-MASAC 的 CPU/GPU 容量、负载和剩余资源、等待/运行数量、
等待工作量、当前任务的立即/最终可行 Host 比例、预计启动和完成耗时比例。
这些是聚合特征，不能无损表示全部 Host 队列；时间估计仍沿用原有编码器。

当前 5 个 Edge DC 的配置下，Actor observation 为 127 维，拼接身份后为 132 维；
Critic 使用原有全局状态，Host observation 不变。Cloud 关闭时保留其观测块，
动作数为 5；Cloud 开启时动作数为 6。关闭历史反馈时对应字段仍占位并填零。

每次决策都读取当前 `env.dc_map`。历史经验保存独立副本，转发任务的后继观测
在下一真实决策时捕获。OPT 不读取尚未到达任务或未来事件，不增加未来负载预测、
资源预约、全局动作掩码或 BGH 策略偏置。状态共享假定实时且免费；任务传输仍有成本。

## 训练

在项目根目录运行：

```powershell
python -m schedulers.OPT.train_opt
```

也支持 `python schedulers/OPT/train_opt.py`。OPT 启动时先执行一次真实 CUDA 计算；
默认 `--device auto` 优先使用 `config.DEVICE` 指定的 GPU。若当前 PyTorch 构建
无法在该 GPU 上执行，程序会提示原因并改用 CPU。CPU 可运行完整训练，但速度可能
明显更慢。指定 `--device cuda:0` 会严格使用该 GPU，预检失败时立即报错；
指定 `--device cpu` 会直接使用 CPU。
将 OPT 代码复制到其他训练环境时，还需要同步本项目的
`schedulers/H-MASAC/h_masac_agent.py` 和 `schedulers/H-MASAC/train_h_masac.py`；
OPT 的 CPU 路径使用这两个文件中的共用 Agent 与训练流程。
默认超参数、奖励、任务生成及 Cloud 开关读取根目录 `config.py`。
三阶段仍由 `HOST_PRETRAIN_EPISODES`、`ROUTING_TRAIN_EPISODES` 和
`JOINT_FINETUNE_EPISODES` 控制：

| 阶段 | Routing | Host |
|---|---|---|
| Host 预训练 | 按现有流程本地接纳，不更新网络 | 更新 |
| Routing 训练 | 全局观测，更新网络 | 冻结 |
| 联合微调 | 全局观测，更新网络 | 更新 |

阶段长度可从入口覆盖，总回合数自动取三个阶段之和：

```powershell
python -m schedulers.OPT.train_opt --host-pretrain-episodes 200 --routing-train-episodes 500 --joint-finetune-episodes 300
```

对照实验建议通过 `--old-env` 指定同一份已保存环境，保持 DC/Host 顺序和链路一致：

```powershell
python -m schedulers.OPT.train_opt --old-env environment/env_keep/你的环境目录
```

未指定旧环境时，沿用环境生成器，在控制台打印新环境的保存目录；评估时使用该目录。

新环境需要 `config.HOST_DATASET_PATH` 指向可读取的 Host CSV（包含
`cpu_num`、`gpu_capacity_num` 两列），并需要 `config.JOB_DATASET_PATH`
指向任务 CSV。若数据文件放在其他位置，可在训练时指定：

```powershell
python -m schedulers.OPT.train_opt --host-dataset /你的路径/node_info_df.csv --job-dataset /你的路径/new_small_job_info_df.csv
```

已有环境快照可用 `--old-env` 加载，此时无须 Host 原始 CSV，仍需任务 CSV。
缺少数据文件或 `NUM_HOST < NUM_DATACENTERS` 时，程序会在创建环境前给出具体错误。

默认产物通过以下独立配置保存：

- `OPT_CHECKPOINT_DIR`：`model/OPT/checkpoints/`。
- `OPT_EPISODE_LOG_CSV_PATH`：`result/OPT/episode_log.csv`，运行时附加时间戳。
- `OPT_DC_LOG_CSV_PATH`：`result/OPT/dc_log.csv`，运行时附加时间戳。
- 日志目录中的 `current_train_log.txt` 和 `current_dc_log.txt` 指向当前日志。

沿用当前双层训练器的保存时机：进入联合微调时保存 `joint_finetune_start.pt`，
训练结束保存 `final.pt`，同时保存对应 `*_hosts/` 和 `*.trainer.json`。
OPT 检查点具有独立算法身份，并保存观测版本、特征及 DC 顺序、反馈设置和训练配置。

继续 OPT 训练时，指定匹配环境和 OPT 检查点，并将阶段总长度设为目标累计长度：

```powershell
python -m schedulers.OPT.train_opt --old-env environment/env_keep/你的环境目录 --resume model/OPT/checkpoints/final.pt --joint-finetune-episodes 400
```

旧 H/BGH Routing 模型不能作为 OPT Routing 模型加载。要复用已有 Host 权重，可设置
`OPT_HOST_INIT_CHECKPOINT` 或传入 `--host-init-checkpoint`，指向双层检查点的 Routing `.pt`
路径；实际只读取其配套 Host 文件。初始化会检查 DC/Host 顺序和观测/动作维度，
需要使用兼容的 Host 网络大小。Host 优化器和训练计数重新开始，随后按配置参加训练。
Host 初始化与完整 `--resume` 不能同时使用。

## 评估

```powershell
python -m schedulers.OPT.evaluate_opt --checkpoint model/OPT/checkpoints/final.pt --old-env environment/env_keep/你的环境目录 --episodes 5 --seed 42
```

评估加载检查点中的模型配置及反馈设置，使用同样的全局观测。
默认使用检查点保存的设备；需要在其他机器使用 CPU 时，可传 `--device cpu`。
默认确定性决策，`--stochastic` 可改为采样；不会更新模型或写入训练检查点。
每个评估回合从空历史反馈开始，并根据该回合已完成任务持续更新。
训练中的历史反馈仍按 H-MASAC 的方式跨回合累计，因此应对其他方法采用一致的评估初始化方式。

结果默认保存到 `result/OPT/evaluation.json`，可用 `--output` 修改。
包含完成率、SLA、平均及 P95 完成时间、能耗、Edge 转发次数和负载统计。
任务生成参数和 Cloud 开关仍读取当前 `config.py`，与对照实验保持一致。
Cloud 开关、DC/Host 顺序或观测结构不匹配时会拒绝加载。

程序调用入口为 `train_opt.train(...)`、`evaluate_opt.load_models(env, checkpoint)`
和 `evaluate_opt.evaluate_episode(env, routing_agent, host_agents, settings)`。
复用模块通过完整包名导入，不依赖 H/BGH 同名文件的搜索路径优先级。

## 验证

```powershell
python -B -m unittest discover -s tests -p test_opt_scheduler.py -v
```

测试覆盖远端负载可见性、固定 DC 顺序、后继全局观测及历史快照、Cloud 开关、
三阶段 Routing/Host 实际更新、模型保存/恢复、旧模型拒绝和评估期间参数不变。
GPU 与 CPU 集成测试使用临时目录和小网络，不会运行完整规模的实验。
