## 总体流程描述

新增 BCGH2-MASCA 后继状态奖励/折扣引导调度器，运行 `python -m schedulers.BCGH2-MASCA.train_bcgh2_masca`。方法、参数、评估和检查点规则见 [BCGH2-MASCA 使用说明](docs/BCGH2-MASCA/README.md)。

V100/V100S 服务器请按 [GPU 环境安装说明](服务器端V100环境安装命令.txt) 安装支持显卡架构的 PyTorch。
四个调度器在训练前检查实际 GPU 运算，失败时会给出当前构建信息及修复方法。

新增 OPT 全局负载可观测调度器，运行 `python -m schedulers.OPT.train_opt`。
训练、评估及检查点兼容规则见 [OPT 使用说明](schedulers/OPT/README.md)。

### 1. 阶段一 全局初始化

实例化环境  
    
    定义好物理环境中的：集群（数据中心）数量、集群容量、host数量、host容量、host与集群的归属关系、集群间的通信时延、host状态、host功耗、host间的通信时延；
    
    定义好任务到来速率、任务需求、任务达到集群；

    定义好系统时间步
实例化智能体

    为每个集群初始化智能体：智能体的网络结构、学习率、奖励、部分可观测、动作空间、经验池大小、经验池存放规则

### 2.阶段二 开启训练
    
清空
