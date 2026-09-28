import pandas as pd
import numpy as np
import random
import math
from environment.job import JobList
from environment.energy_model import (
    calculate_edge_host_power_components_w,
    calculate_edge_host_total_power_w,
)
import networkx as nx
import config as conf
# import matplotlib.pyplot as plt



# sns.set_theme(style="whitegrid")
# matplotlib.use('TkAgg')
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
# plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号···

class Host:
    RESOURCE_EPS = 1e-9
    POWER_EPS = 1e-9

    def __init__(self, host_id,  gpu_capacity_num, cpu_num):

        self.host_id = str(host_id)  # 主机ID
        # self.gpu_model = str(gpu_model)
        self.gpu_capacity_num = int(gpu_capacity_num)
        self.cpu_num = int(cpu_num)

        self.cpu_load = 0.0
        self.gpu_load = 0.0

        self.used_cpu: float = 0.0
        self.used_gpu: float = 0.0

        self.waiting_queue = JobList()
        self.running_queue = JobList()
        self.completed_queue = JobList()

    # 用于修复浮点计算精度误差带来的恶性训练暂停bug
    def _sanitize_resource_usage(self) -> None:
        cpu_capacity = float(self.cpu_num)
        gpu_capacity = float(self.gpu_capacity_num)
        if not math.isfinite(self.used_cpu):
            raise RuntimeError(
                f"Host {self.host_id} used_cpu 非有限数值: "
                f"{self.used_cpu}"
            )
        if not math.isfinite(self.used_gpu):
            raise RuntimeError(
                f"Host {self.host_id} used_gpu 非有限数值: "
                f"{self.used_gpu}"
            )
        if abs(self.used_cpu) <= self.RESOURCE_EPS:
            self.used_cpu = 0.0

        if abs(self.used_gpu) <= self.RESOURCE_EPS:
            self.used_gpu = 0.0

        if abs(self.used_cpu - cpu_capacity) <= self.RESOURCE_EPS:
            self.used_cpu = cpu_capacity

        if abs(self.used_gpu - gpu_capacity) <= self.RESOURCE_EPS:
            self.used_gpu = gpu_capacity

        if (
                self.used_cpu < -self.RESOURCE_EPS
                or self.used_cpu > cpu_capacity + self.RESOURCE_EPS
        ):
            raise RuntimeError(
                f"Host {self.host_id} CPU 资源记账异常："
                f"used_cpu={self.used_cpu}, "
                f"capacity={cpu_capacity}, "
                f"running_count={len(self.running_queue)}"
            )

        if (
                self.used_gpu < -self.RESOURCE_EPS
                or self.used_gpu > gpu_capacity + self.RESOURCE_EPS
        ):
            raise RuntimeError(
                f"Host {self.host_id} GPU 资源记账异常："
                f"used_gpu={self.used_gpu}, "
                f"capacity={gpu_capacity}, "
                f"running_count={len(self.running_queue)}"
            )

            # 这里只可能清理 RESOURCE_EPS 范围内的边界误差。
        self.used_cpu = min(max(self.used_cpu, 0.0), cpu_capacity)
        self.used_gpu = min(max(self.used_gpu, 0.0), gpu_capacity)

        if self.running_queue.is_empty():
            if self.used_cpu != 0.0 or self.used_gpu != 0.0:
                raise RuntimeError(
                    f"Host {self.host_id} 状态不一致："
                    f"running_queue 已空，"
                    f"但 used_cpu={self.used_cpu}, "
                    f"used_gpu={self.used_gpu}"
                )

    # 初始化host
    def initialize_host(self):
        self.waiting_queue.clear()
        self.running_queue.clear()
        self.completed_queue.clear()
        self.cpu_load = 0.0
        self.gpu_load = 0.0
        self.used_cpu = 0.0
        self.used_gpu = 0.0

    # 更新host状态
    def update_hardware_capacity(self, cpu_num: int, gpu_capacity_num: int):
        self.cpu_num = int(cpu_num)
        self.gpu_capacity_num = int(gpu_capacity_num)
        self.calculate_load()

    # 负载计算
    def calculate_load(self):
        self._sanitize_resource_usage()
        if self.cpu_num > 0:
            # total_cpu_run = self.running_queue.get_total_cpu_demand()
            self.cpu_load = min(max(self.used_cpu / self.cpu_num, 0.0), 1.0)
        else:
            self.cpu_load = 0.0

        if self.gpu_capacity_num > 0:
            # total_gpu_run = self.running_queue.get_total_gpu_demand()
            self.gpu_load = min(max(self.used_gpu / self.gpu_capacity_num, 0.0), 1.0)
        else:
            self.gpu_load = 0.0

        return (self.cpu_load, self.gpu_load)

    # Edge Host IT 功率模型
    def get_power_components(self) -> dict:
        return calculate_edge_host_power_components_w(host=self,)

    def get_total_power(self) -> float:
        return calculate_edge_host_total_power_w(host=self,)


    # 检查总容量能不能满足资源要求
    def can_ever_accommodate(self, job) -> bool:
        job_cpu = float(job.cpu_request)
        job_gpu = float(job.gpu_request)

        if (job_cpu > float(self.cpu_num) + self.RESOURCE_EPS):
            return False

        if (job_gpu > float(self.gpu_capacity_num) + self.RESOURCE_EPS):
            return False

        return True

    # 返回经过浮点误差清理后的当前可用 CPU
    def get_available_cpu(self) -> float:
        self._sanitize_resource_usage()
        available_cpu = (float(self.cpu_num) - float(self.used_cpu))
        if abs(available_cpu) <= self.RESOURCE_EPS:
            return 0.0
        return max(available_cpu, 0.0)

    # 返回经过浮点误差清理后的当前可用 GPU
    def get_available_gpu(self) -> float:
        self._sanitize_resource_usage()
        available_gpu = (float(self.gpu_capacity_num) - float(self.used_gpu))
        if abs(available_gpu) <= self.RESOURCE_EPS:
            return 0.0
        return max(available_gpu, 0.0)

    # 检查当前资源剩余量是否满足卸载要求
    def can_accommodate(self, job) -> bool:
        self._sanitize_resource_usage()
        job_cpu = float(job.cpu_request)
        job_gpu = float(job.gpu_request)
        if self.used_cpu + job_cpu > float(self.cpu_num) + self.RESOURCE_EPS:
            return False
        if self.used_gpu + job_gpu > float(self.gpu_capacity_num) + self.RESOURCE_EPS:
            return False
        return True

    # 加入等待队列
    def add_to_waiting_queue(self, job):
        self.waiting_queue.push(job)

    # 移出等待队列
    def remove_from_waiting_queue(self):
        return self.waiting_queue.pop()

    # 加入运行队列
    def add_to_running_queue(self, job, current_time: float):
        if not self.can_accommodate(job):
            # print(f"调度失败：主机 {self.host_id} 资源不足，无法运行任务 {job.job_id}。")
            return False
        job.mark_as_started(current_time)
        self.running_queue.push(job)
        self.used_cpu += float(job.cpu_request)
        self.used_gpu += float(job.gpu_request)
        self._sanitize_resource_usage()
        self.calculate_load()  # 联动计算负载
        # print(f"成功，主机{self.host_id}接收了任务{job.job_id}")
        return True

    # 根据任务id移出运行队列
    def remove_from_running_queue(self, job_id: str):
        removed_job = self.running_queue.remove_by_id(job_id)
        if removed_job is None:
            return  None
        self.used_cpu -= float(removed_job.cpu_request)
        self.used_gpu -= float(removed_job.gpu_request)
        self._sanitize_resource_usage()
        # self.used_cpu = max(self.used_cpu, 0.0,)
        # self.used_gpu = max(self.used_gpu, 0.0,)
        self.calculate_load()
        return removed_job

    # 加入完成队列
    def add_to_completed_queue(self, job, current_time: float):
        job.mark_as_finished(current_time)
        self.completed_queue.push(job)

    # 打印等待队列
    def print_waiting_queue(self):
        print(f"\n--- [主机 {self.host_id}] 等待队列 (共 {len(self.waiting_queue)} 个任务) ---")
        if self.waiting_queue.is_empty():
            print("  (空)")
        else:
            for i, job in enumerate(self.waiting_queue._queue):
                print(f"  {i + 1}. {job}")

    # 打印运行队列
    def print_running_queue(self):
        print(f"\n--- [主机 {self.host_id}] 运行队列 (共 {len(self.running_queue)} 个任务) ---")
        if self.running_queue.is_empty():
            print("  (空)")
        else:
            for i, job in enumerate(self.running_queue._queue):
                print(f"  {i + 1}. {job}")

    # 打印完成队列
    def print_completed_queue(self):
        print(f"\n--- [主机 {self.host_id}] 完成队列 (共 {len(self.completed_queue)} 个任务) ---")
        if self.completed_queue.is_empty():
            print("  (空)")
        else:
            for i, job in enumerate(self.completed_queue._queue):
                print(f"  {i + 1}. {job}")

    # 打印host详细情况
    def print_host_details(self):
        print(f"\n" + "=" * 20 + f" 主机 {self.host_id} 详细面板 " + "=" * 20)
        print("【物理资源总览】")
        print(f"  - CPU 总量 : {self.cpu_num} 核")
        # print(f"  - GPU 总量 : {self.gpu_capacity_num} 单位 (显卡型号: {self.gpu_model})")
        self.calculate_load()
        print("【实时负载状态】")
        print(f"  - CPU 负载率 : {self.cpu_load * 100:>5.1f}%")
        print(f"  - GPU 负载率 : {self.gpu_load * 100:>5.1f}%")
        print("【内部任务排布】")
        self.print_waiting_queue()
        self.print_running_queue()
        self.print_completed_queue()

    def __repr__(self):
        return f"Host(ID: {self.host_id}, CPU: {self.cpu_load:.2f}, GPU: {self.gpu_load:.2f})"

class DataCenter:
    def __init__(self, dc_id, cloud_latency):
        self.dc_id = str(dc_id)
        self.host_list = []
        # self.job_arrival_list = []
        self.cloud_latency = float(cloud_latency)
        self.dc_cpu_load = 0.0
        self.dc_gpu_load = 0.0

    # 计算数据中心的负载，返回CPU负载、GPU负载
    def calculate_dc_loads(self):

        total_cpu_capacity = 0
        total_gpu_capacity = 0
        total_cpu_used = 0
        total_gpu_used = 0

        for host in self.host_list:
            host.calculate_load()
            total_cpu_capacity += float(host.cpu_num)
            total_gpu_capacity += float(host.gpu_capacity_num)
            total_cpu_used += float(host.used_cpu)
            total_gpu_used += float(host.used_gpu)

        if total_cpu_capacity > 0.0:
            self.dc_cpu_load = min(max(total_cpu_used / total_cpu_capacity, 0.0), 1.0)
        else:
            self.dc_cpu_load = 0.0

        if total_gpu_capacity > 0.0:
            self.dc_gpu_load = min(max(total_gpu_used / total_gpu_capacity, 0.0), 1.0)
        else:
            self.dc_gpu_load = 0.0

        return self.dc_cpu_load, self.dc_gpu_load

    # 打印数据中心状态
    def print_datacenter_details(self):
        self.calculate_dc_loads()

        print(f"\n" + "=" * 25 + f" 数据中心 【{self.dc_id}】 详情 " + "=" * 25)
        print(f"  - 节点中心 ID       : {self.dc_id}")
        print(f"  - 挂载主机总数      : {len(self.host_list)} 台")
        print(f"  本地边端 ──(时延: {self.cloud_latency:.2f} s)──> 云计算中心")
        print(f"  - 区域 CPU 整体利用率 : {self.dc_cpu_load * 100:>5.1f}%")
        print(f"  - 区域 GPU 整体利用率 : {self.dc_gpu_load * 100:>5.1f}%")

        print("\n【板块三：下属主机实时明细】")
        if not self.host_list:
            print("  -> [ 提示：当前数据中心暂未初始化或挂载任何物理主机 ]")
        else:
            for host in self.host_list:
                host.print_host_details()

        print(f"\n" + "=" * 50 )

# 生成 host 待分配列表
def hosts_generate(wait_assign_host_list: list, host_num: int, host_dataset_path: str) -> list:
    print("\n开始生成 host 列表...\n")

    if host_num <= 0:
        raise ValueError(
            f"NUM_HOST 必须大于 0，当前为 {host_num}；请检查 config.py。"
        )

    try:
        df = pd.read_csv(host_dataset_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Host 数据集不存在：{host_dataset_path}。"
            "请检查 config.HOST_DATASET_PATH，或为 OPT 指定 --host-dataset；"
            "如果已有环境快照，也可使用 --old-env。"
        ) from exc

    if df.empty:
        raise ValueError(f"Host 数据集没有可采样的记录：{host_dataset_path}")
    missing_columns = {"cpu_num", "gpu_capacity_num"} - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Host 数据集缺少列 {sorted(missing_columns)}：{host_dataset_path}"
        )

    # 在数据集中有放回抽样作为host配置的真实来源
    sampled_nodes = df.sample(n=host_num, replace=True).reset_index(drop=True)
    for index, row in sampled_nodes.iterrows():
        host_id = f"host_{index}"
        new_host = Host(
            host_id=host_id,
            # gpu_model=row.get('gpu_model', 'Unknown'),  # 默认设为 'Unknown' 防错
            gpu_capacity_num=row.get('gpu_capacity_num', 0),  # 默认数量设为 0
            cpu_num=row.get('cpu_num', 0)  # 默认数量设为 0
        )
        wait_assign_host_list.append(new_host)
    return wait_assign_host_list

def datacenters_generate(datacenter_num: int, wait_assign_host_list: list, cloud_latency_range: tuple, edge_latency_range: tuple):
    if len(wait_assign_host_list) < datacenter_num:
        raise ValueError(f"环境生成失败：待分配的 Host 数量 ({len(wait_assign_host_list)}) 必须大于等于 Datacenter 数量 ({datacenter_num})！")

    datacenters = []

    # 根据范围生成截断的正态分布随机数
    def get_normal_latency(val_range):
        low, high = val_range
        mu = (low + high) / 2  # 均值
        sigma = (high - low) / 6  # 标准差（3-sigma法则保证大部分数据在范围内）
        # 生成正态分布随机数，并用 clip 强行限制在 [low, high] 之间
        return round(float(np.clip(np.random.normal(mu, sigma), low, high)), 2)

    # 为每个数据中心生成一个云端时延
    for i in range(datacenter_num):
        dc_id = f"DC-{i + 1}"
        dc = DataCenter(dc_id = dc_id,cloud_latency = get_normal_latency(cloud_latency_range))
        # dc.cloud_latency = get_normal_latency(cloud_latency_range)
        datacenters.append(dc)

    # 构建全连接图
    G = nx.Graph()
    # 绑定顶点
    for dc in datacenters:
        G.add_node(dc.dc_id, dc_instance=dc)
    # 绑定边
    for i in range(datacenter_num):
        for j in range(i + 1, datacenter_num):
            edge_latency = get_normal_latency(edge_latency_range)
            G.add_edge(datacenters[i].dc_id, datacenters[j].dc_id, weight=edge_latency)

    # 分配 host
    shuffled_hosts = wait_assign_host_list.copy()
    random.shuffle(shuffled_hosts)
    for i in range(datacenter_num):
        datacenters[i].host_list.append(shuffled_hosts[i])
    for host in shuffled_hosts[datacenter_num:]:
        chosen_dc = random.choice(datacenters)
        chosen_dc.host_list.append(host)
    print(f"\n✅ 成功生成 {datacenter_num} 个边缘数据中心。开始打印详细状态：")

    # 绘图
    # plt.figure(figsize=(10, 8))
    # pos = nx.spring_layout(G, seed=42)
    # nx.draw(G, pos, with_labels=True,
    #         node_color='lightblue', node_size=2500,
    #         font_size=12, font_weight='bold', edge_color='gray')
    # edge_labels = nx.get_edge_attributes(G, 'weight')
    # edge_labels_formatted = {k: f"{v} ms" for k, v in edge_labels.items()}
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels_formatted, font_size=12, font_color='red')
    # plt.title("Edge DataCenters Fully-Connected Topology", fontsize=16)
    # # plt.show()
    return datacenters, G

# if __name__ == "__main__":
#     NUM_HOSTS = 100
#     NUM_DATACENTERS = 8
#
#     # 边缘节点到云端的时延范围 (ms)
#     CLOUD_LATENCY_RANGE = (50, 150)
#     # 边缘节点互相之间的时延范围 (ms)
#     EDGE_LATENCY_RANGE = (5, 20)
#
#     print("🚀 开始环境生成测试...\n")
#
#     # ================= 2. 测试生成 Host =================
#     empty_host_list = []
#
#     # 调用我们之前写的 host_generate 函数 (注意：如果有 s，请写 hosts_generate)
#     my_hosts = hosts_generate(empty_host_list, NUM_HOSTS,conf.HOST_DATASET_PATH)
#     print(f"✅ 成功从数据集中实例化了 {len(my_hosts)} 个 Host。")
#
#     # ================= 3. 测试生成 Datacenter 并分配 =================
#     print(f"✅ 开始实例化 {NUM_DATACENTERS} 个 Datacenter，并构建拓扑与主机分配...\n")
#
#     # 调用 datacenters_generate 会在内部自动分配主机、建立拓扑、
#     # 打印每个数据中心的详情 (dc.print_datacenter_details) 并弹出可视化拓扑图。
#     my_datacenters, topology_graph = datacenters_generate(
#         datacenter_num=NUM_DATACENTERS,
#         host_list=my_hosts,
#         cloud_latency_range=CLOUD_LATENCY_RANGE,
#         edge_latency_range=EDGE_LATENCY_RANGE
#     )
#
#     # ================= 4. 打印最终分配情况汇总 =================
#     print("\n" + "=" * 20 + " Datacenter 主机分配情况汇总 " + "=" * 20)
#     total_allocated_hosts = 0
#
#     for dc in my_datacenters:
#         # 提取该数据中心分配到的所有 host 的 ID
#         allocated_host_ids = [host.host_id for host in dc.host_list]
#         host_count = len(allocated_host_ids)
#         total_allocated_hosts += host_count
#
#         print(f"【{dc.dc_id}】 分配到 {host_count} 台主机")
#         # 如果你想看具体的列表可以取消下面这行的注释
#         # print(f"  --> 主机列表: {allocated_host_ids}")
#
#     print("-" * 69)
#     print(f"📊 校验分配总数: 待分配 {NUM_HOSTS} 台 / 实际已分配 {total_allocated_hosts} 台")
#
#     for dc in my_datacenters:
#         dc.print_datacenter_details()
#
#     # 简单断言验证
#     if NUM_HOSTS == total_allocated_hosts:
#         print("🎉 测试完美通过：所有 Host 已被完全且不遗漏地分配到了各个数据中心！")
#     else:
#         print("⚠️ 测试出现异常：分配的主机数量存在偏差，请检查拷贝和打乱逻辑！")
