import pandas as pd
import numpy as np

from functools import lru_cache
from typing import Optional


class Job:

    def __init__(self,
                 job_id,
                 cpu_request,
                 gpu_request,
                 duration,
                 target_datacenter=None,
                 origin_datacenter=None):

        # 基础静态属性 (创建后通常不变)
        self.job_id = job_id
        self.cpu_request = cpu_request
        self.gpu_request = gpu_request
        self.duration = duration

        # 动态属性 (随调度过程修改)
        self.target_datacenter = target_datacenter

        # 初始到达 DC 是不可变的物理事实。target_datacenter 会随转发改变，
        # 因此不能再用它统计每个 DC 的原生任务到达量。
        self._origin_datacenter = (
            str(origin_datacenter)
            if origin_datacenter is not None
            else None
        )
        self.arrive_time = 0.0
        self.start_time = None
        self.finish_time = None

        # 任务级能耗记录
        self.compute_energy_j = 0.0
        self.transfer_energy_j = 0.0
        self.edge_edge_transfer_energy_j = 0.0
        self.edge_cloud_transfer_energy_j = 0.0

        self.routing_hop_count = 0
        self.cumulative_transfer_latency_s = 0.0


    # 更新任务所属数据中心
    def set_target_datacenter(self, datacenter_id):
        self.target_datacenter = datacenter_id

    @property
    def origin_datacenter(self):
        """任务首次进入系统时的 Edge DC；设置后不可改写。"""
        return getattr(self, "_origin_datacenter", None)

    def set_origin_datacenter(self, datacenter_id):
        origin_dc_id = str(datacenter_id)
        current_origin = self.origin_datacenter

        if current_origin is not None and current_origin != origin_dc_id:
            raise ValueError(
                f"任务 {self.job_id} 的 origin_datacenter 已固定为 "
                f"{current_origin}，不能改写为 {origin_dc_id}。"
            )

        self._origin_datacenter = origin_dc_id

    # ==============================================================
    # 记录一次已经真实发生的 Edge -> Edge Routing hop
    #
    # 注意：
    #   这里只记录累计物理事实。
    #
    #   不记录：
    #       source DC
    #       target DC
    #       route path
    #       visited DC
    #
    #   这些完整因果信息继续属于 PendingJobTraceStore。
    # ==============================================================

    def record_edge_routing_hop(
            self,
            latency_s,
    ):
        latency_s = max(
            float(latency_s),
            0.0,
        )

        self.routing_hop_count += 1

        self.cumulative_transfer_latency_s += (
            latency_s
        )

    # 设置任务到达时间
    def set_arrive_time(self, arrive_time):
        self.arrive_time = arrive_time

    # 记录任务开始时间
    def mark_as_started(self, current_time):
        self.start_time = current_time

    # 记录任务结束时间
    def mark_as_finished(self, current_time):
        self.finish_time = current_time

    # 设置当前任务最终的可归因计算能耗
    def set_compute_energy(self, energy_j):
        self.compute_energy_j = float(energy_j)

    # 累计一次 Edge -> Edge 转发产生的传输能耗
    def add_edge_edge_transfer_energy(self, energy_j):
        energy_j = float(energy_j)
        self.edge_edge_transfer_energy_j += (energy_j)
        self.transfer_energy_j += (energy_j)

    # 累计一次 Edge -> Cloud 转发产生的传输能耗
    def add_edge_cloud_transfer_energy(self, energy_j):
        energy_j = float(energy_j)
        self.edge_cloud_transfer_energy_j += (energy_j)
        self.transfer_energy_j += (energy_j)

    # 返回当前任务已经产生的总可归因能耗
    def get_total_attributable_energy(self):
        return float(self.compute_energy_j + self.transfer_energy_j)

    # 计算任务等待时间：开始时间 - 到达时间
    def get_waiting_time(self):
        if self.start_time is not None:
            return self.start_time - self.arrive_time
        return None

    # 计算周转时间：完成时间 - 到达时间
    def get_turnaround_time(self):
        if self.finish_time is not None:
            return self.finish_time - self.arrive_time
        return None

    # 详细打印此任务的所有属性
    def print_job_info(self):

        start_str = f"{self.start_time:.2f}" if self.start_time is not None else "尚未开始"
        finish_str = f"{self.finish_time:.2f}" if self.finish_time is not None else "尚未完成"
        target_dc_str = self.target_datacenter if self.target_datacenter is not None else "未分配"
        origin_dc_str = self.origin_datacenter if self.origin_datacenter is not None else "未分配"

        print(f"--- 任务 [ID: {self.job_id:^8}] 详情 ---")
        print(f"  初始到达节点: {origin_dc_str}")
        print(f"  当前目标节点: {target_dc_str}")
        print(f"  资源需求: CPU {self.cpu_request} | GPU {self.gpu_request}")
        print(f"  时间线索: 到达 {self.arrive_time:.2f} | 开始 {start_str} | 完成 {finish_str} | 持续时长 {self.duration}")

        # 如果任务已完成，顺便打印计算出的等待和周转时间
        if self.finish_time is not None:
            print(f"  性能指标: 等待时间 {self.get_waiting_time():.2f} | 周转时间 {self.get_turnaround_time():.2f}")

        print(
            f"  能耗账本: "
            f"计算 {self.compute_energy_j:.2f} J | "
            f"传输 {self.transfer_energy_j:.2f} J | "
            f"Edge->Edge {self.edge_edge_transfer_energy_j:.2f} J | "
            f"Edge->Cloud {self.edge_cloud_transfer_energy_j:.2f} J | "
            f"总可归因 {self.get_total_attributable_energy():.2f} J"
        )
        print("-" * 38)

    def __repr__(self):
        return (f"Job(ID:{self.job_id}, Origin:{self.origin_datacenter}, "
                f"DC:{self.target_datacenter}, "
                f"CPU:{self.cpu_request}, GPU:{self.gpu_request})")

class JobList:

    def __init__(self):
        self._queue = []

    def clear(self):
        self._queue.clear()

    def push(self,job):
        self._queue.append(job)

    # def push_front(self, job):
    #     self._queue.insert(0, job)

    def is_empty(self) -> bool:
        return len(self._queue) == 0

    # 取出任务
    def pop(self):
        if not self.is_empty():
            return self._queue.pop(0)
        return None

    # 通过任务id移出队列
    def remove_by_id(self, job_id: str):
        for job in self._queue:
            if str(job.job_id) == str(job_id):
                self._queue.remove(job)
                return job
        return None

    def __len__(self):
        return len(self._queue)

    # 计算队列中的总CPU需求
    def get_total_cpu_demand(self) -> float:
        return sum(job.cpu_request for job in self._queue)

    # 计算队列中的总GPU需求
    def get_total_gpu_demand(self) -> float:
        return sum(job.gpu_request for job in self._queue)

    # 计算队列中的总剩余时长
    def get_total_duration(self) -> float:
        return sum(job.duration for job in self._queue)


    # 打印此队列状态
    def print_status(self):
        print(f"\n当前队列任务数: {len(self._queue)}")
        print(f"  累计 CPU 需求: {self.get_total_cpu_demand()}")
        print(f"  累计 GPU 需求: {self.get_total_gpu_demand()}")

        if self.is_empty():
            print("  -> (队列为空)")
        else:
            for i, job in enumerate(self._queue):
                print(f"  {i + 1}. {job}")
        print("-" * 40)

# 读取并缓存任务数据集，没有必要每轮重新读取磁盘
@lru_cache(maxsize=4)
def load_job_dataset(job_dataset_path: str) -> pd.DataFrame:
    df = pd.read_csv(job_dataset_path)
    return df

def jobs_generate(job_num: int, lambda_rate: float, job_dataset_path: str, wait_assign_jobs_list: list,rng: Optional[np.random.Generator] = None,job_id_prefix: str = "",) -> list:

    job_num = int(job_num)
    lambda_rate = float(lambda_rate)

    # 待分配 jobs 列表
    job_list = wait_assign_jobs_list

    # 任务抽取
    try:
        df = load_job_dataset(
            job_dataset_path
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"任务数据集读取失败: {job_dataset_path}"
        ) from exc

    if rng is None:
        sampled_jobs = (df.sample( n=job_num, replace=True,).reset_index(drop=True))
        inter_arrival_times = np.random.exponential(scale=1.0 / lambda_rate, size=job_num,)
    else:
        sampled_indices = rng.integers(low=0, high=len(df), size=job_num,)
        sampled_jobs = (df.iloc[sampled_indices].reset_index(drop=True))

        # 使用当前 Episode RNG 重新生成泊松到达过程。
        inter_arrival_times = rng.exponential(scale=1.0 / lambda_rate,size=job_num,)

    arrival_times = np.cumsum(inter_arrival_times)

    for sample_index, row in sampled_jobs.iterrows():
        source_job_id = str(row.get( "job_name", f"job_unknown_{sample_index}",))

        # Episode prefix 防止不同 Episode 中 Job ID 重复。
        prefix = (
            f"{job_id_prefix}__"
            if job_id_prefix
            else ""
        )
        job_id = (
            f"{prefix}"
            f"{source_job_id}"
            f"__sample_{sample_index:06d}"
        )
        cpu_req = float(row.get( "cpu_request", 1.0,))
        gpu_req = float(row.get("gpu_request", 0.0,))
        duration = float(row.get("duration", 1.0, ))
        new_job = Job(job_id=job_id, cpu_request=cpu_req, gpu_request=gpu_req, duration=duration,)
        new_job.set_arrive_time(float( arrival_times[sample_index]))
        job_list.append(new_job)

    return job_list


# if __name__ == "__main__":
#     # ================= 1. 测试参数配置 =================
#     NUM_JOBS = 100  # 测试生成 100 个任务
#     TEST_LAMBDA_RATE = 5.0  # 假设单位时间内平均到达 5 个任务
#     Wait_desdin_list = []
#     print(f"🚀 开始测试生成 {NUM_JOBS} 个任务...\n")
#
#     # ================= 2. 调用任务生成函数 =================
#     generated_jobs = jobs_generate(
#         job_num=NUM_JOBS,
#         lambda_rate=conf.LAMBDA_RATE,
#         job_dataset_path=conf.JOB_DATASET_PATH,
#         wait_assign_jobs_list=Wait_desdin_list
#     )
#
#     # ================= 3. 打印与验证测试结果 =================
#     if not generated_jobs:
#         print("⚠️ 任务生成失败，返回了空列表。请优先检查 CSV 数据集路径是否正确！")
#     else:
#         print(f"✅ 成功实例化了 {len(generated_jobs)} 个任务！\n")
#
#         print("=" * 15 + " 任务到达详情 (展示前 10 个) " + "=" * 15)
#         # 遍历前 10 个任务，观察泊松过程模拟的时间序列
#         for i, job in enumerate(generated_jobs[:100]):
#             # 格式化打印，确保对齐。由于不知道你数据集里具体的 job_name 长度，给了 15 个字符的占位
#             print(f"序号: {i + 1:03d} | 任务ID: {str(job.job_id)[:15]:<15} "
#                   f"| CPU: {job.cpu_request:<5.1f} | GPU: {job.gpu_request:<4.1f} "
#                   f"| 耗时: {job.duration:<5.1f} | ⏰ 到达时间: {job.arrive_time:.4f}")
#
#
#
#         print("\n🎉 测试完成：请观察上方【到达时间】是否呈非均匀的递增趋势！")
