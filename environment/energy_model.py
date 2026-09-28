from __future__ import annotations
import math
import config as conf

# 任务在指定 Edge Host 上执行时功率
def estimate_task_dynamic_power_on_edge_host_w(job, host,) -> float:

    job_cpu = float(job.cpu_request)
    job_gpu = float(job.gpu_request)

    host_cpu = float(host.cpu_num)
    host_gpu = float(host.gpu_capacity_num)


    cpu_utilization = job_cpu / host_cpu
    cpu_utilization = min(max(cpu_utilization, 0.0), 1.0,)
    cpu_dynamic_power_w = (float(conf.EDGE_CPU_FULL_POWER_W) - float(conf.EDGE_CPU_IDLE_POWER_W)) * cpu_utilization

    gpu_utilization = (job_gpu / host_gpu)
    gpu_utilization = min(max(gpu_utilization, 0.0), 1.0,)
    gpu_dynamic_power_w = (float(conf.EDGE_GPU_FULL_POWER_W) - float(conf.EDGE_GPU_IDLE_POWER_W)) * math.log2(1.0 + gpu_utilization)

    return float(cpu_dynamic_power_w + gpu_dynamic_power_w)

# 计算任务在指定 Edge Host 上执行能耗
def calculate_edge_task_attributable_compute_energy_j(job, host,) -> float:

    dynamic_power_w = (estimate_task_dynamic_power_on_edge_host_w( job=job, host=host,))
    duration_s = float(job.duration)

    return float(dynamic_power_w * duration_s)

# Cloud计算功率
def calculate_cloud_attributable_power_w(used_cpu: float, used_gpu: float,) -> float:
    return float(
        float(conf.CLOUD_CPU_POWER_PER_UNIT_W) * float(used_cpu) +
        float(conf.CLOUD_GPU_POWER_PER_UNIT_W) * float(used_gpu)
    )

# 任务在 Cloud 上运行产生的计算能耗
def calculate_cloud_task_attributable_compute_energy_j(job,) -> float:
    task_power_w = calculate_cloud_attributable_power_w(used_cpu=float(job.cpu_request), used_gpu=float(job.gpu_request),)
    duration_s = float(job.duration)
    return float(task_power_w * duration_s)

# 传输能耗
def calculate_transfer_energy_j(latency_s: float,) -> float:
    send_energy_j = float(conf.TRANSFER_SEND_FIXED_ENERGY_J)
    receive_energy_j = float(conf.TRANSFER_RECEIVE_FIXED_ENERGY_J)
    latency_energy_j = (float(conf.TRANSFER_LATENCY_ENERGY_COEFFICIENT_W) * float(latency_s))
    return float(send_energy_j + receive_energy_j + latency_energy_j)

# 计算host当前功率
def calculate_edge_host_power_components_w(host,) -> dict:

    host.calculate_load()
    cpu_load = float(host.cpu_load)
    gpu_load = float(host.gpu_load)

    cpu_idle_power_w = float(conf.EDGE_CPU_IDLE_POWER_W)
    cpu_full_power_w = float(conf.EDGE_CPU_FULL_POWER_W)
    gpu_idle_power_w = float(conf.EDGE_GPU_IDLE_POWER_W)
    gpu_full_power_w = float(conf.EDGE_GPU_FULL_POWER_W)

    idle_power_w = cpu_idle_power_w + gpu_idle_power_w
    cpu_dynamic_power_w = (cpu_full_power_w - cpu_idle_power_w) * cpu_load
    gpu_dynamic_power_w = (gpu_full_power_w - gpu_idle_power_w) * math.log2(1.0 + gpu_load)
    total_power_w = (idle_power_w + cpu_dynamic_power_w + gpu_dynamic_power_w)

    return {
        "idle_power_w": float(idle_power_w),
        "cpu_dynamic_power_w": float(cpu_dynamic_power_w),
        "gpu_dynamic_power_w": float(gpu_dynamic_power_w),
        "total_power_w": float(total_power_w),
    }

# 返回host总功率
def calculate_edge_host_total_power_w(host,) -> float:
    power_components = (calculate_edge_host_power_components_w(host=host,))

    return float(power_components["total_power_w"])















