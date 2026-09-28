from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

import config as conf


# 能耗模型内部使用的浮点容差。
ENERGY_EPS = 1e-9

# 单个任务在 Edge Host 上的参考动态计算功率
@dataclass(frozen=True)
class EdgeTaskEnergyReference:
    job_id: str
    cpu_request: float
    gpu_request: float
    duration_s: float
    reference_dynamic_power_w: float
    reference_dynamic_energy_j: float

# Cloud 单位资源功率模型标定结果
@dataclass(frozen=True)
class CloudPowerCalibrationResult:
    cpu_power_per_unit_w: float
    gpu_power_per_unit_w: float

    sample_count: int

    # Cloud 预测功率 / Edge 参考功率 的中位数。
    median_power_ratio: float

    # 有多少比例任务的 Cloud/Edge 功率比位于 [0.85, 1.15]。
    within_15_percent_ratio: float

    # Cloud 预测动态功率与 Edge 参考动态功率之间的 RMSE。
    rmse_power_w: float

# 传输能耗模型标定结果
@dataclass(frozen=True)
class TransferEnergyCalibrationResult:
    send_fixed_energy_j: float
    receive_fixed_energy_j: float

    latency_energy_coefficient_w: float

    reference_compute_energy_j: float

    median_edge_edge_latency_s: float
    median_edge_cloud_latency_s: float

# 合法性检查
def _as_nonnegative_finite(value: float,name: str,) -> float:
    """
    将输入转换为 float，并检查其必须是有限非负数。

    能耗模型一旦出现 NaN、Inf 或负资源/负时延，
    应立即暴露问题，而不是继续进入训练。
    """

    value = float(value)

    if not math.isfinite(value):
        raise ValueError(
            f"{name} 必须为有限数值，当前值为 {value}。"
        )

    if value < -ENERGY_EPS:
        raise ValueError(
            f"{name} 不能为负数，当前值为 {value}。"
        )

    # 清理非常小的负浮点误差。
    return max(value, 0.0)

#  判断某个任务是否能够在指定 Edge Host 上执行
def _can_job_run_on_edge_host(job, host,) -> bool:
    job_cpu = _as_nonnegative_finite(job.cpu_request, "job.cpu_request",)
    job_gpu = _as_nonnegative_finite(job.gpu_request, "job.gpu_request",)
    host_cpu = _as_nonnegative_finite(host.cpu_num, "host.cpu_num",)
    host_gpu = _as_nonnegative_finite(host.gpu_capacity_num, "host.gpu_capacity_num",)
    return (
            job_cpu <= host_cpu + ENERGY_EPS
            and job_gpu <= host_gpu + ENERGY_EPS
    )

# 估算任务单独运行在指定 Edge Host 上时造成的动态功率增量
# 用于大概算任务在云上产生的能耗
def estimate_task_dynamic_power_on_edge_host_w(job, host,) -> float:

    job_cpu = float(job.cpu_request)
    job_gpu = float(job.gpu_request)
    host_cpu = float(host.cpu_num)
    host_gpu = float(host.gpu_capacity_num)

    if host_cpu > ENERGY_EPS:
        cpu_utilization = (job_cpu / host_cpu)
    else:
        cpu_utilization = 0.0

    cpu_utilization = float(np.clip(cpu_utilization, 0.0, 1.0,))
    cpu_dynamic_power_w = (float(conf.EDGE_CPU_FULL_POWER_W) - float(conf.EDGE_CPU_IDLE_POWER_W)) * cpu_utilization

    if (job_gpu > ENERGY_EPS and host_gpu > ENERGY_EPS):
        gpu_utilization = (job_gpu / host_gpu)
        gpu_utilization = float(np.clip(gpu_utilization, 0.0, 1.0,))
        gpu_dynamic_power_w = (float(conf.EDGE_GPU_FULL_POWER_W) - float(conf.EDGE_GPU_IDLE_POWER_W)) * math.log2( 1.0 + gpu_utilization)

    else:
        gpu_dynamic_power_w = 0.0

    dynamic_power_w = (cpu_dynamic_power_w + gpu_dynamic_power_w)

    if not math.isfinite(dynamic_power_w):
        raise RuntimeError(
            f"任务 {job.job_id} 在 Host {host.host_id} "
            "上的动态功率计算得到非有限值。"
        )

    return max(float(dynamic_power_w), 0.0,)

# 为任务集建立 Edge 动态计算能耗参考样本
def build_edge_task_energy_references(jobs: Iterable,edge_hosts: Iterable,) -> List[EdgeTaskEnergyReference]:
    jobs = list(jobs)
    edge_hosts = list(edge_hosts)
    references: List[EdgeTaskEnergyReference] = []

    for job in jobs:

        duration_s = _as_nonnegative_finite(
            job.duration,
            f"job[{job.job_id}].duration",
        )

        if duration_s <= ENERGY_EPS:
            # duration=0 的样本对能耗标定没有实际意义。
            continue

        feasible_dynamic_powers = []

        for host in edge_hosts:

            if not _can_job_run_on_edge_host(
                    job=job,
                    host=host,
            ):
                continue

            dynamic_power_w = (
                estimate_task_dynamic_power_on_edge_host_w(
                    job=job,
                    host=host,
                )
            )

            feasible_dynamic_powers.append(
                dynamic_power_w
            )

        # 没有任何 Edge Host 可以执行：
        # 不能建立 Edge reference，因此不参与 Cloud 参数拟合。
        if not feasible_dynamic_powers:
            continue

        reference_dynamic_power_w = float(
            np.median(
                np.asarray(
                    feasible_dynamic_powers,
                    dtype=np.float64,
                )
            )
        )

        reference_dynamic_energy_j = (
                reference_dynamic_power_w
                * duration_s
        )

        references.append(
            EdgeTaskEnergyReference(
                job_id=str(job.job_id),

                cpu_request=float(
                    job.cpu_request
                ),
                gpu_request=float(
                    job.gpu_request
                ),
                duration_s=float(
                    duration_s
                ),

                reference_dynamic_power_w=float(
                    reference_dynamic_power_w
                ),

                reference_dynamic_energy_j=float(
                    reference_dynamic_energy_j
                ),
            )
        )

    if not references:
        raise RuntimeError(
            "无法建立任何 Edge 任务能耗参考样本。"
            "请检查任务资源需求和 Edge Host 容量。"
        )

    return references

def _fit_nonnegative_two_feature_least_squares(cpu_requests: np.ndarray,gpu_requests: np.ndarray,target_power_w: np.ndarray,) -> tuple[float, float]:
    """
    拟合：

        target_power
            ≈ k_cpu * cpu_request
            + k_gpu * gpu_request

    并约束：

        k_cpu >= 0
        k_gpu >= 0

    当前只有两个自变量，所以无需额外引入 SciPy NNLS 依赖。

    做法：
        1. 求普通最小二乘内部解；
        2. 检查 CPU-only 边界；
        3. 检查 GPU-only 边界；
        4. 检查 (0,0)；
        5. 从所有合法候选中选择 SSE 最小者。

    对二维凸 NNLS 问题，上述候选覆盖内部最优点和边界最优点。
    """

    cpu_requests = np.asarray(
        cpu_requests,
        dtype=np.float64,
    )

    gpu_requests = np.asarray(
        gpu_requests,
        dtype=np.float64,
    )

    target_power_w = np.asarray(
        target_power_w,
        dtype=np.float64,
    )

    design_matrix = np.column_stack(
        (
            cpu_requests,
            gpu_requests,
        )
    )

    candidates = []

    # --------------------------------------------------------------
    # 1. 无约束最小二乘解
    # --------------------------------------------------------------
    unconstrained_solution, _, _, _ = (
        np.linalg.lstsq(
            design_matrix,
            target_power_w,
            rcond=None,
        )
    )

    k_cpu_unconstrained = float(
        unconstrained_solution[0]
    )

    k_gpu_unconstrained = float(
        unconstrained_solution[1]
    )

    if (
            k_cpu_unconstrained >= -ENERGY_EPS
            and k_gpu_unconstrained >= -ENERGY_EPS
    ):
        candidates.append(
            (
                max(k_cpu_unconstrained, 0.0),
                max(k_gpu_unconstrained, 0.0),
            )
        )

    # --------------------------------------------------------------
    # 2. CPU-only 边界
    # --------------------------------------------------------------
    cpu_norm = float(
        np.dot(
            cpu_requests,
            cpu_requests,
        )
    )

    if cpu_norm > ENERGY_EPS:
        k_cpu_only = max(
            float(
                np.dot(
                    cpu_requests,
                    target_power_w,
                )
                / cpu_norm
            ),
            0.0,
        )

        candidates.append(
            (
                k_cpu_only,
                0.0,
            )
        )

    # --------------------------------------------------------------
    # 3. GPU-only 边界
    # --------------------------------------------------------------
    gpu_norm = float(
        np.dot(
            gpu_requests,
            gpu_requests,
        )
    )

    if gpu_norm > ENERGY_EPS:
        k_gpu_only = max(
            float(
                np.dot(
                    gpu_requests,
                    target_power_w,
                )
                / gpu_norm
            ),
            0.0,
        )

        candidates.append(
            (
                0.0,
                k_gpu_only,
            )
        )

    # 原点也是合法候选。
    candidates.append(
        (
            0.0,
            0.0,
        )
    )

    best_solution = None
    best_sse = float("inf")

    for k_cpu, k_gpu in candidates:

        predicted_power = (
            k_cpu * cpu_requests
            + k_gpu * gpu_requests
        )

        residual = (
            predicted_power
            - target_power_w
        )

        sse = float(
            np.dot(
                residual,
                residual,
            )
        )

        if sse < best_sse:
            best_sse = sse
            best_solution = (
                k_cpu,
                k_gpu,
            )

    if best_solution is None:
        raise RuntimeError(
            "Cloud 单位资源功率参数拟合失败。"
        )

    return best_solution

def calibrate_cloud_unit_resource_power(
        references: Sequence[
            EdgeTaskEnergyReference
        ],
) -> CloudPowerCalibrationResult:
    """
    根据 Edge 任务参考动态功率，
    自动拟合 Cloud 单位 CPU/GPU 功率参数。

    Cloud 功率模型：

        P_cloud
            = k_cpu * used_cpu
            + k_gpu * used_gpu

    Cloud 不使用：
        used_cpu / 999999
        used_gpu / 999999

    从而避免当前 Cloud 逻辑无限容量导致 utilization
    几乎为 0、动态能耗被严重低估的问题。
    """

    references = list(references)

    if not references:
        raise ValueError(
            "Cloud 功率标定失败：reference 样本为空。"
        )

    cpu_requests = np.asarray(
        [
            ref.cpu_request
            for ref in references
        ],
        dtype=np.float64,
    )

    gpu_requests = np.asarray(
        [
            ref.gpu_request
            for ref in references
        ],
        dtype=np.float64,
    )

    target_power_w = np.asarray(
        [
            ref.reference_dynamic_power_w
            for ref in references
        ],
        dtype=np.float64,
    )

    k_cpu, k_gpu = (
        _fit_nonnegative_two_feature_least_squares(
            cpu_requests=cpu_requests,
            gpu_requests=gpu_requests,
            target_power_w=target_power_w,
        )
    )

    predicted_power_w = (
        k_cpu * cpu_requests
        + k_gpu * gpu_requests
    )

    residual = (
        predicted_power_w
        - target_power_w
    )

    rmse_power_w = float(
        np.sqrt(
            np.mean(
                residual ** 2
            )
        )
    )

    # 只对参考动态功率 > 0 的样本计算比值。
    positive_mask = (
        target_power_w > ENERGY_EPS
    )

    if np.any(positive_mask):

        power_ratios = (
            predicted_power_w[positive_mask]
            / target_power_w[positive_mask]
        )

        median_power_ratio = float(
            np.median(
                power_ratios
            )
        )

        within_15_percent_ratio = float(
            np.mean(
                (
                    power_ratios >= 0.85
                )
                &
                (
                    power_ratios <= 1.15
                )
            )
        )

    else:
        median_power_ratio = 1.0
        within_15_percent_ratio = 1.0

    return CloudPowerCalibrationResult(
        cpu_power_per_unit_w=float(
            k_cpu
        ),

        gpu_power_per_unit_w=float(
            k_gpu
        ),

        sample_count=int(
            len(references)
        ),

        median_power_ratio=float(
            median_power_ratio
        ),

        within_15_percent_ratio=float(
            within_15_percent_ratio
        ),

        rmse_power_w=float(
            rmse_power_w
        ),
    )


def assert_cloud_calibration_quality(
        calibration: CloudPowerCalibrationResult,
) -> None:
    """
    检查 Cloud 与 Edge 的同任务动态计算功率是否足够接近。

    第一版验收条件：

        median(Cloud / Edge) ∈ [0.95, 1.05]

    并且至少 90% 的任务：

        Cloud / Edge ∈ [0.85, 1.15]

    如果不满足，不应直接通过调整 Reward 权重掩盖，
    而应该先重新检查 Cloud 功率模型或任务/Host 数据。
    """

    median_ratio = float(
        calibration.median_power_ratio
    )

    within_ratio = float(
        calibration.within_15_percent_ratio
    )

    if not (
            0.95
            <= median_ratio
            <= 1.05
    ):
        raise RuntimeError(
            "Cloud 功率标定未达到要求："
            f"median Cloud/Edge ratio="
            f"{median_ratio:.4f}，"
            "要求位于 [0.95, 1.05]。"
        )

    if within_ratio < 0.90:
        raise RuntimeError(
            "Cloud 功率标定未达到要求："
            f"仅 {within_ratio:.2%} 的任务位于 "
            "Cloud/Edge=[0.85, 1.15]，"
            "要求至少 90%。"
        )


def calculate_cloud_attributable_power_w(
        used_cpu: float,
        used_gpu: float,
        calibration: CloudPowerCalibrationResult,
) -> float:
    """
    根据 Cloud 当前实际使用的绝对 CPU/GPU 资源量，
    返回当前 workload-attributable Cloud 功率。

    P_cloud
        = k_cpu * used_cpu
        + k_gpu * used_gpu

    注意：
        这里没有 Cloud idle power。

    原因：
        当前 Cloud 在项目中表示逻辑无限资源池，
        而不是一台真实服务器；
        因此这里只统计当前模拟 workload
        可以归因的计算功率。
    """

    used_cpu = _as_nonnegative_finite(
        used_cpu,
        "cloud.used_cpu",
    )

    used_gpu = _as_nonnegative_finite(
        used_gpu,
        "cloud.used_gpu",
    )

    power_w = (
        calibration.cpu_power_per_unit_w
        * used_cpu
        +
        calibration.gpu_power_per_unit_w
        * used_gpu
    )

    if not math.isfinite(power_w):
        raise RuntimeError(
            "Cloud 功率计算得到非有限值。"
        )

    return max(
        float(power_w),
        0.0,
    )


# ======================================================================
# Transmission Energy 参数标定
# ======================================================================

def calibrate_transfer_energy_model(
        references: Sequence[
            EdgeTaskEnergyReference
        ],
        edge_edge_latencies_s: Sequence[float],
        edge_cloud_latencies_s: Sequence[float],
) -> TransferEnergyCalibrationResult:
    """
    自动标定传输能耗模型：

        E_trans
            = E_send_fixed
            + E_recv_fixed
            + kappa_trans * latency

    根据 config 中的目标比例：

        Edge -> Edge
            ≈ EDGE_EDGE_TRANSFER_ENERGY_RATIO
              * typical_compute_energy

        Edge -> Cloud
            ≈ EDGE_CLOUD_TRANSFER_ENERGY_RATIO
              * typical_compute_energy

    其中 typical_compute_energy 使用
    Edge 任务参考动态计算能耗的中位数。
    """

    references = list(references)

    if not references:
        raise ValueError(
            "传输能耗标定失败：Edge reference 为空。"
        )

    edge_edge_latencies_s = np.asarray(
        edge_edge_latencies_s,
        dtype=np.float64,
    )

    edge_cloud_latencies_s = np.asarray(
        edge_cloud_latencies_s,
        dtype=np.float64,
    )

    if edge_edge_latencies_s.size == 0:
        raise ValueError(
            "传输能耗标定失败："
            "Edge->Edge latency 样本为空。"
        )

    if edge_cloud_latencies_s.size == 0:
        raise ValueError(
            "传输能耗标定失败："
            "Edge->Cloud latency 样本为空。"
        )

    if (
            not np.all(
                np.isfinite(
                    edge_edge_latencies_s
                )
            )
            or np.any(
                edge_edge_latencies_s < 0.0
            )
    ):
        raise ValueError(
            "Edge->Edge latency 必须全部为有限非负秒数。"
        )

    if (
            not np.all(
                np.isfinite(
                    edge_cloud_latencies_s
                )
            )
            or np.any(
                edge_cloud_latencies_s < 0.0
            )
    ):
        raise ValueError(
            "Edge->Cloud latency 必须全部为有限非负秒数。"
        )

    positive_compute_energies = [
        ref.reference_dynamic_energy_j
        for ref in references
        if ref.reference_dynamic_energy_j > ENERGY_EPS
    ]

    if not positive_compute_energies:
        raise RuntimeError(
            "不存在正的 Edge 参考计算能耗，"
            "无法标定传输能耗。"
        )

    reference_compute_energy_j = float(
        np.median(
            np.asarray(
                positive_compute_energies,
                dtype=np.float64,
            )
        )
    )

    median_edge_edge_latency_s = float(
        np.median(
            edge_edge_latencies_s
        )
    )

    median_edge_cloud_latency_s = float(
        np.median(
            edge_cloud_latencies_s
        )
    )

    edge_edge_ratio = float(
        conf.EDGE_EDGE_TRANSFER_ENERGY_RATIO
    )

    edge_cloud_ratio = float(
        conf.EDGE_CLOUD_TRANSFER_ENERGY_RATIO
    )

    if not (
            0.0
            <= edge_edge_ratio
            < edge_cloud_ratio
    ):
        raise ValueError(
            "传输能耗比例配置非法：必须满足 "
            "0 <= EDGE_EDGE_TRANSFER_ENERGY_RATIO "
            "< EDGE_CLOUD_TRANSFER_ENERGY_RATIO。"
        )

    # 要通过 latency 实现 Cloud 比 Edge 传输能耗更大，
    # 典型 Cloud latency 必须大于典型 Edge latency。
    latency_difference = (
        median_edge_cloud_latency_s
        - median_edge_edge_latency_s
    )

    if latency_difference <= ENERGY_EPS:
        raise RuntimeError(
            "传输能耗无法按当前目标标定："
            "Edge->Cloud 的中位时延必须大于 "
            "Edge->Edge 的中位时延。"
        )

    # --------------------------------------------------------------
    # 根据两组标定条件反求 kappa：
    #
    # E_fixed + kappa * tau_EE
    #     = q_EE * E_compute
    #
    # E_fixed + kappa * tau_EC
    #     = q_EC * E_compute
    # --------------------------------------------------------------
    latency_energy_coefficient_w = (
        (
            edge_cloud_ratio
            - edge_edge_ratio
        )
        * reference_compute_energy_j
        / latency_difference
    )

    total_fixed_energy_j = (
        edge_edge_ratio
        * reference_compute_energy_j
        -
        latency_energy_coefficient_w
        * median_edge_edge_latency_s
    )

    # 固定发送/接收能耗不能是明显负值。
    if total_fixed_energy_j < -ENERGY_EPS:
        raise RuntimeError(
            "根据当前传输目标比例和网络时延分布，"
            "反推出的固定传输能耗为负数。"
            "请调整 Edge/Cloud 传输能耗目标比例。"
        )

    total_fixed_energy_j = max(
        float(total_fixed_energy_j),
        0.0,
    )

    # 当前假设发送端和接收端固定能耗相同，
    # 因此各占总固定 endpoint energy 的一半。
    send_fixed_energy_j = (
        total_fixed_energy_j / 2.0
    )

    receive_fixed_energy_j = (
        total_fixed_energy_j / 2.0
    )

    return TransferEnergyCalibrationResult(
        send_fixed_energy_j=float(
            send_fixed_energy_j
        ),

        receive_fixed_energy_j=float(
            receive_fixed_energy_j
        ),

        latency_energy_coefficient_w=float(
            latency_energy_coefficient_w
        ),

        reference_compute_energy_j=float(
            reference_compute_energy_j
        ),

        median_edge_edge_latency_s=float(
            median_edge_edge_latency_s
        ),

        median_edge_cloud_latency_s=float(
            median_edge_cloud_latency_s
        ),
    )


def calculate_transfer_energy_j(
        latency_s: float,
        calibration: TransferEnergyCalibrationResult,
) -> float:
    """
    计算一次 Edge->Edge 或 Edge->Cloud 调度行为
    产生的传输能耗。

    E_trans
        = E_send_fixed
        + E_recv_fixed
        + kappa_trans * latency

    latency 的单位必须为秒（s）。

    返回单位：
        J
    """

    latency_s = _as_nonnegative_finite(
        latency_s,
        "transfer_latency_s",
    )

    transfer_energy_j = (
        calibration.send_fixed_energy_j
        +
        calibration.receive_fixed_energy_j
        +
        calibration.latency_energy_coefficient_w
        * latency_s
    )

    if not math.isfinite(
            transfer_energy_j
    ):
        raise RuntimeError(
            "传输能耗计算得到非有限值。"
        )

    return max(
        float(transfer_energy_j),
        0.0,
    )




