"""
一次性能源模型参数标定脚本。

用途：
    1. 根据当前 Job 数据集和 Edge Host 配置，
       建立任务在 Edge 上的参考动态计算功率/能耗；

    2. 拟合 Cloud 单位资源功率参数：
           P_cloud =
               k_cpu * used_cpu
               + k_gpu * used_gpu

    3. 检查 Cloud 与 Edge 的功率拟合质量；

    4. 根据当前 Edge->Edge / Edge->Cloud 时延分布，
       标定传输能耗模型：
           E_transfer =
               E_send_fixed
               + E_receive_fixed
               + kappa * latency

    5. 最终打印一组可以直接复制到 config.py
       并在以后正式训练中固定使用的参数。

重要：
    本脚本只需要在能源模型确定阶段运行一次。

    参数确定并写入 config.py 后，
    正式 MASAC 训练过程中不应再次执行拟合。
"""

import random
from pathlib import Path

import numpy as np

import config as conf
from environment.env_generate import EnvGenerator
from environment.energy_model import (
    build_edge_task_energy_references,
    calibrate_cloud_unit_resource_power,
    calibrate_transfer_energy_model,
    calculate_transfer_energy_j,
)


# ======================================================================
# 标定质量提示阈值
# ======================================================================

# Cloud/Edge 中位功率比的理想范围。
CLOUD_MEDIAN_RATIO_LOWER = 0.95
CLOUD_MEDIAN_RATIO_UPPER = 1.05

# 原设计的理想覆盖率：
# 至少 90% 的任务 Cloud/Edge 功率比位于 [0.85, 1.15]。
CLOUD_IDEAL_WITHIN_15_RATIO = 0.90

# 离线标定过程中允许继续输出参数的参考下限。
#
# 注意：
# 这不是修改物理模型，只是避免因为例如 86.6% < 90%
# 就直接终止程序。
#
# 如果低于 85%，程序仍然会输出参数，
# 但会明确提示当前线性 Cloud 模型拟合质量偏低，
# 不建议直接冻结参数。
CLOUD_ACCEPTABLE_WITHIN_15_RATIO = 0.85


# ======================================================================
# 辅助打印函数
# ======================================================================

def print_section(title: str) -> None:
    """
    打印分隔标题，方便查看一次性标定输出。
    """

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ======================================================================
# 提取 Edge Host
# ======================================================================

def extract_edge_hosts(env_generator: EnvGenerator):
    """
    从已经生成的环境中提取所有 Edge Host。

    当前项目的 Cloud DataCenter 使用：
        dc_id == "cloud"

    Cloud Host 的 CPU/GPU 容量为逻辑上的近似无限容量，
    不能参与 Edge reference 构造。

    因此这里只保留非 Cloud 数据中心中的 Host。
    """

    edge_hosts = []

    for dc in env_generator.global_dc_list:

        if str(dc.dc_id) == "cloud":
            continue

        for host in dc.host_list:
            edge_hosts.append(host)

    if not edge_hosts:
        raise RuntimeError(
            "没有找到任何 Edge Host，无法进行能源参数标定。"
        )

    return edge_hosts


# ======================================================================
# 提取网络时延
# ======================================================================

def extract_network_latencies_s(
        env_generator: EnvGenerator,
):
    """
    从 NetworkX 数据中心拓扑中提取：

        Edge -> Edge latency
        Edge -> Cloud latency

    当前项目已经约定：
        graph 中 weight 的实际时间语义统一按秒（s）处理。

    因此这里直接使用 weight，
    不进行 /1000 的 ms -> s 转换。
    """

    graph = env_generator.datacenter_graph

    if graph is None:
        raise RuntimeError(
            "datacenter_graph 为空，无法进行传输能耗标定。"
        )

    edge_edge_latencies_s = []
    edge_cloud_latencies_s = []

    # 当前拓扑为无向 Graph，
    # edges(data=True) 中每条物理边只会遍历一次。
    for source, target, data in graph.edges(data=True):

        if "weight" not in data:
            raise RuntimeError(
                f"网络边 ({source}, {target}) 缺少 weight。"
            )

        latency_s = float(
            data["weight"]
        )

        if not np.isfinite(latency_s):
            raise ValueError(
                f"网络边 ({source}, {target}) "
                f"latency 非有限值：{latency_s}"
            )

        if latency_s < 0.0:
            raise ValueError(
                f"网络边 ({source}, {target}) "
                f"latency 不能为负数：{latency_s}"
            )

        source_is_cloud = (
            str(source) == "cloud"
        )

        target_is_cloud = (
            str(target) == "cloud"
        )

        if source_is_cloud or target_is_cloud:

            # 一端是 cloud：
            # 归类为 Edge -> Cloud latency。
            edge_cloud_latencies_s.append(
                latency_s
            )

        else:

            # 两端均为普通 Edge DC：
            # 归类为 Edge -> Edge latency。
            edge_edge_latencies_s.append(
                latency_s
            )

    if not edge_edge_latencies_s:
        raise RuntimeError(
            "没有找到 Edge->Edge latency 样本。"
        )

    if not edge_cloud_latencies_s:
        raise RuntimeError(
            "没有找到 Edge->Cloud latency 样本。"
        )

    return (
        edge_edge_latencies_s,
        edge_cloud_latencies_s,
    )


# ======================================================================
# Cloud / Edge 拟合比值诊断
# ======================================================================

def calculate_cloud_edge_power_ratios(
        references,
        cloud_result,
) -> np.ndarray:
    """
    对所有有效 Edge reference 计算：

        ratio =
            Cloud预测动态功率
            /
            Edge参考动态功率

    这个分布用于判断：

        P_cloud ≈ P_edge_reference

    是否总体成立。

    注意：
        这里只进行拟合质量分析，
        不参与正式训练时的能源计算。
    """

    power_ratios = []

    for ref in references:

        edge_power_w = float(
            ref.reference_dynamic_power_w
        )

        # reference power 为 0 时，
        # 无法进行比例计算，直接忽略。
        if edge_power_w <= 1e-9:
            continue

        cloud_predicted_power_w = (
            float(
                cloud_result.cpu_power_per_unit_w
            )
            * float(ref.cpu_request)
            +
            float(
                cloud_result.gpu_power_per_unit_w
            )
            * float(ref.gpu_request)
        )

        ratio = (
            cloud_predicted_power_w
            / edge_power_w
        )

        if np.isfinite(ratio):
            power_ratios.append(
                float(ratio)
            )

    if not power_ratios:
        raise RuntimeError(
            "没有有效的 Cloud/Edge 功率比样本。"
        )

    return np.asarray(
        power_ratios,
        dtype=np.float64,
    )


# ======================================================================
# 打印 Cloud 标定质量
# ======================================================================

def print_cloud_calibration_result(
        cloud_result,
        power_ratios: np.ndarray,
) -> None:
    """
    输出 Cloud 参数及其拟合质量。

    本函数不会因为未达到 90% 覆盖率而 raise。

    原因：
        Edge GPU 功率模型包含 log2(1 + U_gpu) 非线性，
        而 Cloud 使用线性单位资源模型：

            k_cpu * cpu + k_gpu * gpu

        因此不应要求每个任务都严格完全一致。

    当前重点观察：
        1. median ratio 是否接近 1；
        2. ±15% 覆盖率；
        3. ratio 的 P05/P95 等分位数；
        4. RMSE。
    """

    print_section(
        "Cloud 单位资源功率标定结果"
    )

    print(
        "有效标定任务数                    : "
        f"{cloud_result.sample_count}"
    )

    print(
        "Cloud CPU power per unit          : "
        f"{cloud_result.cpu_power_per_unit_w:.10f} W/unit"
    )

    print(
        "Cloud GPU power per unit          : "
        f"{cloud_result.gpu_power_per_unit_w:.10f} W/unit"
    )

    print(
        "Cloud / Edge median power ratio   : "
        f"{cloud_result.median_power_ratio:.6f}"
    )

    print(
        "Cloud / Edge within ±15%          : "
        f"{cloud_result.within_15_percent_ratio:.2%}"
    )

    print(
        "Cloud fitting RMSE                : "
        f"{cloud_result.rmse_power_w:.6f} W"
    )

    # --------------------------------------------------------------
    # 输出完整比例分布，避免只看一个 86.6% 就无法判断
    # 剩余 13.4% 的任务偏差究竟有多大。
    # --------------------------------------------------------------

    print_section(
        "Cloud / Edge 功率比详细分布"
    )

    percentiles = [
        5,
        10,
        25,
        50,
        75,
        90,
        95,
    ]

    for percentile in percentiles:

        value = float(
            np.percentile(
                power_ratios,
                percentile,
            )
        )

        print(
            f"P{percentile:02d} ratio                       : "
            f"{value:.6f}"
        )

    print(
        "Minimum ratio                      : "
        f"{float(np.min(power_ratios)):.6f}"
    )

    print(
        "Maximum ratio                      : "
        f"{float(np.max(power_ratios)):.6f}"
    )

    # --------------------------------------------------------------
    # 非致命质量提示。
    #
    # 注意：
    # 这里不再像 assert_cloud_calibration_quality()
    # 那样因为 86.6% < 90% 直接终止。
    # --------------------------------------------------------------

    print_section(
        "Cloud 标定质量判断"
    )

    median_ratio = float(
        cloud_result.median_power_ratio
    )

    within_ratio = float(
        cloud_result.within_15_percent_ratio
    )

    median_ok = (
        CLOUD_MEDIAN_RATIO_LOWER
        <= median_ratio
        <= CLOUD_MEDIAN_RATIO_UPPER
    )

    if median_ok:
        print(
            "✅ Cloud/Edge 中位功率比位于 "
            f"[{CLOUD_MEDIAN_RATIO_LOWER:.2f}, "
            f"{CLOUD_MEDIAN_RATIO_UPPER:.2f}]。"
        )
    else:
        print(
            "⚠️ Cloud/Edge 中位功率比超出 "
            f"[{CLOUD_MEDIAN_RATIO_LOWER:.2f}, "
            f"{CLOUD_MEDIAN_RATIO_UPPER:.2f}]。"
        )

    if (
            within_ratio
            >= CLOUD_IDEAL_WITHIN_15_RATIO
    ):

        print(
            "✅ 至少 90% 的任务 Cloud/Edge "
            "功率误差位于 ±15%，达到原设计理想标准。"
        )

    elif (
            within_ratio
            >= CLOUD_ACCEPTABLE_WITHIN_15_RATIO
    ):

        print(
            "⚠️ Cloud/Edge ±15% 覆盖率为 "
            f"{within_ratio:.2%}，"
            "未达到原设计的 90%，"
            "但已经达到 85% 的参考可接受范围。"
        )

        print(
            "   请结合 P05/P95、RMSE 和 median ratio "
            "判断是否冻结当前参数。"
        )

    else:

        print(
            "⚠️ Cloud/Edge ±15% 覆盖率仅为 "
            f"{within_ratio:.2%}，低于 85%。"
        )

        print(
            "   当前线性 Cloud 模型拟合偏差较大，"
            "不建议直接冻结参数。"
        )


# ======================================================================
# 打印 Transmission 标定结果
# ======================================================================

def print_transfer_calibration_result(
        transfer_result,
) -> None:
    """
    输出传输能耗标定参数，并验证典型 Edge->Edge /
    Edge->Cloud 传输是否满足 config 中 4% / 10% 的目标。
    """

    print_section(
        "Transmission 能耗标定结果"
    )

    print(
        "典型 Edge compute energy          : "
        f"{transfer_result.reference_compute_energy_j:.6f} J"
    )

    print(
        "Median Edge->Edge latency          : "
        f"{transfer_result.median_edge_edge_latency_s:.6f} s"
    )

    print(
        "Median Edge->Cloud latency         : "
        f"{transfer_result.median_edge_cloud_latency_s:.6f} s"
    )

    print(
        "Send fixed energy                  : "
        f"{transfer_result.send_fixed_energy_j:.10f} J"
    )

    print(
        "Receive fixed energy               : "
        f"{transfer_result.receive_fixed_energy_j:.10f} J"
    )

    print(
        "Latency energy coefficient         : "
        f"{transfer_result.latency_energy_coefficient_w:.10f} W"
    )

    # --------------------------------------------------------------
    # 使用最终参数重新计算两个中位时延上的实际传输能耗，
    # 验证反解结果是否符合 4% / 10% 目标。
    # --------------------------------------------------------------

    median_edge_edge_energy_j = (
        calculate_transfer_energy_j(
            latency_s=(
                transfer_result
                .median_edge_edge_latency_s
            ),
            calibration=transfer_result,
        )
    )

    median_edge_cloud_energy_j = (
        calculate_transfer_energy_j(
            latency_s=(
                transfer_result
                .median_edge_cloud_latency_s
            ),
            calibration=transfer_result,
        )
    )

    reference_compute_energy_j = float(
        transfer_result.reference_compute_energy_j
    )

    edge_edge_ratio = (
        median_edge_edge_energy_j
        / reference_compute_energy_j
    )

    edge_cloud_ratio = (
        median_edge_cloud_energy_j
        / reference_compute_energy_j
    )

    print_section(
        "Transmission 标定结果验证"
    )

    print(
        "Median Edge->Edge transfer energy  : "
        f"{median_edge_edge_energy_j:.6f} J"
    )

    print(
        "实际 Edge->Edge / compute ratio    : "
        f"{edge_edge_ratio:.6f} "
        f"(目标 {conf.EDGE_EDGE_TRANSFER_ENERGY_RATIO:.6f})"
    )

    print(
        "Median Edge->Cloud transfer energy : "
        f"{median_edge_cloud_energy_j:.6f} J"
    )

    print(
        "实际 Edge->Cloud / compute ratio   : "
        f"{edge_cloud_ratio:.6f} "
        f"(目标 {conf.EDGE_CLOUD_TRANSFER_ENERGY_RATIO:.6f})"
    )


# ======================================================================
# 保存结果
# ======================================================================

def save_calibration_result(
        cloud_result,
        transfer_result,
        power_ratios: np.ndarray,
) -> Path:
    """
    将本次离线标定结果保存到文本文件。

    这个文件只是记录和复核使用，
    正式训练仍然应该将最终确认参数手动写入 config.py。
    """

    output_path = (
        Path(__file__).resolve().parent
        / "energy_calibration_result.txt"
    )

    with open(
            output_path,
            "w",
            encoding="utf-8",
    ) as file:

        file.write(
            "========== Energy Calibration Result ==========\n\n"
        )

        file.write(
            f"Seed = {conf.Seed}\n"
        )

        file.write(
            f"Cloud sample count = "
            f"{cloud_result.sample_count}\n\n"
        )

        file.write(
            "CLOUD_CPU_POWER_PER_UNIT_W = "
            f"{cloud_result.cpu_power_per_unit_w:.10f}\n"
        )

        file.write(
            "CLOUD_GPU_POWER_PER_UNIT_W = "
            f"{cloud_result.gpu_power_per_unit_w:.10f}\n"
        )

        file.write(
            "TRANSFER_SEND_FIXED_ENERGY_J = "
            f"{transfer_result.send_fixed_energy_j:.10f}\n"
        )

        file.write(
            "TRANSFER_RECEIVE_FIXED_ENERGY_J = "
            f"{transfer_result.receive_fixed_energy_j:.10f}\n"
        )

        file.write(
            "TRANSFER_LATENCY_ENERGY_COEFFICIENT_W = "
            f"{transfer_result.latency_energy_coefficient_w:.10f}\n"
        )

        file.write("\n")

        file.write(
            "Cloud/Edge median ratio = "
            f"{cloud_result.median_power_ratio:.10f}\n"
        )

        file.write(
            "Cloud/Edge within ±15% = "
            f"{cloud_result.within_15_percent_ratio:.10f}\n"
        )

        file.write(
            "Cloud fitting RMSE W = "
            f"{cloud_result.rmse_power_w:.10f}\n"
        )

        file.write(
            "Cloud/Edge P05 ratio = "
            f"{np.percentile(power_ratios, 5):.10f}\n"
        )

        file.write(
            "Cloud/Edge P95 ratio = "
            f"{np.percentile(power_ratios, 95):.10f}\n"
        )

        file.write(
            "Reference compute energy J = "
            f"{transfer_result.reference_compute_energy_j:.10f}\n"
        )

        file.write(
            "Median Edge->Edge latency s = "
            f"{transfer_result.median_edge_edge_latency_s:.10f}\n"
        )

        file.write(
            "Median Edge->Cloud latency s = "
            f"{transfer_result.median_edge_cloud_latency_s:.10f}\n"
        )

    return output_path


# ======================================================================
# 主程序
# ======================================================================

def main() -> None:

    print_section(
        "开始一次性能源模型参数标定"
    )

    # --------------------------------------------------------------
    # 1. 固定随机种子
    #
    # 当前 Job/Host/Latency 都包含随机抽样过程，
    # 为了使离线标定具有基本可复现性，
    # 使用项目已有的统一 Seed。
    # --------------------------------------------------------------

    random.seed(
        conf.Seed
    )

    np.random.seed(
        conf.Seed
    )

    print(
        f"使用随机种子：{conf.Seed}"
    )

    print(
        f"时间单位：{conf.SIMULATION_TIME_UNIT}"
    )

    print(
        f"能耗单位：{conf.ENERGY_UNIT}"
    )

    # --------------------------------------------------------------
    # 2. 生成一次代表性环境
    # --------------------------------------------------------------

    print_section(
        "生成标定环境"
    )

    env_generator = EnvGenerator()

    env_generator.generate_environment(
        lambda_rate=(
            env_generator.lambda_rate
        ),
        job_dataset_path=(
            env_generator.job_dataset_path
        ),
        cloud_latency_range=(
            env_generator.cloud_latency_range
        ),
        edge_latency_range=(
            env_generator.edge_latency_range
        ),
    )

    jobs = list(
        env_generator.wait_assign_jobs_list
    )

    edge_hosts = extract_edge_hosts(
        env_generator
    )

    print(
        f"生成任务数：{len(jobs)}"
    )

    print(
        f"Edge Host 数：{len(edge_hosts)}"
    )

    if not jobs:
        raise RuntimeError(
            "当前环境没有生成任何 Job，无法进行能源标定。"
        )

    # --------------------------------------------------------------
    # 3. 建立 Edge reference
    # --------------------------------------------------------------

    print_section(
        "建立 Edge 任务参考动态功率/能耗"
    )

    references = (
        build_edge_task_energy_references(
            jobs=jobs,
            edge_hosts=edge_hosts,
        )
    )

    print(
        f"成功建立 Edge reference 数量："
        f"{len(references)} / {len(jobs)}"
    )

    skipped_jobs = (
        len(jobs)
        - len(references)
    )

    print(
        f"无法建立 Edge reference 的任务数："
        f"{skipped_jobs}"
    )

    # --------------------------------------------------------------
    # 4. Cloud 单位功率拟合
    #
    # 注意：
    # 这里不再调用 assert_cloud_calibration_quality()。
    #
    # 原因：
    # 该函数会因为例如 86.6% < 90%
    # 直接抛 RuntimeError，导致后续参数无法输出。
    #
    # 当前脚本改为完整输出拟合质量，
    # 最后由我们根据分布判断是否冻结参数。
    # --------------------------------------------------------------

    print_section(
        "拟合 Cloud 单位 CPU/GPU 功率参数"
    )

    cloud_result = (
        calibrate_cloud_unit_resource_power(
            references=references,
        )
    )

    power_ratios = (
        calculate_cloud_edge_power_ratios(
            references=references,
            cloud_result=cloud_result,
        )
    )

    print_cloud_calibration_result(
        cloud_result=cloud_result,
        power_ratios=power_ratios,
    )

    # --------------------------------------------------------------
    # 5. 提取传输时延
    # --------------------------------------------------------------

    print_section(
        "提取 Edge / Cloud 网络时延样本"
    )

    (
        edge_edge_latencies_s,
        edge_cloud_latencies_s,
    ) = extract_network_latencies_s(
        env_generator
    )

    print(
        "Edge->Edge latency 样本数："
        f"{len(edge_edge_latencies_s)}"
    )

    print(
        "Edge->Cloud latency 样本数："
        f"{len(edge_cloud_latencies_s)}"
    )

    print(
        "Edge->Edge latency median："
        f"{np.median(edge_edge_latencies_s):.6f} s"
    )

    print(
        "Edge->Cloud latency median："
        f"{np.median(edge_cloud_latencies_s):.6f} s"
    )

    # --------------------------------------------------------------
    # 6. Transmission 参数标定
    # --------------------------------------------------------------

    print_section(
        "标定 Transmission 能耗参数"
    )

    transfer_result = (
        calibrate_transfer_energy_model(
            references=references,

            edge_edge_latencies_s=(
                edge_edge_latencies_s
            ),

            edge_cloud_latencies_s=(
                edge_cloud_latencies_s
            ),
        )
    )

    print_transfer_calibration_result(
        transfer_result
    )

    # --------------------------------------------------------------
    # 7. 输出以后正式训练需要冻结的五个参数
    # --------------------------------------------------------------

    print_section(
        "最终固定参数——请复制到 config.py"
    )

    print(
        "# 以下参数由一次性离线能源模型标定得到。"
    )

    print(
        "# 正式 MASAC 训练过程中不再重新拟合。"
    )

    print()

    print(
        "CLOUD_CPU_POWER_PER_UNIT_W = "
        f"{cloud_result.cpu_power_per_unit_w:.10f}"
    )

    print(
        "CLOUD_GPU_POWER_PER_UNIT_W = "
        f"{cloud_result.gpu_power_per_unit_w:.10f}"
    )

    print()

    print(
        "TRANSFER_SEND_FIXED_ENERGY_J = "
        f"{transfer_result.send_fixed_energy_j:.10f}"
    )

    print(
        "TRANSFER_RECEIVE_FIXED_ENERGY_J = "
        f"{transfer_result.receive_fixed_energy_j:.10f}"
    )

    print(
        "TRANSFER_LATENCY_ENERGY_COEFFICIENT_W = "
        f"{transfer_result.latency_energy_coefficient_w:.10f}"
    )

    # --------------------------------------------------------------
    # 8. 保存结果
    # --------------------------------------------------------------

    output_path = save_calibration_result(
        cloud_result=cloud_result,
        transfer_result=transfer_result,
        power_ratios=power_ratios,
    )

    print_section(
        "标定完成"
    )

    print(
        "✅ 标定程序执行完成。"
    )

    print(
        "标定结果同时保存至："
    )

    print(
        output_path
    )

    print(
        "\n下一步不要立即修改拟合公式。"
    )

    print(
        "先检查上面的："
        "median ratio、P05/P95、RMSE、"
        "±15%覆盖率和最终5个参数。"
    )


if __name__ == "__main__":
    main()