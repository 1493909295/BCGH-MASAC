import pandas as pd
import networkx as nx
import os
import re
import config as conf
from datetime import datetime
from environment.datacenter import (DataCenter, Host)
from environment.job import (Job, JobList,jobs_generate)
from environment.datacenter import (hosts_generate, datacenters_generate)






class EnvGenerator:
    def __init__(self):
        self.datacenter_num = conf.NUM_DATACENTERS
        self.host_num = conf.NUM_HOST
        # self.job_num = conf.NUM_JOBS
        # self.lambda_rate = conf.LAMBDA_RATE

        # self.wait_assign_jobs_list = []  # 待分配 job 列表

        self.wait_assign_host_list = []  # 待分配 host 列表
        self.global_dc_list = []  # 全局 datacenter 列表

        self.datacenter_graph = None  # datacenter 拓扑图

        # self.job_dataset_path = conf.JOB_DATASET_PATH
        self.host_dataset_path = conf.HOST_DATASET_PATH
        self.cloud_latency_range = conf.CLOUD_LATENCY_RANGE
        self.edge_latency_range = conf.EDGE_LATENCY_RANGE
        self.env_keep_path = conf.ENV_KEEP_PATH

    def keep_environment_state(self):
        # 在路径下新建文件夹
        base_keep_path = self.env_keep_path
        folder_name = datetime.now().strftime("%Y_%m_%d_%H_%M")
        target_dir = os.path.join(base_keep_path, folder_name)
        os.makedirs(target_dir, exist_ok=True)

        # 保存 datacenters.csv
        dc_records = []
        for dc in self.global_dc_list:
            # 提取该节点在 networkx 拓扑图中的邻居及互连延迟权重
            topology_links = []
            if self.datacenter_graph and dc.dc_id in self.datacenter_graph:
                for neighbor in self.datacenter_graph.neighbors(dc.dc_id):
                    weight = self.datacenter_graph[dc.dc_id][neighbor].get('weight', 0)
                    topology_links.append(f"{neighbor}({weight}s)")

            dc_records.append({
                'datacenter_id': dc.dc_id,
                'cloud_latency_s': getattr(dc, 'cloud_latency', 'N/A'),
                'edge_topology_links': "; ".join(topology_links)  # 用分号隔开每个邻居关系
            })
        pd.DataFrame(dc_records).to_csv(os.path.join(target_dir, 'datacenters.csv'), index=False, encoding='utf-8-sig')

        # 保存 host.csv
        host_records = []
        for dc in self.global_dc_list:
            for host in dc.host_list:
                host_records.append({
                    'host_id': host.host_id,
                    # 'gpu_model': host.gpu_model,
                    'gpu_capacity_num': host.gpu_capacity_num,
                    'cpu_num': host.cpu_num,
                    'datacenter_id': dc.dc_id  # 标记当前主机被分配到了哪个边缘节点
                })
        pd.DataFrame(host_records).to_csv(os.path.join(target_dir, 'hosts.csv'), index=False, encoding='utf-8-sig')

        # 保存 job.csv
        # job_records = []
        # for job in self.wait_assign_jobs_list:
        #     job_records.append({
        #         'job_id': job.job_id,
        #         'cpu_request': job.cpu_request,
        #         'gpu_request': job.gpu_request,
        #         'duration': job.duration,
        #         'arrive_time': job.arrive_time
        #         # 'target_datacenter_id': job.target_datacenter
        #     })
        # pd.DataFrame(job_records).to_csv(os.path.join(target_dir, 'jobs.csv'), index=False, encoding='utf-8-sig')

        # 保存 env_information.txt
        info_txt_path = os.path.join(target_dir, 'env_information.txt')
        with open(info_txt_path, 'w', encoding='utf-8') as f:
            f.write(f"================ 环境生成元数据报告 ================\n")
            f.write(f"生成时间         : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据中心数量     : {self.datacenter_num} 个\n")
            f.write(f"实例化主机总数   : {self.host_num} 台\n")
            # f.write(f"模拟生成的任务数 : {self.job_num} 个\n")
            # f.write(f"任务到达率     : {self.lambda_rate}\n")
            f.write(f"边->云端时延范围 : {self.cloud_latency_range} s\n")
            f.write(f"边边互相时延范围 : {self.edge_latency_range} s\n")
            f.write(f"===================================================\n")

        print(f"💾 [环境保持成功] 本次实验环境快照已安全持久化至: {target_dir}")

    def generate_environment(self,cloud_latency_range: tuple, edge_latency_range: tuple):

        # self.wait_assign_jobs_list = []  # 待分配 job 列表
        self.wait_assign_host_list = []  # 待分配 host 列表
        self.global_dc_list = []  # 全局 datacenter 列表
        self.datacenter_graph = None  # datacenter 拓扑图

        # 生成待分配 job
        # self.wait_assign_jobs_list = jobs_generate(
        #     self.job_num,
        #     self.lambda_rate,
        #     self.job_dataset_path,
        #     self.wait_assign_jobs_list)
        # self.wait_assign_jobs_list.sort(key=lambda job: job.arrive_time)

        # 生成待分配 host
        self.wait_assign_host_list = hosts_generate(
            self.wait_assign_host_list,
            self.host_num,
            self.host_dataset_path
        )

        # 生成 datacenter 拓扑
        self.global_dc_list,self.datacenter_graph = datacenters_generate(
            self.datacenter_num,
            self.wait_assign_host_list,
            self.cloud_latency_range,
            self.edge_latency_range
        )

        # # 为 datacenter 分发job
        # for job in self.wait_assign_jobs_list:
        #     chosen_dc = random.choice(self.global_dc_list)
        #     job.set_target_datacenter(chosen_dc.dc_id)

        # 实例化云计算节点
        cloud_host = Host(
            host_id="cloud_host",
            # gpu_model="Cloud_Infinite_Power",
            gpu_capacity_num=999999,
            cpu_num=999999
        )
        cloud_dc = DataCenter(dc_id="cloud", cloud_latency=0.0)
        cloud_dc.host_list = [cloud_host]
        # cloud_dc.job_list = []
        if self.datacenter_graph is not None:
            self.datacenter_graph.add_node("cloud", dc_instance=cloud_dc)
            for dc in self.global_dc_list:
                # 获取该边缘节点到云的时延
                lat_weight = getattr(dc, 'cloud_latency', 0.0)
                # 连接边缘节点与中心云节点
                self.datacenter_graph.add_edge(dc.dc_id, "cloud", weight=lat_weight)
        self.global_dc_list.append(cloud_dc)




        # 环境持久化
        self.keep_environment_state()



class UseOldEnv:
    def __init__(self, target_env_folder: str):
        self.env_keep_path = target_env_folder
        self.datacenter_num = 0
        self.host_num = 0
        # self.job_num = 0
        # self.lambda_rate = 0.0
        self.cloud_latency_range = None
        self.edge_latency_range = None
        # self.job_dataset_path = conf.JOB_DATASET_PATH
        # self.host_dataset_path = conf.HOST_DATASET_PATH
        # self.wait_assign_jobs_list = []
        self.wait_assign_host_list = []
        self.global_dc_list = []
        self.datacenter_graph = nx.Graph()
        self.load_environment()

    def load_environment(self):
        # 读取环境路径
        if not os.path.exists(self.env_keep_path):
            raise FileNotFoundError(f"找不到指定的历史环境目录: {self.env_keep_path}")
        print(f" 正在从 {self.env_keep_path} 恢复环境状态...")

        # 读取 env_information.txt
        info_path = os.path.join(self.env_keep_path, 'env_information.txt')
        if os.path.exists(info_path):
            with open(info_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.datacenter_num = int(re.search(r'数据中心数量\s*:\s*(\d+)', content).group(1))
                self.host_num = int(re.search(r'实例化主机总数\s*:\s*(\d+)', content).group(1))
                # self.job_num = int(re.search(r'生成的任务数\s*:\s*(\d+)', content).group(1))
                # self.lambda_rate = float(re.search(r'任务到达率\s*:\s*([\d\.]+)', content).group(1))
                cloud_match = re.search(r'云端时延范围\s*:\s*\((.*?)\)', content)
                edge_match = re.search(r'互相时延范围\s*:\s*\((.*?)\)', content)
                if cloud_match:
                    self.cloud_latency_range = tuple(map(float, cloud_match.group(1).split(',')))
                if edge_match:
                    self.edge_latency_range = tuple(map(float, edge_match.group(1).split(',')))

        # 读取 datacenters.csv
        dc_df = pd.read_csv(os.path.join(self.env_keep_path, 'datacenters.csv'))
        for _, row in dc_df.iterrows():
            dc_id = row['datacenter_id']
            cloud_lat = float(row['cloud_latency_s'])
            dc = DataCenter(dc_id=dc_id, cloud_latency=cloud_lat)
            dc.cloud_latency = float(row['cloud_latency_s'])
            dc.host_list = []
            dc.job_list = []
            self.global_dc_list.append(dc)
            self.datacenter_graph.add_node(dc_id, dc_instance=dc)

        # 重建拓扑连线 (解析形如 "DC-2(15.2s); DC-3(8.5s)")
        for _, row in dc_df.iterrows():
            dc_id = row['datacenter_id']
            links_str = str(row['edge_topology_links'])
            if links_str and links_str != 'nan':
                links = links_str.split(';')
                for link in links:
                    link = link.strip()
                    if not link: continue
                    # 正则匹配提取目标 DC 和 权重
                    match = re.match(r'(.*?)\(([\d\.]+)s\)', link)
                    if match:
                        target_dc = match.group(1).strip()
                        weight = float(match.group(2))
                        # networkx 自动处理无向图的重复边
                        self.datacenter_graph.add_edge(dc_id, target_dc, weight=weight)

        # 读取 hosts.csv
        host_df = pd.read_csv(os.path.join(self.env_keep_path, 'hosts.csv'))
        for _, row in host_df.iterrows():
            host = Host(
                host_id=row['host_id'],
                # gpu_model=row['gpu_model'],
                gpu_capacity_num=row['gpu_capacity_num'],
                cpu_num=row['cpu_num']
            )
            self.wait_assign_host_list.append(host)
            # 挂载到对应的 DataCenter
            target_dc_id = row['datacenter_id']
            for dc in self.global_dc_list:
                if dc.dc_id == target_dc_id:
                    dc.host_list.append(host)
                    break

        # 读取 jobs.csv
        # job_df = pd.read_csv(os.path.join(self.env_keep_path, 'jobs.csv'))
        # for _, row in job_df.iterrows():
        #     job = Job(
        #         job_id=row['job_id'],
        #         cpu_request=row['cpu_request'],
        #         gpu_request=row['gpu_request'],
        #         duration=row['duration']
        #     )
        #     job.set_arrive_time(row['arrive_time'])
        #     # target_dc_id = row['target_datacenter_id']
        #     # job.set_target_datacenter(target_dc_id)
        #     self.wait_assign_jobs_list.append(job)
        #     # 分发到对应的 DataCenter
        #     # for dc in self.global_dc_list:
        #     #     if dc.dc_id == target_dc_id:
        #     #         dc.job_list.append(job)
        #     #         break
        #     self.wait_assign_jobs_list.sort(key=lambda job: job.arrive_time)

        print("✅ 环境历史状态加载并重建完成！")

if __name__ == "__main__":

    print("🚀 正在初始化全局调度环境仿真测试...")
    env = EnvGenerator()
    # print(f"正在生成 {env.job_num} 个任务, {env.host_num} 台主机，以及 {env.datacenter_num} 个数据中心拓扑...")
    # env.generate_environment(
    #     lambda_rate=env.lambda_rate,
    #     job_dataset_path=env.job_dataset_path,
    #     cloud_latency_range=env.cloud_latency_range,
    #     edge_latency_range=env.edge_latency_range
    # )
    print("\n✅ 环境生成完毕！详细分配情况如下：")
    dc_list = env.global_dc_list
    for dc in dc_list:
        print("\n" + "=" * 50)
        print(f"🏢 数据中心: 【{dc.dc_id}】")

        print(f"  💻 挂载主机 (Hosts) 共 {len(dc.host_list)} 台:")
        for host in dc.host_list:
            print(
                f"      - 主机ID: {host.host_id} | CPU: {host.cpu_num}核 | GPU: {host.gpu_capacity_num}")

    print("\n✅ 环境生成完毕！待分配任务队列详情如下：")
    # total_jobs = len(env.wait_assign_jobs_list)
    # print(f"📊 总计生成待分配任务数: {total_jobs}")

    # if total_jobs > 0:
    #     # 验证排序：打印前 10 个任务
    #     print("\n📋 队列头部任务 (前 10 个):")
    #     for i, job in enumerate(env.wait_assign_jobs_list[:10]):
    #         print(
    #             f"  [{i + 1:04d}] 任务ID: {str(job.job_id)[:10]:<10} | ⏰ 到达时间: {job.arrive_time:<8.4f} | CPU需求: {job.cpu_request:<4.1f}")
    #
    #     print("\n  ...... (省略中间任务) ......")
    #
    #     # 验证排序：打印最后 5 个任务
    #     print("\n📋 队列尾部任务 (最后 5 个):")
    #     for i, job in enumerate(env.wait_assign_jobs_list[-5:]):
    #         idx = total_jobs - 5 + i + 1
    #         print(
    #             f"  [{idx:04d}] 任务ID: {str(job.job_id)[:10]:<10} | ⏰ 到达时间: {job.arrive_time:<8.4f} | CPU需求: {job.cpu_request:<4.1f}")
    #
    #     # 严格的断言验证：检查整个列表是否为非递减状态
    #     is_sorted = all(
    #         env.wait_assign_jobs_list[i].arrive_time <= env.wait_assign_jobs_list[i + 1].arrive_time
    #         for i in range(len(env.wait_assign_jobs_list) - 1)
    #     )
    #     if is_sorted:
    #         print("\n🎉 排序校验通过: 整个任务队列已经严格按照到达时间升序排列！")
    #     else:
    #         print("\n⚠️ 排序校验失败: 任务队列中存在时间乱序情况！")
    # else:
    #     print("⚠️ 未生成任何任务，请检查配置参数。")


    print("\n" + "=" * 50)
    print("🎉 环境生成测试顺利完成！所有模块运转正常。")

    #======================================================================================================================================

    # # 假设你的保存路径是这个（请替换为你真实生成的文件夹名字）
    # TARGET_FOLDER = "./env_keep/2026_07_03_14_39"
    #
    # print(f"🔄 测试启动: 尝试从 {TARGET_FOLDER} 加载环境...")
    #
    # # 实例化并恢复环境
    # old_env = UseOldEnv(target_env_folder=TARGET_FOLDER)
    #
    # # 简单的断言验证
    # print(f"📊 加载元数据比对：")
    # print(f" -> DC数量: {old_env.datacenter_num} | 实际解析出对象数: {len(old_env.global_dc_list)}")
    # print(f" -> Host数量: {old_env.host_num} | 实际解析出对象数: {len(old_env.wait_assign_host_list)}")
    # print(f" -> Job数量: {old_env.job_num} | 实际解析出对象数: {len(old_env.wait_assign_jobs_list)}")
    #
    # # 打印其中一个 DC 来看看是否真的挂载成功了
    # test_dc = old_env.global_dc_list[0]
    # print(f"\n🏢 抽查节点 【{test_dc.dc_id}】 详情:")
    # print(f"   🌐 云端时延: {test_dc.cloud_latency} s")
    # print(f"   💻 恢复的主机数: {len(test_dc.host_list)}")
    # print(f"   📝 恢复的任务数: {len(test_dc.job_list)}")
    # print(f"   🔗 拓扑邻居数: {len(list(old_env.datacenter_graph.neighbors(test_dc.dc_id)))}")


#===================================================================================================