from pathlib import Path
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parent
paths={'A':ROOT/'result/BCGH-MASAC/dis/0.35-2-8.csv','B':ROOT/'result/LMD=1/BCGH/DIS/0.35-2-8.csv'}
data={k:pd.read_csv(p) for k,p in paths.items()}
dc={k:pd.read_csv(p.with_name('dc_log-'+p.name)) for k,p in paths.items()}
metrics=['training_episode_reward','completion_rate','sla_satisfaction_rate','sla_violation_rate','avg_completion_time','mean_sla_violation_degree','total_system_energy_kwh','system_energy_per_completed_job_j','task_attributable_energy_per_total_job_j','queued_jobs','waiting_timeout_drops','routing_decision_count','routing_self_rate','routing_edge_rate','routing_cloud_rate','routing_drop_rate','multi_hop_job_rate','avg_routing_edge_hops_per_job','edge_target_absorption_rate','edge_to_edge_reforward_rate','host_started_rate','host_queued_rate','routing_critic_loss','routing_actor_loss','routing_alpha','routing_policy_entropy','routing_target_entropy','host_critic_loss','host_actor_loss','host_alpha_mean','host_policy_entropy','host_target_entropy','edge_avg_cpu_load','edge_avg_gpu_load','wall_time_seconds','bcgh_guidance_lambda','simulation_end_time']
for k,d in data.items():
 print('MODEL',k,'shape',d.shape,'dc',dc[k].shape,'episode',d.episode.min(),d.episode.max())
 print('stages',d.groupby('training_stage').episode.agg(['min','max','size']).to_dict('index'))
 flags=[c for c in d if c.endswith('_enabled') or c in ['energy_normalization_j','energy_cost_weight','bcgh_zero_diff_mode','dc_arrival_mode','configured_total_arrival_rate']]
 print('flags',{c:d[c].unique().tolist() for c in flags})
 print('quality',{'duplicate_episodes':int(d.episode.duplicated().sum()),'duplicate_dc':int(dc[k].duplicated(['episode','dc_id']).sum()),'dc_per_ep':dc[k].groupby('episode').size().value_counts().to_dict(),'missing':d.isna().sum()[d.isna().any()].to_dict(),'dc_missing':dc[k].isna().sum()[dc[k].isna().any()].to_dict(),'unresolved':d.unresolved_jobs.max(),'job_gap':(d.total_jobs-d.completed_jobs-d.dropped_jobs-d.unresolved_jobs).abs().max(),'causal_gap':d.causal_terminal_job_gap.abs().max()})
 print('last100',d.tail(100)[metrics].agg(['mean','std','min','max']).round(5).T.to_string())
 print('stage_means',d.groupby('training_stage')[metrics].mean().round(4).T.to_string())
 print('dc_tail',dc[k][dc[k].episode>=d.episode.max()-99].groupby('dc_id')[['exogenous_arrival_count','routing_decisions','route_self_count','route_out_edge_count','route_in_edge_count','incoming_edge_absorption_rate','net_edge_migration_count','host_started_count','host_queued_count','host_dropped_count','dc_completed_jobs','dc_avg_cpu_load','dc_avg_gpu_load','dc_cpu_capacity','dc_gpu_capacity','host_critic_loss','host_policy_entropy','host_target_entropy','host_alpha']].mean().round(3).to_string())
 print('lambda',d.groupby('training_stage').bcgh_guidance_lambda.agg(['min','max','first','last']).to_string())
a,b=data['A'].set_index('episode'),data['B'].set_index('episode')
common=a.index.intersection(b.index)
print('SCHEMA',set(a)-set(b),set(b)-set(a),'common',len(common))
work=[c for c in a if c.startswith('workload_') or c in ['episode_seed','total_jobs','configured_total_arrival_rate','edge_total_cpu_capacity','edge_total_gpu_capacity']]
print('WORK_DIFF',{c:float((a.loc[common,c]-b.loc[common,c]).abs().max()) for c in work if pd.api.types.is_numeric_dtype(a[c])})
print('SEEDS',a.episode_seed.head().tolist(),b.episode_seed.head().tolist())
