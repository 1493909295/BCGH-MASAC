from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'analysis_outputs'
OUT.mkdir(exist_ok=True)
paths={
 'BCGH-MASAC':ROOT/'result/BGH-MASAC/BCGH-MASAC-disCloud/0.35-2000-异构-1.csv',
 'H-MASAC':ROOT/'result/H-MASAC/H-MASAC-disCloud/0.35-2000-异构1.0.csv',
 'OPT':ROOT/'result/OPT/disCloud/0.35-2000-异构1.0.csv'}
data={k:pd.read_csv(v) for k,v in paths.items()}
dcs={k:pd.read_csv(v.with_name('dc_log-'+v.name)) for k,v in paths.items()}
windows=[(151,200),(601,650),(651,700),(701,710),(711,750),(751,800),(801,900),(901,1000)]
metrics=['training_episode_reward','completion_rate','sla_satisfaction_rate','avg_completion_time','total_system_energy_kwh','task_attributable_energy_per_total_job_j','queued_jobs','waiting_timeout_drops','routing_decision_count','routing_self_rate','routing_edge_rate','routing_cloud_rate','multi_hop_job_rate','avg_routing_edge_hops_per_job','host_started_rate','host_queued_rate','routing_critic_loss','routing_alpha','routing_policy_entropy','routing_target_entropy','host_critic_loss','host_alpha_mean','host_policy_entropy','host_target_entropy','edge_avg_cpu_load','edge_avg_gpu_load','host_episode_updates','host_replay_size_total','bgh_guidance_lambda']
rows=[]
for name,d in data.items():
 print('\nMODEL',name,'dc_shape',dcs[name].shape)
 print('flags',d[[c for c in ['cloud_enabled','neighbor_feedback_decision_enabled','energy_cost_weight','energy_normalization_j','bgh_bayesian_game_enabled','bgh_heuristic_guidance_enabled'] if c in d]].drop_duplicates().to_dict('records'))
 print('sanity',{'duplicate_episodes':int(d.episode.duplicated().sum()),'unresolved_max':int(d.unresolved_jobs.max()),'trace_gap_max':float(d.causal_terminal_job_gap.abs().max()),'dc_rows_per_episode':dcs[name].groupby('episode').size().unique().tolist()})
 for a,b in windows:
  x=d[d.episode.between(a,b)]
  row={'model':name,'window':f'{a}-{b}',**{c:float(x[c].mean()) for c in metrics if c in x}}
  rows.append(row)
 print(pd.DataFrame([r for r in rows if r['model']==name]).drop(columns='model').round(4).to_string(index=False))
 print('BOUNDARY',d[d.episode.between(698,704)][['episode','completion_rate','sla_satisfaction_rate','host_started_rate','host_episode_updates','host_random_action_count','host_policy_action_count','host_alpha_mean','host_policy_entropy','host_replay_size_total']].round(4).to_string(index=False))
 for a,b in [(651,700),(701,710),(901,1000)]:
  x=dcs[name][dcs[name].episode.between(a,b)]
  cols=['exogenous_arrival_count','routing_decisions','route_self_count','route_out_edge_count','route_in_edge_count','net_edge_migration_count','route_cloud_count','host_decision_count','host_started_count','host_queued_count','host_dropped_count','host_episode_updates','host_replay_size','host_alpha','host_policy_entropy','host_target_entropy','host_critic_loss','dc_completed_jobs','dc_cpu_capacity','dc_gpu_capacity','dc_avg_cpu_load','dc_avg_gpu_load']
  print('DC',a,b,x.groupby('dc_id')[[c for c in cols if c in x]].mean().round(3).to_string())
summary=pd.DataFrame(rows)
summary.to_csv(OUT/'window_metrics.csv',index=False,encoding='utf-8-sig')
base=data['H-MASAC']
work=[c for c in base if c.startswith('workload_') or c in ['episode_seed','edge_total_cpu_capacity','edge_total_gpu_capacity','configured_total_arrival_rate']]
for name,d in data.items():
 diffs={c:float((d[c]-base[c]).abs().max()) for c in work if pd.api.types.is_numeric_dtype(d[c])}
 print('WORKLOAD_MAX_DIFF',name,{c:v for c,v in diffs.items() if v!=0})
 for a,b in [(651,700),(901,1000)]:
  x=d[d.episode.between(a,b)]
  print('ENERGY',name,a,b,x[['edge_idle_energy_j','system_dynamic_compute_energy_j','transfer_energy_j','task_attributable_energy_j','total_system_energy_j','simulation_end_time']].mean().round(1).to_dict())
  m={}
  for s in x.routing_source_target_matrix_json:
   for src,targets in json.loads(s).items():
    for dst,n in targets.items():m[(src,dst)]=m.get((src,dst),0)+n/len(x)
  print('ROUTES',name,a,b,m)
try:
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(2,3,figsize=(14,7),constrained_layout=True)
 specs=[('completion_rate','Completion rate (%)',100),('sla_satisfaction_rate','SLA satisfaction (%)',100),('host_started_rate','Immediate host starts (%)',100),('routing_edge_rate','Edge routing decisions (%)',100),('avg_completion_time','Mean completion time (s)',1),('total_system_energy_kwh','System energy (kWh)',1)]
 for ax,(c,title,scale) in zip(axs.flat,specs):
  for name,d in data.items():ax.plot(d.episode,d[c].rolling(20,min_periods=20).mean()*scale,label=name,lw=1.5)
  ax.axvline(700,color='black',ls='--',lw=.8)
  ax.axvline(200,color='gray',ls=':',lw=.8)
  ax.set_title(title);ax.set_xlabel('Episode');ax.grid(alpha=.2)
 axs[0,0].legend(fontsize=8)
 fig.suptitle('Three schedulers: trailing 20-episode means; joint training starts at 701')
 fig.savefig(OUT/'scheduler_comparison.png',dpi=170)
except ImportError as e:print(e)
