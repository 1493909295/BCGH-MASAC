from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'result/BCGH-MASAC/0.35-2000-cloud-异构/不同的三阶段'
METRICS = ['training_episode_reward','routing_layer_reward_sum','host_layer_reward_sum','completion_rate','sla_satisfaction_rate','avg_completion_time','total_system_energy_kwh','task_attributable_energy_per_total_job_j','routing_self_rate','routing_edge_rate','routing_cloud_rate','multi_hop_job_rate','host_started_rate','host_queued_rate','host_dropped_rate','routing_episode_updates','host_episode_updates','bcgh_guidance_lambda','routing_alpha','routing_policy_entropy','host_alpha_mean','host_policy_entropy']
data = {}
for name in ['200-200-700','200-800','500-500']:
    d = pd.read_csv(SRC / f'{name}.csv')
    data[name] = d
    print('\nRUN',name,'shape',d.shape,'episodes',d.episode.min(),d.episode.max(),'duplicates',d.episode.duplicated().sum())
    print(d.groupby('training_stage',sort=False).agg(start=('episode','min'),end=('episode','max'),n=('episode','size'),routing_updates=('routing_episode_updates','mean'),host_updates=('host_episode_updates','mean'),lambda_min=('bcgh_guidance_lambda','min'),lambda_max=('bcgh_guidance_lambda','max')).to_string())
    s = int(d.loc[d.routing_episode_updates.gt(0),'episode'].min())
    d['routing_relative_episode'] = d.episode - s
    print('FIRST_ROUTING_UPDATE',s,'seed',d.loc[d.episode.eq(s),'episode_seed'].tolist())
    print('FLAGS',d[['cloud_enabled','bcgh_bayesian_game_enabled','bcgh_heuristic_guidance_enabled','energy_normalization_j','energy_cost_weight','dc_arrival_mode','configured_total_arrival_rate','total_jobs']].drop_duplicates().to_dict('records'))
    print('SANITY',d[['unresolved_jobs','pending_trace_count_end','causal_terminal_job_gap','energy_time_gap_s','transfer_split_gap_j','transfer_job_accounting_gap_j','workload_missing_origin_count','workload_unknown_origin_count']].abs().max().to_dict())
    rows=[]
    for a,b in [(-50,-1),(0,49),(50,99),(100,199),(200,299),(300,399),(400,499),(500,599),(600,699),(700,799),(800,899)]:
        x=d[d.routing_relative_episode.between(a,b)]
        if len(x):rows.append({'window':f'{a}:{b}','n':len(x),'original_start':int(x.episode.min()),'original_end':int(x.episode.max()),'stage':','.join(x.training_stage.unique()),**{c:x[c].mean() for c in METRICS}})
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print('BOUNDARIES',d.loc[d.training_stage.ne(d.training_stage.shift()),['episode','training_stage','bcgh_guidance_lambda','routing_random_action_count','routing_policy_action_count','host_random_action_count','host_policy_action_count','routing_episode_updates','host_episode_updates']].to_string(index=False))
    print('MISSING',d[METRICS].isna().sum().loc[lambda x:x.gt(0)].to_dict())
for i,(name,d) in enumerate(data.items()):
    for other,q in list(data.items())[i+1:]:
        cols=['episode_seed']+[c for c in d if c.startswith('workload_') and pd.api.types.is_numeric_dtype(d[c])]
        a=d.set_index('episode')[cols];b=q.set_index('episode')[cols]
        idx=a.index.intersection(b.index)
        diff=(a.loc[idx]-b.loc[idx]).abs().max()
        print('SAME_ORIGINAL_EPISODE_WORKLOAD_DIFF',name,other,diff[diff.gt(1e-9)].to_dict())
        idx=d.routing_relative_episode[d.routing_relative_episode.ge(0)].tolist()
        ra=d.set_index('routing_relative_episode');rb=q.set_index('routing_relative_episode')
        ix=ra.index.intersection(rb.index);ix=ix[ix>=0]
        print('ALIGNED_SEED_EQUAL_FRACTION',name,other,float((ra.loc[ix,'episode_seed']==rb.loc[ix,'episode_seed']).mean()))
