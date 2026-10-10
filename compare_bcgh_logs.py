from pathlib import Path
import json, base64, html, sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'analysis_outputs'/'bcgh_pair_0.35-2-8'
OUT.mkdir(parents=True,exist_ok=True)
PATHS={'A':ROOT/'result/BCGH-MASAC/dis/0.35-2-8.csv','B':ROOT/'result/LMD=1/BCGH/DIS/0.35-2-8.csv'}
DATA={k:pd.read_csv(p).sort_values('episode').reset_index(drop=True) for k,p in PATHS.items()}
DC={k:pd.read_csv(p.with_name('dc_log-'+p.name)).sort_values(['episode','dc_id']).reset_index(drop=True) for k,p in PATHS.items()}
for p in ['C:/Windows/Fonts/msyh.ttc','C:/Windows/Fonts/simhei.ttf']:
    if Path(p).exists():
        font_manager.fontManager.addfont(p)
        plt.rcParams['font.family']=font_manager.FontProperties(fname=p).get_name()
        break
plt.rcParams.update({'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.titleweight':'bold','svg.fonttype':'none'})
COLORS={'A':'#1769AA','B':'#D66A20'}
NAMES={'A':'A：BCGH-MASAC/dis','B':'B：LMD=1/BCGH/DIS'}
IDS=sorted(DC['A'].dc_id.unique())
SPECS=[
 ('training_episode_reward','训练回报',1),('completion_rate','完成率 (%)',100),('sla_satisfaction_rate','SLA 满足率 (%)',100),
 ('avg_completion_time','完成任务平均周转时间 (s)',1),('total_system_energy_kwh','系统总能耗 (kWh)',1),('system_energy_per_completed_job_j','每完成任务系统能耗 (kJ)',.001),
 ('routing_decision_count','路由决策数 / 回合',1),('routing_edge_rate','跨边缘路由决策比例 (%)',100),('multi_hop_job_rate','多跳任务比例 (%)',100),
 ('avg_routing_edge_hops_per_job','边缘跳数 / 全部任务',1),('edge_target_absorption_rate','迁入后 self 决策比例 (%)',100),('edge_to_edge_reforward_rate','迁入后继续转发比例 (%)',100),
 ('host_started_rate','Host 立即启动比例 (%)',100),('host_queued_rate','Host 排队决策比例 (%)',100),('host_dropped_count','Host 丢弃数 / 回合',1),
 ('queued_jobs','排队任务数 / 回合',1),('waiting_timeout_drops','等待超时丢弃数 / 回合',1),('drop_rate','任务丢弃率 (%)',100),
 ('task_attributable_energy_per_total_job_j','任务归属能耗 / 全部任务 (kJ)',.001),('transfer_energy_j','传输能耗 (kWh)',1/3600000),
 ('simulation_end_time','仿真结束时间 (s)',1),('wall_time_seconds','实际运行时间 / 回合 (s)',1),
 ('routing_critic_loss','Routing Critic 损失',1),('routing_actor_loss','Routing Actor 损失',1),('routing_alpha','Routing 温度 α',1),
 ('routing_policy_entropy','Routing 策略熵',1),('edge_avg_cpu_load','边缘平均 CPU 负载 (%)',100),('edge_avg_gpu_load','边缘平均 GPU 负载 (%)',100),
 ('mean_sla_violation_degree','平均 SLA 违规程度',1),('first_edge_absorption_rate','首次迁入后 self 比例 (%)',100),
 ('host_critic_loss','Host Critic 损失',1),('host_policy_entropy','Host 策略熵',1),('host_alpha_mean','Host 平均温度 α',1)]
LABEL={c:(lab,scale) for c,lab,scale in SPECS}
METRICS=[c for c,_,_ in SPECS]
TAIL={k:d[d.episode.between(901,1000)] for k,d in DATA.items()}
def fmt(v):
    if pd.isna(v):return '未记录'
    return f'{v:,.3f}' if abs(v)<10 else f'{v:,.2f}'
def table(frame):
    headers=list(frame.columns)
    vals=[[fmt(x) if isinstance(x,(float,np.floating)) else str(x) for x in row] for row in frame.itertuples(index=False,name=None)]
    md='| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'
    md+='\n'.join('| '+' | '.join(r)+' |' for r in vals)
    ht='<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in headers)+'</tr></thead><tbody>'
    ht+=''.join('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in row)+'</tr>' for row in vals)+'</tbody></table></div>'
    return md,ht
BLOCKS=[]
def heading(text,level=2):BLOCKS.append(('#'*level+' '+text,f'<h{level}>{html.escape(text)}</h{level}>'))
def para(text):BLOCKS.append((text,'<p>'+html.escape(text)+'</p>'))
def addtable(frame):BLOCKS.append(table(frame))
def image(name,caption):
    b64=base64.b64encode((OUT/name).read_bytes()).decode()
    BLOCKS.append((f'![{caption}]({name})\n\n{caption}',f'<figure><img src="data:image/png;base64,{b64}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'))
def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=175,bbox_inches='tight')
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight')
    plt.close(fig)
def trend(ax,c,raw=True,pretrain=True):
    label,scale=LABEL[c]
    for k,d in DATA.items():
        x=d if pretrain else d[d.episode>=201]
        if raw:ax.plot(x.episode,x[c]*scale,color=COLORS[k],alpha=.12,lw=.55)
        smooth=x.groupby('training_stage',sort=False)[c].transform(lambda s:s.rolling(20,min_periods=20).mean())
        ax.plot(x.episode,smooth*scale,color=COLORS[k],lw=1.8,label=k)
    if pretrain:
        ax.axvspan(1,200,color='#D9E1E8',alpha=.28)
        ax.axvline(200.5,color='#667',ls='--',lw=.8)
    ax.set_title(label,fontsize=11);ax.set_xlabel('回合');ax.grid(alpha=.18)
def panel(cols,title,name,pretrain=True):
    fig,axes=plt.subplots(2,3,figsize=(15,8.2),constrained_layout=True)
    for ax,c in zip(axes.flat,cols):trend(ax,c,pretrain=pretrain)
    axes[0,0].legend();fig.suptitle(title+'\nA（蓝）与 B（橙）；细线为原始值，粗线为阶段内后向 20 回合均值',fontsize=14)
    save(fig,name)

# Audit original records without altering them.
QUALITY={}
for k,d in DATA.items():
    q=DC[k]
    assert len(d)==1000 and d.episode.tolist()==list(range(1,1001))
    assert not q.duplicated(['episode','dc_id']).any()
    assert q.groupby('episode').size().eq(5).all()
    assert (d.total_jobs-d.completed_jobs-d.dropped_jobs-d.unresolved_jobs).eq(0).all()
    for count,rate in [('completed_jobs','completion_rate'),('dropped_jobs','drop_rate'),('sla_satisfied_jobs','sla_satisfaction_rate')]:
        assert np.allclose(d[count]/d.total_jobs,d[rate])
    agg=q.groupby('episode')[['dc_completed_jobs','routing_decisions','host_decision_count','route_out_edge_count','route_in_edge_count']].sum()
    assert np.allclose(agg.dc_completed_jobs,d.completed_jobs)
    assert np.allclose(agg.routing_decisions,d.routing_decision_count)
    assert np.allclose(agg.host_decision_count,d.host_decision_count)
    assert agg.route_out_edge_count.equals(agg.route_in_edge_count)
    assert np.allclose(d.edge_idle_energy_j+d.system_dynamic_compute_energy_j+d.transfer_energy_j,d.total_system_energy_j)
    QUALITY[k]={'episode_rows':len(d),'episode_columns':len(d.columns),'dc_rows':len(q),'dc_columns':len(q.columns),'max_unresolved':int(d.unresolved_jobs.max()),'max_causal_gap':int(d.causal_terminal_job_gap.abs().max()),'max_energy_time_gap_s':float(d.energy_time_gap_s.abs().max()),'max_transfer_accounting_gap_j':float(d.transfer_job_accounting_gap_j.abs().max()),'missing_episode_fields':d.isna().sum()[d.isna().any()].to_dict()}
work=[c for c in DATA['A'] if c.startswith('workload_') and pd.api.types.is_numeric_dtype(DATA['A'][c])]+['episode_seed','total_jobs','edge_total_cpu_capacity','edge_total_gpu_capacity']
assert all(np.array_equal(DATA['A'][c],DATA['B'][c]) for c in work)
assert np.array_equal(DC['A'].exogenous_arrival_count,DC['B'].exogenous_arrival_count)
host_records=[]
host_capacity_equal=True
for k,q in DC.items():
    for row in q[q.episode>=901].itertuples():
        for host,vals in json.loads(row.host_load_details_json).items():
            host_records.append({'model':k,'episode':row.episode,'dc_id':row.dc_id,'host':host,**vals})
HOST=pd.DataFrame(host_records)
ha=HOST[HOST.model=='A'].set_index(['episode','host']);hb=HOST[HOST.model=='B'].set_index(['episode','host'])
host_capacity_equal=ha.index.equals(hb.index) and np.array_equal(ha[['cpu_capacity','gpu_capacity']],hb[['cpu_capacity','gpu_capacity']])
QUALITY['shared']={'workload_numeric_columns_equal':work,'exogenous_dc_arrivals_equal':True,'tail_host_capacities_equal':bool(host_capacity_equal)}
(OUT/'audit.json').write_text(json.dumps(QUALITY,ensure_ascii=False,indent=2),encoding='utf-8')

# Descriptive windows and paired uncertainty within this single training trajectory.
windows=[('全部',1,1000),('Host 预训练',1,200),('Routing 训练',201,1000),('末 200 回合',801,1000),('末 100 回合',901,1000)]+[(f'{lo}–{lo+99}',lo,lo+99) for lo in range(1,1001,100)]
rows=[]
for w,lo,hi in windows:
    for k,d in DATA.items():
        t=d[d.episode.between(lo,hi)]
        for c in METRICS:
            rows.append({'window':w,'start':lo,'end':hi,'model':k,'metric':c,'n_valid':int(t[c].count()),'mean':t[c].mean(),'std':t[c].std(),'median':t[c].median(),'p05':t[c].quantile(.05),'p95':t[c].quantile(.95),'min':t[c].min(),'max':t[c].max()})
pd.DataFrame(rows).to_csv(OUT/'window_statistics.csv',index=False,encoding='utf-8-sig')
rng=np.random.default_rng(20261008)
cirows=[]
for w,lo,hi in [('末 100 回合',901,1000),('末 200 回合',801,1000),('Routing 训练',201,1000)]:
    a=DATA['A'][DATA['A'].episode.between(lo,hi)];b=DATA['B'][DATA['B'].episode.between(lo,hi)]
    n=len(a)
    for length in [10,20,50]:
        starts=rng.integers(0,n-length+1,size=(6000,int(np.ceil(n/length))))
        idx=(starts[:,:,None]+np.arange(length)[None,None,:]).reshape(6000,-1)[:,:n]
        for c in METRICS:
            delta=b[c].to_numpy()-a[c].to_numpy()
            if not np.isfinite(delta).all():continue
            low,high=np.quantile(delta[idx].mean(axis=1),[.025,.975])
            cirows.append({'window':w,'metric':c,'block_length':length,'A_mean':a[c].mean(),'B_mean':b[c].mean(),'B_minus_A':delta.mean(),'ci95_low':low,'ci95_high':high,'B_greater_share':np.mean(delta>0),'paired_lag1_corr':pd.Series(delta).autocorr(1) if np.std(delta)>0 else np.nan})
CIS=pd.DataFrame(cirows);CIS.to_csv(OUT/'paired_block_bootstrap.csv',index=False,encoding='utf-8-sig')
summary=[]
for c,lab,scale in SPECS:
    a,b=TAIL['A'][c].mean(),TAIL['B'][c].mean()
    summary.append({'指标':lab,'A':a*scale,'B':b*scale,'B−A':(b-a)*scale,'B 相对 A (%)':(b/a-1)*100 if pd.notna(a) and a!=0 else np.nan})
SUMMARY=pd.DataFrame(summary);SUMMARY.to_csv(OUT/'tail100_summary.csv',index=False,encoding='utf-8-sig')
DCMEAN={k:q[q.episode>=901].groupby('dc_id').mean(numeric_only=True) for k,q in DC.items()}
dcrows=[]
for k,m in DCMEAN.items():
    for idx,row in m.iterrows():dcrows.append({'model':k,'dc_id':idx,**row.to_dict()})
pd.DataFrame(dcrows).to_csv(OUT/'dc_tail100.csv',index=False,encoding='utf-8-sig')
hostsummary=HOST.groupby(['model','dc_id']).agg(hosts=('host','nunique'),zero_cpu_share=('avg_cpu_load',lambda s:(s==0).mean()),zero_gpu_share=('avg_gpu_load',lambda s:(s==0).mean()),host_cpu_p95=('avg_cpu_load',lambda s:s.quantile(.95)),host_gpu_p95=('avg_gpu_load',lambda s:s.quantile(.95)),host_busy_mean=('busy_ratio','mean')).reset_index()
hostsummary.to_csv(OUT/'host_tail100.csv',index=False,encoding='utf-8-sig')

panel(['completion_rate','sla_satisfaction_rate','training_episode_reward','avg_completion_time','drop_rate','host_started_rate'],'图 1｜总体服务表现与训练阶段','01_performance')
panel(['routing_decision_count','routing_edge_rate','multi_hop_job_rate','avg_routing_edge_hops_per_job','edge_target_absorption_rate','edge_to_edge_reforward_rate'],'图 2｜路由代价与迁入后行为','02_routing')
panel(['total_system_energy_kwh','system_energy_per_completed_job_j','task_attributable_energy_per_total_job_j','transfer_energy_j','simulation_end_time','wall_time_seconds'],'图 3｜能耗与运行开销','03_energy')
fig,axes=plt.subplots(2,3,figsize=(15,8.2),constrained_layout=True)
for ax,c in zip([axes[0,0],axes[0,1],axes[0,2],axes[1,0]],['routing_critic_loss','routing_actor_loss','routing_alpha','host_critic_loss']):trend(ax,c)
axes[1,0].set_xlim(1,210)
for k,d in DATA.items():axes[1,1].plot(d.episode,d.host_episode_updates,color=COLORS[k],label=k)
axes[1,1].set_title('Host 更新次数 / 回合（201 后恒为 0）');axes[1,1].set_xlabel('回合');axes[1,1].grid(alpha=.18)
axes[1,2].clear()
for k,d in DATA.items():
    y=d[d.episode>=201].routing_policy_entropy-d[d.episode>=201].routing_target_entropy
    axes[1,2].plot(d[d.episode>=201].episode,y.rolling(20,min_periods=20).mean(),color=COLORS[k],label=k)
axes[1,2].axhline(0,color='#667',lw=.8);axes[1,2].set_title('Routing 实际熵 − 目标熵（20 回合均值）');axes[1,2].set_xlabel('回合');axes[1,2].grid(alpha=.18)
axes[0,0].legend();fig.suptitle('图 4｜学习状态：Host 预训练后冻结，Routing 持续训练',fontsize=15)
save(fig,'04_learning')

fig,axes=plt.subplots(2,3,figsize=(15,8.2),constrained_layout=True)
x=np.arange(len(IDS));width=.36
for ax,(c,title,scale) in zip(axes.flat,[('dc_completed_jobs','完成任务数 / 回合',1),('net_edge_migration_count','净迁入计数 / 回合（可包含重复转发）',1),('incoming_edge_absorption_rate','迁入后 self 比例 (%)',100),('dc_avg_cpu_load','DC 平均 CPU 负载 (%)',100),('dc_avg_gpu_load','DC 平均 GPU 负载 (%)',100),('host_dropped_count','Host 丢弃数 / 回合',1)]):
    for k,offset in [('A',-.5),('B',.5)]:ax.bar(x+offset*width,DCMEAN[k].loc[IDS,c]*scale,width,color=COLORS[k],label=k)
    ax.set_xticks(x,IDS);ax.set_title(title,fontsize=11);ax.grid(axis='y',alpha=.18)
axes[0,0].legend();fig.suptitle('图 5｜5 个数据中心：末 100 回合均值',fontsize=15);save(fig,'05_datacenters')

MATRIX={}
for k,t in TAIL.items():
    m=pd.DataFrame(0.,index=IDS,columns=IDS)
    for s in t.routing_source_target_matrix_json:
        for src,targets in json.loads(s).items():
            for dst,val in targets.items():
                if src in IDS and dst in IDS:m.loc[src,dst]+=val/len(t)
    assert np.allclose(m.values.sum(),t.routing_self_count.mean()+t.routing_edge_count.mean())
    MATRIX[k]=m;m.to_csv(OUT/f'routing_matrix_{k}.csv',encoding='utf-8-sig')
fig,axes=plt.subplots(1,3,figsize=(16,5.2),constrained_layout=True)
vmax=max(m.values.max() for m in MATRIX.values());dif=MATRIX['B']-MATRIX['A'];dmax=np.abs(dif.values).max()
for ax,k in zip(axes[:2],['A','B']):
    im=ax.imshow(MATRIX[k],cmap='Blues',vmin=0,vmax=vmax)
    ax.set_title(k+'：平均路由计数 / 回合')
    for i in range(5):
        for j in range(5):ax.text(j,i,f'{MATRIX[k].iloc[i,j]:.1f}',ha='center',va='center',color='white' if MATRIX[k].iloc[i,j]>.55*vmax else '#222',fontsize=10)
fig.colorbar(im,ax=axes[:2],shrink=.77,label='决策数 / 回合')
im=axes[2].imshow(dif,cmap='RdBu_r',vmin=-dmax,vmax=dmax);axes[2].set_title('B − A：红色为 B 更多')
for i in range(5):
    for j in range(5):axes[2].text(j,i,f'{dif.iloc[i,j]:+.1f}',ha='center',va='center',color='white' if abs(dif.iloc[i,j])>.55*dmax else '#222',fontsize=10)
fig.colorbar(im,ax=axes[2],shrink=.77,label='决策数差 / 回合')
for ax in axes:ax.set_xticks(x,IDS,rotation=30);ax.set_yticks(x,IDS);ax.set_xlabel('目标 DC（对角线为 self 决策）');ax.set_ylabel('源 DC')
fig.suptitle('图 6｜末 100 回合路由矩阵：计数包含同一任务的重复迁移',fontsize=15);save(fig,'06_route_matrix')

fig,axes=plt.subplots(2,3,figsize=(15,8.2),constrained_layout=True)
for ax,c in zip(axes.flat,['completion_rate','sla_satisfaction_rate','training_episode_reward','routing_decision_count','total_system_energy_kwh','transfer_energy_j']):
    ci=CIS[(CIS.metric==c)&(CIS.window=='末 100 回合')&(CIS.block_length==20)].iloc[0]
    lab,scale=LABEL[c];d=(TAIL['B'][c].to_numpy()-TAIL['A'][c].to_numpy())*scale
    ax.plot(TAIL['A'].episode,d,color='#536579',alpha=.45,lw=.8)
    ax.plot(TAIL['A'].episode,pd.Series(d).rolling(20,min_periods=20).mean(),color='#673D8D',lw=1.8)
    ax.axhline(0,color='#333',lw=.8);ax.axhline(ci.B_minus_A*scale,color='#673D8D',ls='--',lw=1)
    ax.set_title(lab+'：B − A',fontsize=11);ax.set_xlabel('回合');ax.grid(alpha=.18)
    ax.text(.03,.98,f'均值 {ci.B_minus_A*scale:+.3f}\n20 回合块 bootstrap 95% 区间\n[{ci.ci95_low*scale:+.3f}, {ci.ci95_high*scale:+.3f}]',transform=ax.transAxes,va='top',fontsize=9,bbox={'facecolor':'white','alpha':.88,'edgecolor':'#DDD'})
fig.suptitle('图 7｜配对差值与单次运行内的不确定性（901–1000）',fontsize=15);save(fig,'07_paired_differences')

fig,axes=plt.subplots(1,2,figsize=(12,5.2),constrained_layout=True)
parts=[('edge_idle_energy_j','边缘空闲'),('system_dynamic_compute_energy_j','动态计算'),('transfer_energy_j','传输')]
bottom=np.zeros(2)
for (c,label),color in zip(parts,['#96A7BA','#3478AE','#E59958']):
    vals=np.array([TAIL[k][c].mean()/3600000 for k in ['A','B']]);axes[0].bar(['A','B'],vals,bottom=bottom,label=label,color=color)
    for i in range(2):axes[0].text(i,bottom[i]+vals[i]/2,f'{vals[i]:.2f}',ha='center',va='center',fontsize=10)
    bottom+=vals
axes[0].set_title('系统能耗构成 (kWh / 回合)');axes[0].set_ylim(0,112);axes[0].legend(loc='upper center',ncol=3,frameon=False);axes[0].grid(axis='y',alpha=.15)
delta=np.array([(TAIL['B'][c].mean()-TAIL['A'][c].mean())/3600000 for c,_ in parts]+[(TAIL['B'].total_system_energy_j.mean()-TAIL['A'].total_system_energy_j.mean())/3600000])
axes[1].bar(['空闲','动态计算','传输','总能耗'],delta,color=['#96A7BA','#3478AE','#E59958','#673D8D']);axes[1].axhline(0,color='#444',lw=.8)
for i,v in enumerate(delta):axes[1].text(i,v+(.025 if v>=0 else -.025),f'{v:+.3f}',ha='center',va='bottom' if v>=0 else 'top')
axes[1].set_title('B − A 的能耗分解 (kWh / 回合)');axes[1].set_ylim(-.67,.76);axes[1].grid(axis='y',alpha=.15)
fig.suptitle('图 8｜末 100 回合：较短仿真时长抵消了部分传输能耗增加',fontsize=15);save(fig,'08_energy_decomposition')

# Deliver a Chinese report with exact units, denominators and caveats.
heading('BCGH 两组训练日志对比分析',1)
para('数据：A = result/BCGH-MASAC/dis；B = result/LMD=1/BCGH/DIS。每组包含 0.35-2-8.csv 和 dc_log-0.35-2-8.csv。分析日期：2026-10-08。主要结论以末 100 回合（901–1000）为依据，同时检查末 200 回合及全部 Routing 训练期。')
heading('1. 主要结论')
para('A 的主要优势是减少重复路由与传输开销。末 100 回合，A 的训练回报为 378.18，B 为 151.69；A 每回合路由决策少 558.72 次（相对 B 少 12.45%），边缘跳数少 0.279 次/任务，迁入后 self 决策比例高 12.89 个百分点。B 的服务完成率和 SLA 满足率仅略低，单次运行内的配对区间不能清楚区分两组在这些服务指标上的表现。')
para('两组日志在 Routing 训练期都记录 bcgh_guidance_lambda=1，Host 预训练期都是 0；云端都关闭、总到达率都是 0.35、能耗权重都是 0.3。因此本分析比较两个运行结果，不能将差异归因于 λ 的取值，也不能从目录名推断消融配置。Host 预训练期间两组表现已经不完全相同，说明进入 Routing 阶段的状态也不是完全一致。')
addtable(SUMMARY.iloc[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,18,19,21]].reset_index(drop=True))
para('表中 B−A 为绝对差；比例指标的绝对差单位是百分点。相对差 = (B/A−1)×100%。回报是带惩罚的带符号综合指标，其相对百分比不能等同于任务成功收益的增幅。各项均为逐回合指标的算术平均；单位完成任务能耗不是对 100 回合重新合并后的比值。')
heading('2. 数据完整性与可比性')
para('每组 Episode 日志为 1000×207，DC 日志为 5000×63；回合编号连续，Episode 无重复，每回合恰有 DC-1 至 DC-5。任务守恒、DC 完成任务和路由决策汇总、全网迁入迁出守恒、能耗分解均核对通过。两组未完成任务数、因果终结缺口、期末待终结 trace 和能耗计时时间缺口均为 0。传输能耗的任务核算误差最大约 1.03×10⁻⁵ J，相比总量可忽略。')
para('逐回合 episode_seed（42–1041）、2000 个任务、所有数值 workload 汇总、总 CPU/GPU 容量完全一致；逐 DC 外生到达任务数也完全一致。末 100 回合 Host 明细中的主机标识及 CPU/GPU 容量一致。这里验证的是日志可见的种子、汇总及容量；没有完整的逐任务轨迹和历史版本快照，不能证明所有训练随机状态、模型初始化和内部配置均一致。')
para('缺失值集中在未训练的层：Routing 的学习指标在 1–200 回合为空；Host 的学习指标在 Routing 阶段为空，且预训练最初尚未更新时也为空。这些空值没有填成 0。Host alpha 在冻结期保持常量，并不表示 Host 仍在更新。')
heading('3. 训练过程：明显阶段收益，但没有第三阶段')
image('01_performance.png','图 1：灰色区域为 Host 预训练（1–200），其后为 Routing 训练（201–1000）。滑动均值在阶段边界重新计算，避免混合不同策略阶段。')
wf=[]
for lo in range(1,1001,100):
    r={'回合':f'{lo}–{lo+99}'}
    for k,d in DATA.items():
        t=d[d.episode.between(lo,lo+99)]
        for c,lab,scale in [('completion_rate','完成率 %',100),('sla_satisfaction_rate','SLA %',100),('training_episode_reward','回报',1),('avg_routing_edge_hops_per_job','跳数/任务',1)]:r[k+' '+lab]=t[c].mean()*scale
    wf.append(r)
addtable(pd.DataFrame(wf))
para('Host 预训练后半段（101–200）完成率约 55.7%，Routing 学习后多为 73%–75%，SLA 满足率从约 49.5% 提高到约 67%。这是训练阶段内的描述性变化，阶段切换同时改变了可用策略，不能当作单因素因果实验。预训练日志的 training_episode_reward 全为 0，但 host_layer_reward_sum 并非全为 0；该列在这一阶段不能用来判断 Host 策略是否有效。')
para('B 在 201–300 回合更快达到较高完成率（68.07%，A 为 63.72%），回报也较少为负；A 在后续多数窗口有更高回报。A 在 401–900 回合回报均值约 462–537，末 100 回合降到 378；B 在 601–700 回合回报为 337，末 100 回合降到 152。两组后期边缘跳数增加、迁入后的 self 比例下降，提示路由效率后期变差。不能仅依据服务指标平台或 loss 平稳就宣称已经收敛。')
heading('4. 路由差异：B 的更多转发没有带来更好的服务结果')
image('02_routing.png','图 2：路由比例以路由决策为分母，多跳比例和平均边缘跳数以全部任务/终结 trace 为分母。')
para('末 100 回合，B 的跨边缘路由比例 54.46%，A 为 47.91%；多跳任务比例 17.11% vs 14.16%；每任务平均边缘跳数 1.244 vs 0.965。B 的迁入后 self 决策比例为 42.09%，A 为 54.98%；继续边缘转发比例为 57.54% vs 44.71%。因此 B 更常把已转移到目标 DC 的任务再次转发。迁入后 self 只是下一次路由选择留在本 DC，不能解释为任务最终成功完成。')
para('按照当前工作区的奖励实现，跨边缘/云端路由会产生基础卸载惩罚、时延成本和预期 SLA 风险成本，终结奖励另计完成、SLA 及任务归属能耗。因此在最终服务率接近时，额外路由会拉低回报，这与 B 的更低回报一致。这是由日志与当前代码支持的机制解释；缺少当时运行的完整奖励配置和即时/终结奖励拆分，不能精确计算 226.49 的回报差有多少来自每一项惩罚。')
para('B 的 Host 立即启动比例略高（43.22% vs 41.92%），排队比例略低（55.60% vs 57.79%），但 Host 直接丢弃数更多（23.73 vs 5.84 / 回合）。这表明立即启动比例本身不足以评价整体策略。waiting_timeout_drops 在两组均为 0，而总丢弃仍约 524–536 / 回合；该字段只是特定等待队列超时类别，不能据此认为没有超时或失败。日志没有完整原因分类，未将剩余丢弃强行归入某一类别。')
heading('5. 数据中心与主机：净迁入接近，重复转发差异很大')
image('05_datacenters.png','图 5：所有柱图采用相同末 100 回合窗口；DC 平均负载是时间平均值。')
dcview=[]
for idx in IDS:
    a,b=DCMEAN['A'].loc[idx],DCMEAN['B'].loc[idx]
    dcview.append({'DC':idx,'外生任务/回合':a.exogenous_arrival_count,'CPU 容量':a.dc_cpu_capacity,'GPU 容量':a.dc_gpu_capacity,'A 完成':a.dc_completed_jobs,'B 完成':b.dc_completed_jobs,'A 净迁入':a.net_edge_migration_count,'B 净迁入':b.net_edge_migration_count,'A 迁入 self %':a.incoming_edge_absorption_rate*100,'B 迁入 self %':b.incoming_edge_absorption_rate*100})
addtable(pd.DataFrame(dcview))
image('06_route_matrix.png','图 6：行是路由源 DC，列是目标 DC，对角线代表 self。迁移和路由计数不是唯一任务数，不能作为最终任务归属矩阵。')
para('DC-1/2/3 的外生到达负荷较高，平均每回合约 598/602/399 个；DC-4/5 各约 200 个。两组都把大量任务转向 DC-5，净迁入 A 为 625.47、B 为 629.15，非常接近。但 DC-5 总路由决策 B 为 1519.93，A 为 1217.29；B 在 DC-5 的迁入次数更多（1319.59 vs 1016.95），迁出次数也更多（690.44 vs 391.48）。净流量掩盖了双向重复迁移的开销。日志支持更多重复转发，不能在没有逐任务轨迹时认定每次都是闭环循环。')
para('B 在 DC-5 每回合多完成 27.81 个任务，但 DC-1、DC-2、DC-4 分别少完成 13.20、19.53、12.05 个，DC-3 多完成 5.15 个；全网每回合净少完成 11.82 个。两组平均 CPU 负载约 20%，GPU 约 41%；DC-1/2/3 的 GPU 均值约 53%–56%，DC-5 约 22%。不能用这些平均值断言所有任务都有空闲可用资源，异构主机容量、任务资源组合及瞬时峰值都可能限制分配。')
addtable(hostsummary.rename(columns={'model':'组别','dc_id':'DC','hosts':'主机数','zero_cpu_share':'零 CPU 负载主机回合占比','zero_gpu_share':'零 GPU 负载主机回合占比','host_cpu_p95':'主机 CPU 均值 P95','host_gpu_p95':'主机 GPU 均值 P95','host_busy_mean':'主机平均忙碌比例'}))
para('上表 P95 针对末 100 回合所有“主机×回合”的时间平均负载；零负载占比也采用主机回合为分母，不是永远未使用主机的比例。主机明细说明低总体均值与个别主机较高负载可以同时出现，Host 预训练后的分配策略值得单独检查。')
heading('6. 能耗：总能耗接近，传输代价有明确差异')
image('03_energy.png','图 3：系统能耗和任务归属能耗采用不同核算口径；实际运行时间受硬件、同时运行的进程和训练更新开销影响。')
image('08_energy_decomposition.png','图 8：系统总能耗 = 边缘空闲 + 系统动态计算 + 传输（两组云计算均为 0）。')
para('末 100 回合，系统总能耗 A 为 93.520 kWh，B 为 93.769 kWh，B 仅高 0.267%。但传输能耗为 2.096 vs 2.690 kWh，B 高 28.34%；其每任务归属能耗为 77.409 kJ，A 为 76.260 kJ，B 高 1.51%。B 的平均仿真结束时间短约 137.65 s，降低空闲能耗 0.516 kWh，抵消了部分传输增加的 0.594 kWh 和动态计算增加的 0.172 kWh。空闲能耗占系统总量约 57%–58%，解释了为什么总能耗差异很小。')
para('单位完成任务系统能耗 A 为 228.747 kJ，B 为 231.184 kJ。该值同时受总能耗和完成数量影响；完成数少也会使其上升。平均完成时间仅统计最终完成的任务，丢弃任务不进入该均值，因此应同时查看完成率和 SLA 满足率。当前任务归属计算能耗按任务/主机模型计算，与按时间积分的系统动态计算不是相同口径；没有把两者强行对账。')
para('实际运行时间 A 为 76.35 s/回合，B 为 87.23 s/回合，B 高 14.24%。该开销与更多路由一致，但运行时硬件和并发环境未知，因此它只是这两份日志的实际耗时差，不能作为严格的算法速度基准。')
heading('7. 学习指标及后期稳定性')
image('04_learning.png','图 4：Host 在第 201 回合起更新次数恒为 0。两份日志都没有 joint_finetune 阶段。')
para('末 100 回合，B 的 Routing Critic loss 更低（2.744 vs 3.106），但 Actor loss 更高（1.125 vs 0.775），回报更低。Critic loss 反映对各自目标样本的拟合误差，样本分布和目标回报不同，不能直接作为跨运行的策略优劣评分。两组 Routing 策略熵都约 0.322，目标熵约 0.32189；接近目标只表示熵约束基本满足，不保证动作分布质量或训练已收敛。')
para('Host 的 alpha 在后 800 回合恒定，分别约 0.05180 和 0.04824；其 Critic、Actor、熵指标为空且更新次数为 0，与 Host 冻结一致。路由将改变流入各 DC 的任务分布，而 Host 未在新分布下继续学习；这是值得后续验证的解释，但当前日志不足以认定冻结就是低完成率或差异的唯一原因。')
para('末 100 回合训练回报标准差：A '+fmt(TAIL['A'].training_episode_reward.std())+'，B '+fmt(TAIL['B'].training_episode_reward.std())+'；完成率标准差：A '+fmt(TAIL['A'].completion_rate.std()*100)+'、B '+fmt(TAIL['B'].completion_rate.std()*100)+' 个百分点。两组都有明显逐回合波动，单个最好/最后回合不适合做结论。')
heading('8. 配对差值与不确定性：服务率优势尚不能确定')
image('07_paired_differences.png','图 7：只重采样末 100 回合的配对差值序列，不把 A/B 分开重采样。区间表示这一次运行内、给定窗口与块长假设下的不确定性。')
view=[]
for c in ['training_episode_reward','completion_rate','sla_satisfaction_rate','avg_completion_time','total_system_energy_kwh','system_energy_per_completed_job_j','routing_decision_count','avg_routing_edge_hops_per_job','edge_target_absorption_rate','transfer_energy_j']:
    for length in [10,20,50]:
        r=CIS[(CIS.metric==c)&(CIS.window=='末 100 回合')&(CIS.block_length==length)].iloc[0]
        lab,scale=LABEL[c]
        view.append({'指标':lab,'连续块长度':length,'均值 B−A':r.B_minus_A*scale,'95% 下界':r.ci95_low*scale,'95% 上界':r.ci95_high*scale})
addtable(pd.DataFrame(view))
para('使用移动连续块 bootstrap（6000 次，固定随机种子 20261008），保留部分序列相关性，分别检查 10/20/50 回合块长。20 回合块的完成率、SLA 满足率、平均周转时间和总能耗差值区间跨 0；路由决策、边缘跳数、迁入 self 比例、传输能耗以及回报差异方向更清楚。不能因此宣称方法跨随机种子显著胜出：每组只有一次完整训练，回合不是独立训练重复，训练后期仍有变化；区间仅是本次日志的描述性辅助，也未作多重指标检验。')
para('50 回合块下，完成率、SLA 满足率和总能耗的区间没有跨 0，而 10/20 回合块下跨 0，说明这些小差异对重采样设定敏感。100 回合窗口采用 50 回合块时每次仅拼接两个块，信息有限；不选择其中某一个块长作为显著性证据。路由次数、传输开销和回报的差异在三个块长下方向一致。')
view=[]
for w in ['末 100 回合','末 200 回合','Routing 训练']:
    for c in ['training_episode_reward','completion_rate','sla_satisfaction_rate','routing_decision_count','total_system_energy_kwh']:
        r=CIS[(CIS.metric==c)&(CIS.window==w)&(CIS.block_length==20)].iloc[0];lab,scale=LABEL[c]
        view.append({'窗口':w,'指标':lab,'A':r.A_mean*scale,'B':r.B_mean*scale,'B−A':r.B_minus_A*scale})
addtable(pd.DataFrame(view))
para('全 Routing 阶段 B 的完成率略高于 A，而后期窗口 A 略高；B 更快的初期改善与后期 A 的小幅优势同时存在。路由决策和回报差异在末 100/200 回合及 Routing 全阶段都保持相同主要方向。为避免阶段 1 的 0 回报和固定 self 路由稀释结果，不使用 1–1000 的整体平均作为主结论。')
heading('9. 下一步实验与排查优先级')
para('第一，核实两次运行实际配置和代码版本：两组 λ 都是 1，应保存配置快照、提交号、完整训练随机种子、初始化/恢复 checkpoint 信息及奖励参数，确认预期修改是否真正生效。第二，针对重复迁移增加可追踪诊断：按唯一任务记录路径、再访问次数、最终落地 DC，以及即时路由成本/终结奖励拆分，重点查看 DC-5 的大规模双向流量。第三，使用同一预训练 Host checkpoint 比较 Routing 变体，避免 Host 初始差异干扰判断；独立重复多个完整训练种子并在固定评估集上评估。')
para('第四，可分别验证迁入后的继续转发惩罚、落地吸收反馈、限制回访/跳数及后期引导退火；它们是由当前现象提出的实验方向，不是已证明有效的改进。第五，检查 Host 在新到达分布下的适配情况，再比较冻结与联合微调；不要同时更改路由和 Host 多项设置，否则难以定位收益来源。')
heading('10. 指标口径与可复现材料')
para('完成率 = 完成任务 / 全部任务；SLA 满足率 = SLA 满足任务 / 全部任务，丢弃计入 SLA 违规。周转时间 = 完成任务从到达到完成的时间均值。跨边缘路由率 = edge 决策 / 全部路由决策；Host 启动/排队/丢弃率的分母是 Host 决策数；DC 迁入后 self 比例 = incoming self / incoming successor 决策。DC 净迁入 = route_in_edge_count−route_out_edge_count，重复迁移可重复计数。')
para('统计表：tail100_summary.csv、window_statistics.csv、paired_block_bootstrap.csv、dc_tail100.csv、host_tail100.csv、routing_matrix_A/B.csv；完整性检查记录见 audit.json。所有图同时提供 PNG 与 SVG；HTML 报告内嵌全部图，可直接打开或独立分享。原始日志未改动。')
para('字段解释核对当前工作区 schedulers/BCGH-MASAC/training_support.py 的奖励函数、SLA 统计、能耗统计、Episode/DC 行构建，以及 environment/energy_model.py。当前代码用于解释可见字段，不能替代缺失的运行历史快照。')
heading('原始数据路径')
for k,p in PATHS.items():
    para(k+' Episode：'+str(p));para(k+' DC：'+str(p.with_name('dc_log-'+p.name)))
(OUT/'详细分析.md').write_text('\n\n'.join(x[0] for x in BLOCKS),encoding='utf-8')
css='body{font-family:"Microsoft YaHei",sans-serif;color:#223244;max-width:1280px;margin:32px auto;padding:0 28px;line-height:1.8}h1{font-size:28px}h2{font-size:21px;margin-top:36px;border-bottom:1px solid #cbd6e0;padding-bottom:8px}p{margin:14px 0}img{width:100%;height:auto}figure{margin:24px 0}figcaption{color:#536579;font-size:14px}.table-wrap{overflow-x:auto}table{border-collapse:collapse;white-space:nowrap;width:100%;font-size:14px}th{background:#213b55;color:white}td,th{padding:8px 10px;text-align:right;border-bottom:1px solid #dde4eb}td:first-child,th:first-child{text-align:left}tr:nth-child(even){background:#f4f7fa}@media print{body{max-width:none;padding:0}figure,table{break-inside:avoid}}'
(OUT/'详细分析.html').write_text('<!DOCTYPE html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>BCGH 两组日志对比分析</title><style>'+css+'</style><body>'+''.join(x[1] for x in BLOCKS)+'</body></html>',encoding='utf-8')
print('OUTPUT',OUT)
print('TAIL_SUMMARY',SUMMARY.iloc[[0,1,2,3,4,6,9,10,11,14,19,21]].round(5).to_json(orient='records',force_ascii=False))
print('BOOTSTRAP',CIS[(CIS.window=='末 100 回合')&(CIS.block_length==20)&CIS.metric.isin(['training_episode_reward','completion_rate','sla_satisfaction_rate','avg_completion_time','total_system_energy_kwh','routing_decision_count','transfer_energy_j'])][['metric','B_minus_A','ci95_low','ci95_high']].to_json(orient='records'))
print('HOST_CAPACITY_EQUAL',host_capacity_equal)
