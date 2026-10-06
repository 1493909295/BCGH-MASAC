"""Compare the supplied training logs without changing source files."""
from pathlib import Path
import json
import math
import hashlib
import numpy as np
import pandas as pd
from reportlab.graphics.shapes import Drawing, String, Line, Rect
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics import renderSVG
from reportlab.lib.colors import HexColor, Color

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'result/BCGH-MASAC/0.35-2000-cloud-异构/不同的三阶段'
OUT = ROOT / 'analysis_outputs/routing_alignment_20261005'
OUT.mkdir(exist_ok=True)
NAMES = ['200-200-700', '200-800', '500-500']
COLORS = ['#2563EB', '#D97706', '#059669']
METRICS = ['training_episode_reward','completion_rate','sla_satisfaction_rate','avg_completion_time',
 'total_system_energy_kwh','system_energy_per_completed_job_j','task_attributable_energy_per_total_job_j',
 'routing_self_rate','routing_edge_rate','routing_cloud_rate','multi_hop_job_rate',
 'host_started_rate','host_queued_rate','host_dropped_rate','bcgh_guidance_lambda',
 'routing_episode_updates','host_episode_updates','routing_critic_loss','host_critic_loss',
 'routing_alpha','host_alpha_mean','routing_policy_entropy','host_policy_entropy',
 'routing_decision_count','routing_random_action_count','routing_policy_action_count',
 'routing_update_step','routing_training_action_steps','wall_time_seconds',
 'completed_jobs','sla_satisfied_jobs','total_jobs','simulation_end_time',
 'waiting_timeout_drops','queued_jobs','edge_avg_cpu_load','edge_avg_gpu_load']
data, boundaries, checks = {}, [], {}
for name in NAMES:
    path = SRC / f'{name}.csv'
    d = pd.read_csv(path).copy()
    assert d.episode.is_unique and np.array_equal(d.episode, np.arange(1, len(d)+1))
    stage_start = int(d.loc[d.training_stage.eq('routing_train'),'episode'].min())
    update_start = int(d.loc[d.routing_episode_updates.gt(0),'episode'].min())
    assert stage_start == update_start
    d['routing_relative_episode'] = d.episode - stage_start
    d['routing_training_round'] = d.routing_relative_episode + 1
    d['experiment'] = name
    d['routing_cumulative_updates'] = d.routing_episode_updates.cumsum()
    data[name] = d
    for stage,x in d.groupby('training_stage',sort=False):
        boundaries.append({'experiment':name,'stage':stage,'original_start':int(x.episode.min()),
         'original_end':int(x.episode.max()),'rounds':len(x),
         'relative_start':int(x.routing_relative_episode.min()),'relative_end':int(x.routing_relative_episode.max()),
         'lambda_start':float(x.bcgh_guidance_lambda.iloc[0]),'lambda_end':float(x.bcgh_guidance_lambda.iloc[-1])})
    assert np.allclose(d.completion_rate, d.completed_jobs/d.total_jobs)
    assert np.allclose(d.sla_satisfaction_rate, d.sla_satisfied_jobs/d.total_jobs)
    assert np.allclose(d.total_system_energy_kwh, d.total_system_energy_j/3600000)
    assert np.allclose(d.training_episode_reward, d.routing_layer_reward_sum)
    assert (d[['unresolved_jobs','pending_trace_count_end','causal_terminal_job_gap',
               'workload_missing_origin_count','workload_unknown_origin_count']] == 0).all().all()
    checks[name] = {'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                   'source_rows':len(d),'routing_start':stage_start,'routing_rounds':int((d.routing_relative_episode>=0).sum()),
                   'first_seed':int(d.loc[d.episode.eq(stage_start),'episode_seed'].iloc[0]),
                   'missing_learning_metrics':d[['routing_critic_loss','host_critic_loss','routing_policy_entropy','host_policy_entropy']].isna().sum().to_dict()}

COMMON_END = min(int(d.routing_relative_episode.max()) for d in data.values())
assert COMMON_END == 499
windows = [(-50,-1),(0,49),(50,99),(100,199),(150,199),(200,249),(250,299),
           (200,299),(300,399),(400,499),(0,499),(500,599),(600,699),(700,799)]
rows=[]
for name,d in data.items():
    for a,b in windows:
        x=d.loc[d.routing_relative_episode.between(a,b)]
        if len(x) != b-a+1: continue
        row={'experiment':name,'window':f'{a}:{b}','relative_start':a,'relative_end':b,'n':len(x),
             'original_start':int(x.episode.min()),'original_end':int(x.episode.max()),
             'stages':','.join(x.training_stage.unique())}
        for c in METRICS:
            row[c+'_mean']=float(x[c].mean()) if x[c].notna().any() else None
            row[c+'_sd']=float(x[c].std(ddof=1)) if x[c].notna().sum()>1 else None
        rows.append(row)
    x=d.loc[d.routing_relative_episode.ge(0)].tail(100)
    rows.append({'experiment':name,'window':'last100','relative_start':int(x.routing_relative_episode.min()),
                 'relative_end':int(x.routing_relative_episode.max()),'n':len(x),
                 'original_start':int(x.episode.min()),'original_end':int(x.episode.max()),
                 'stages':','.join(x.training_stage.unique()),
                 **{c+'_mean':float(x[c].mean()) if x[c].notna().any() else None for c in METRICS},
                 **{c+'_sd':float(x[c].std(ddof=1)) if x[c].notna().sum()>1 else None for c in METRICS}})
summary=pd.DataFrame(rows)
summary.to_csv(OUT/'window_summary.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(boundaries).to_csv(OUT/'stage_boundaries.csv',index=False,encoding='utf-8-sig')
aligned=pd.concat([d[['experiment','episode','episode_seed','training_stage',
                     'routing_relative_episode','routing_training_round','routing_cumulative_updates']+METRICS]
                    for d in data.values()],ignore_index=True)
aligned.to_csv(OUT/'routing_aligned_metrics.csv',index=False,encoding='utf-8-sig')

def stats(name,a,b,c):
    x=data[name].loc[data[name].routing_relative_episode.between(a,b),c]
    return x.mean(),x.std(ddof=1)

def nice_axis(values):
    lo=float(np.min(values));hi=float(np.max(values))
    span=max(hi-lo,abs(hi)*0.03,1e-6)
    raw=span/5
    p=10**math.floor(math.log10(raw))
    step=next(k*p for k in [1,2,2.5,5,10] if k*p>=raw)
    return math.floor((lo-span*.035)/step)*step,math.ceil((hi+span*.035)/step)*step,step

def fig(specs, filename, heading, subtitle):
    drawing=Drawing(1460,1060)
    drawing.add(Rect(0,0,1460,1060,fillColor=HexColor('#FFFFFF'),strokeColor=None))
    drawing.add(String(45,1020,heading,fontName='Helvetica-Bold',fontSize=22,fillColor=HexColor('#172033')))
    drawing.add(String(45,993,subtitle,fontName='Helvetica',fontSize=13,fillColor=HexColor('#526077')))
    for i,(name,color) in enumerate(zip(NAMES,COLORS)):
        x=45+i*245
        drawing.add(Line(x,967,x+32,967,strokeColor=HexColor(color),strokeWidth=3))
        drawing.add(String(x+41,962,name,fontName='Helvetica-Bold',fontSize=13,fillColor=HexColor('#172033')))
    drawing.add(String(830,962,'Thin: raw episodes    Thick: trailing 20-episode mean',fontName='Helvetica',fontSize=12,fillColor=HexColor('#526077')))
    for j,(metric,title,scale) in enumerate(specs):
        col=j%3;row=j//3
        x=75+col*477;y=550-row*438;w=403;h=314
        drawing.add(String(x,y+h+49,title,fontName='Helvetica-Bold',fontSize=15,fillColor=HexColor('#172033')))
        all_values=[]; series=[]
        for name in NAMES:
            d=data[name].loc[data[name].routing_relative_episode.ge(0)]
            values=d[metric]*scale
            smooth=values.rolling(20,min_periods=20).mean()
            all_values.extend(values.dropna().tolist())
            series.append([(float(t),float(v)) for t,v in zip(d.routing_relative_episode,values) if np.isfinite(v)])
            series.append([(float(t),float(v)) for t,v in zip(d.routing_relative_episode,smooth) if np.isfinite(v)])
        lo,hi,step=nice_axis(all_values)
        if scale == 100:
            lo=max(0,lo);hi=min(100,hi)
        if metric == 'bcgh_guidance_lambda':
            lo,hi,step=0,1,.2
        xmax=807
        drawing.add(Rect(x+w*499/xmax,y,w*(xmax-499)/xmax,h,fillColor=HexColor('#F0F2F5'),strokeColor=None))
        for tick in np.arange(lo,hi+step*.1,step):
            yy=y+h*(tick-lo)/(hi-lo)
            drawing.add(Line(x,yy,x+w,yy,strokeColor=HexColor('#DEE3EA'),strokeWidth=.6))
        lp=LinePlot();lp.x=x;lp.y=y;lp.width=w;lp.height=h;lp.data=series;lp.joinedLines=1
        lp.xValueAxis.valueMin=0;lp.xValueAxis.valueMax=xmax;lp.xValueAxis.valueStep=200
        lp.yValueAxis.valueMin=lo;lp.yValueAxis.valueMax=hi;lp.yValueAxis.valueStep=step
        lp.xValueAxis.labels.fontName='Helvetica';lp.yValueAxis.labels.fontName='Helvetica'
        lp.xValueAxis.labels.fontSize=10;lp.yValueAxis.labels.fontSize=10
        lp.xValueAxis.strokeColor=HexColor('#8894A5');lp.yValueAxis.strokeColor=HexColor('#8894A5')
        lp.yValueAxis.labelTextFormat='%g'
        for k,color in enumerate(COLORS):
            c=HexColor(color)
            lp.lines[2*k].strokeColor=Color(.78+.22*c.red,.78+.22*c.green,.78+.22*c.blue)
            lp.lines[2*k].strokeWidth=.55
            lp.lines[2*k+1].strokeColor=c;lp.lines[2*k+1].strokeWidth=2.3
        drawing.add(lp)
        for t,color,dash in [(200,COLORS[0],[5,3]),(499,'#6B7280',[2,3])]:
            xx=x+w*t/xmax
            drawing.add(Line(xx,y,xx,y+h,strokeColor=HexColor(color),strokeWidth=.9,strokeDashArray=dash))
        drawing.add(String(x+w/2,y-40,'Episodes since routing start (first = 0)',textAnchor='middle',fontSize=11,fillColor=HexColor('#526077')))
    drawing.add(String(45,46,'Blue dashed line at 200: 200-200-700 begins joint training. Grey region after 499: 500-500 has no observations.',fontSize=12,fillColor=HexColor('#526077')))
    drawing.add(String(45,25,'All runs: 2,000 jobs per episode, arrival rate 0.35, heterogeneous arrivals, cloud enabled. Curves describe single training runs.',fontSize=12,fillColor=HexColor('#526077')))
    svg_path=OUT/f'{filename}.svg'
    renderSVG.drawToFile(drawing,str(svg_path))
    svg=svg_path.read_text(encoding='utf-8')
    svg=svg.replace('font-family: Helvetica-Bold;', 'font-family: Arial; font-weight: bold;')
    svg=svg.replace('font-family: Helvetica;', 'font-family: Arial;')
    svg_path.write_text(svg,encoding='utf-8')

fig([('completion_rate','Completion rate (%)',100),('sla_satisfaction_rate','SLA satisfaction (%)',100),
     ('avg_completion_time','Mean turnaround of completed jobs (s)',1),
     ('total_system_energy_kwh','Total system energy per episode (kWh)',1),
     ('training_episode_reward','Routing training return',1),('bcgh_guidance_lambda','Heuristic guidance coefficient',1)],
    'routing_aligned_performance','Training performance aligned at routing start',
    'Common comparison: relative episodes 0-499. Same routing episode count; stage and guidance schedules differ.')
fig([('routing_self_rate','Local routing decisions (%)',100),('routing_edge_rate','Edge routing decisions (%)',100),
     ('routing_cloud_rate','Cloud routing decisions (%)',100),('multi_hop_job_rate','Jobs with multiple edge hops (%)',100),
     ('host_started_rate','Immediate starts among host decisions (%)',100),('host_queued_rate','Queued among host decisions (%)',100)],
    'routing_aligned_behavior','Routing and host behavior aligned at routing start',
    'Routing percentages use decision counts. Host percentages use host decision counts. These are not job outcome shares.')

def table(window, fields, sd=False):
    lines=['| 配置 | '+' | '.join(label for _,label,_ in fields)+' |',
           '|---|'+'---:|'*len(fields)]
    for name in NAMES:
        r=summary.loc[(summary.experiment==name)&(summary.window==window)].iloc[0]
        cells=[]
        for metric,label,scale in fields:
            value=r[metric+'_mean']*scale
            digits=2 if scale==100 else (1 if metric in ['training_episode_reward','avg_completion_time'] else 2)
            s=f'{value:.{digits}f}'
            if sd:s+=f' ± {r[metric+"_sd"]*scale:.{digits}f}'
            cells.append(s)
        lines.append('| '+name+' | '+' | '.join(cells)+' |')
    return '\n'.join(lines)

fields=[('completion_rate','完成率（%）',100),('sla_satisfaction_rate','SLA 满足率（%）',100),
        ('avg_completion_time','平均完成时间（秒）',1),('total_system_energy_kwh','系统能耗（kWh/轮）',1),
        ('training_episode_reward','Routing 回报',1)]
behavior=[('routing_self_rate','本地决策（%）',100),('routing_edge_rate','边缘决策（%）',100),
          ('routing_cloud_rate','云端决策（%）',100),('multi_hop_job_rate','多跳任务（%）',100),
          ('host_started_rate','Host 立即启动（%）',100),('host_queued_rate','Host 排队（%）',100)]
boundary_lines=['| 配置 | Host 预训练 | Routing 单独训练 | 联合训练 | routing 总轮数 |',
                '|---|---|---|---|---:|']
for name in NAMES:
    spans=[]
    for stage in ['host_pretrain','routing_train','joint_finetune']:
        x=data[name].loc[data[name].training_stage.eq(stage)]
        spans.append(f'{int(x.episode.min())}–{int(x.episode.max())}（{len(x)} 轮）' if len(x) else '未记录')
    boundary_lines.append('| '+name+' | '+' | '.join(spans)+f' | {checks[name]["routing_rounds"]} |')

convergence=[]
for name,d in data.items():
    x=d.loc[d.routing_relative_episode.between(0,499)]
    for metric,threshold in [('completion_rate',.97),('sla_satisfaction_rate',.93)]:
        good=x[metric].rolling(20,min_periods=20).mean().ge(threshold)
        consecutive=good.rolling(50,min_periods=50).sum().eq(50)
        ends=x.loc[consecutive,'routing_relative_episode']
        start=int(ends.iloc[0]-49) if len(ends) else None
        convergence.append({'experiment':name,'metric':metric,'threshold':threshold,
                            'first_relative_episode_of_50_consecutive_good_ma20':start})

A,B,C=NAMES
ar={c:stats(A,400,499,c)[0] for c in METRICS}
br={c:stats(B,400,499,c)[0] for c in METRICS}
cr={c:stats(C,400,499,c)[0] for c in METRICS}
deltaA=(br['sla_satisfaction_rate']-ar['sla_satisfaction_rate'])*100
deltaC=(br['sla_satisfaction_rate']-cr['sla_satisfaction_rate'])*100
speedA=ar['avg_completion_time']-br['avg_completion_time']
speedC=cr['avg_completion_time']-br['avg_completion_time']
report=f'''# Routing 开始轮次对齐分析

## 结论

在相同的 routing 训练进度下，三组完成率接近。以共同区间最后 100 轮（相对轮次 400–499，routing 第 401–500 轮）衡量，200-800 的 SLA 满足率最高、平均完成时间最短；200-200-700 的系统总能耗略低于 200-800，差距约 0.1%。500-500 初期学习更快，延长 Host 预训练的优势没有持续到后期。200-200-700 的联合训练没有在已记录区间表现出持续的 SLA／时延优势。

这只描述当前三次训练日志。阶段长度、Host 是否继续更新及启发式引导进度同时变化，不能把结果解释为某一个因素的独立因果效果。

## 对齐定义与实际阶段

以日志中第一个 `routing_train` episode 为零点，定义 `routing_relative_episode = episode − routing_start_episode`，`routing_training_round = routing_relative_episode + 1`。三份日志的第一轮 routing 更新也发生在该轮。200-200-700、200-800 的起点是 201，500-500 的起点是 501。

{chr(10).join(boundary_lines)}

200-200-700 的文件名是实验配置标签；实际只记录了 608 轮联合训练，不能视为已经完成 700 轮。另两份日志尚无联合训练记录。200-200-700 的联合阶段在相对轮次 200 开始。共同有效 routing 区间为 0–499，共 500 轮。对应前两组原始轮次 201–700，以及 500-500 的原始轮次 501–1000。

图中细线为每轮原始值，粗线为向后 20 轮移动平均，移动平均在 routing 开始后重新计算，不混入预训练。蓝色虚线表示 200-200-700 进入联合训练，灰色区间表示 500-500 已无观测。图中不插值、不延长短日志。

![对齐后的性能曲线](routing_aligned_performance.png)

## 相同 routing 预算下的结果

共同区间最后 100 轮，相对轮次 400–499。表中的 ± 是这 100 轮的样本标准差，反映轮间波动，不是独立重复实验的置信区间。

{table('400:499',fields,sd=True)}

- 200-800 对比 200-200-700：SLA 高 {deltaA:.2f} 个百分点，平均完成时间少 {speedA:.1f} 秒（{speedA/ar['avg_completion_time']*100:.2f}%），系统能耗多 {br['total_system_energy_kwh']-ar['total_system_energy_kwh']:.2f} kWh/轮。完成率差 {(br['completion_rate']-ar['completion_rate'])*100:+.2f} 个百分点。
- 200-800 对比 500-500：SLA 高 {deltaC:.2f} 个百分点，平均完成时间少 {speedC:.1f} 秒（{speedC/cr['avg_completion_time']*100:.2f}%），系统能耗少 {cr['total_system_energy_kwh']-br['total_system_energy_kwh']:.2f} kWh/轮。完成率差 {(br['completion_rate']-cr['completion_rate'])*100:+.2f} 个百分点。
- 因此，按这一共同训练预算，200-800 的优势主要在 SLA 和时延，不能称为完成率全面领先。能耗上的小差异不足以据单次运行确立稳健排名。

前 500 轮整体均值用于观察整段学习过程，包括初期探索损失：

{table('0:499',fields)}

## 初期学习速度

Routing 前 50 轮，相对轮次 0–49：

{table('0:49',fields)}

Routing 第 51–100 轮，相对轮次 50–99：

{table('50:99',fields)}

500-500 在这两个初期窗口的完成率、SLA 和 routing 回报都最高，说明当前日志中 500 轮 Host 预训练对应更快的初期 routing 学习。起点前 50 轮 Host 预训练的完成率仍约 56%–57%，延长预训练没有在仅本地执行的阶段带来类似 routing 开始后的跃升。这里的初期优势也可能包含随机初始化、到达负载和引导进度的影响。

## 联合训练后的变化

200-200-700 在相对轮次 200 进入联合训练。为观察过渡，比较它在切换前 50 轮、切换后首 50 轮，以及接下来的 50 轮：

| 区间 | 完成率（%） | SLA（%） | 平均完成时间（秒） | Host 立即启动（%） | Host 排队（%） |
|---|---:|---:|---:|---:|---:|
'''
for a,b in [(150,199),(200,249),(250,299),(300,399),(400,499),(700,799)]:
    vals=[stats(A,a,b,m)[0]*s for m,s in [('completion_rate',100),('sla_satisfaction_rate',100),('avg_completion_time',1),('host_started_rate',100),('host_queued_rate',100)]]
    report+=f'| {a}–{b} | '+' | '.join(f'{v:.2f}' for v in vals)+' |\n'
report+=f'''
联合训练切换后没有出现完成率崩溃。随后 Host 立即启动比例下降、排队比例上升，云端 routing 决策比例增加，同时 SLA 和时延没有持续改善。这些走势与联合更新带来的策略变化相伴出现，但引导系数也在同期下降，无法仅凭日志把变化归因于 Host 更新。

共同区间最后 100 轮的行为比较：

{table('400:499',behavior)}

![对齐后的策略行为](routing_aligned_behavior.png)

## 后续走势与各自最后 100 轮

200-200-700 与 200-800 可继续比较到相对轮次 799，且每个相对轮次对应相同的 workload seed。相对轮次 700–799，200-200-700 的完成率约 {stats(A,700,799,'completion_rate')[0]*100:.2f}%，200-800 为 {stats(B,700,799,'completion_rate')[0]*100:.2f}%；SLA 分别为 {stats(A,700,799,'sla_satisfaction_rate')[0]*100:.2f}% 和 {stats(B,700,799,'sla_satisfaction_rate')[0]*100:.2f}%；平均完成时间分别为 {stats(A,700,799,'avg_completion_time')[0]:.1f} 秒和 {stats(B,700,799,'avg_completion_time')[0]:.1f} 秒。前者总能耗稍低，但 SLA／时延差距仍存在。末尾额外 8 轮只保留在曲线和逐轮数据，不作为完整 100 轮窗口。

各自最后 100 轮如下。其 routing 预算不同，仅用于描述日志结尾，不能替代同进度比较：

{table('last100',fields)}

对应相对区间：200-200-700 为 708–807，200-800 为 700–799，500-500 为 400–499。

## 指标口径与可比性

1. SLA 满足率为 SLA 满足任务数／全部任务数，丢弃任务不会被排除。平均完成时间只统计已完成任务的 turnaround time，包含其等待等时间。完成率与 SLA 一起看，避免仅比较成功任务的耗时。
2. `training_episode_reward` 在三份文件中逐轮等于 `routing_layer_reward_sum`。它是 routing 训练回报。Host 预训练阶段该字段为零，不能解释为系统没有收益；也不把 routing 和 host 的回报相加构造新的系统回报。
3. Routing 的本地／边缘／云端比例分母是 routing 决策次数。任务可被重复路由，所以这些比例不是最终任务去向或云端完成任务占比。Host 立即启动／排队比例的分母是 host 决策次数，排队并不等同于最终失败。
4. 共同最后 100 轮的平均引导系数分别为 {ar['bcgh_guidance_lambda']:.4f}、{br['bcgh_guidance_lambda']:.4f}、{cr['bcgh_guidance_lambda']:.4f}。200-200-700 的 λ 在相对轮次 199 升至 1，再随联合阶段下降；200-800 在相对轮次 799 才升至 1；500-500 在 499 升至 1。因此对齐轮次后，引导强度仍然不同。
5. 三份日志在相同原始 episode 的 workload seed 和所有数值型 `workload_*` 字段一致。对齐后前两组仍是相同 workload；500-500 的 workload seed 相比前两组偏移 300，不能按对齐后的每一轮进行同负载配对差值检验。
6. 每轮均为 2000 个任务，总到达率 0.35，异构到达，cloud 启用，能耗权重 0.3，能耗归一化常数 170000 J。没有重复或缺失 episode，未决任务、pending trace、causal terminal gap、未知／缺失来源均为零。学习损失／熵的 NaN 主要出现在该学习器未更新的阶段，保留为缺失，不填零。
7. 对齐的是训练轮次，而非真实时间或梯度更新次数。相对 0–499 累积 routing 更新分别为 {int(data[A].loc[data[A].routing_relative_episode.between(0,499),'routing_episode_updates'].sum()):,}、{int(data[B].loc[data[B].routing_relative_episode.between(0,499),'routing_episode_updates'].sum()):,}、{int(data[C].loc[data[C].routing_relative_episode.between(0,499),'routing_episode_updates'].sum()):,} 次。每轮 routing 决策数不同，所以更新预算也有小幅差异。CSV 没有训练开始时间戳；这里的“开始时间对齐”使用 routing 首轮作为横轴零点。
8. 当前只有每个配置的一次训练运行。轮间标准差包含负载与训练过程波动，不能代表跨随机种子的实验稳定性，也不据此宣称统计显著或测试集性能。

## 后续实验建议

如果要选当前已观察到的训练安排，200-800 是 SLA／时延表现较好的参照方案。下一步可固定共同的 workload seed 序列、相同 routing 更新预算及独立于阶段的 λ 变化曲线，再比较 200 与 500 轮 Host 预训练、以及 Host 冻结与联合更新。用多个独立训练随机种子，并对保存的策略进行同负载的确定性评估，才能判断阶段轮数的稳健影响。

## 输出说明

- `routing_aligned_metrics.csv`：保留原始轮次、seed、阶段，以及相对轮次、routing 轮号和分析所需指标。
- `window_summary.csv`：所有完整窗口的均值和样本标准差，及各自最后 100 轮。
- `stage_boundaries.csv`：由实际日志提取的阶段边界。
- 两组 SVG／PNG 图：性能曲线与策略行为曲线。
- `verification.json`：数据口径核对和原始文件 SHA-256，用于复现溯源。

数据来源：200-200-700.csv、200-800.csv、500-500.csv，均位于用户提供的“不同的三阶段”目录。原始日志未修改。
'''
(OUT/'analysis_report.md').write_text(report,encoding='utf-8')
(OUT/'verification.json').write_text(json.dumps({'checks':checks,'common_relative_interval':[0,COMMON_END],
 'boundary_records':boundaries,'convergence_descriptive':convergence},ensure_ascii=False,indent=2),encoding='utf-8')
print(summary.loc[summary.window.isin(['400:499','0:499','last100']),['experiment','window','completion_rate_mean','sla_satisfaction_rate_mean','avg_completion_time_mean','total_system_energy_kwh_mean','training_episode_reward_mean']].round(5).to_string(index=False))
print('OUTPUT',OUT)
