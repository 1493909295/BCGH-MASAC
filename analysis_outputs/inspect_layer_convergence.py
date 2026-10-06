from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'result/BCGH-MASAC/0.35-2000-cloud-异构/不同的三阶段'
OUT=ROOT/'analysis_outputs/layer_convergence_20261006'
OUT.mkdir(exist_ok=True)
names=['200-200-700','200-800','500-500']
general=['completion_rate','sla_satisfaction_rate','avg_completion_time','total_system_energy_kwh',
         'training_episode_reward','host_layer_reward_sum','host_started_rate','host_queued_rate']
routing=['routing_critic_loss','routing_actor_loss','routing_alpha','routing_policy_entropy','routing_target_entropy',
         'routing_mean_q1','routing_mean_target_q','routing_episode_updates','routing_self_rate','routing_edge_rate','routing_cloud_rate','multi_hop_job_rate','bcgh_guidance_lambda']
host=['host_critic_loss','host_actor_loss','host_alpha_mean','host_policy_entropy','host_target_entropy',
      'host_mean_q1','host_mean_target_q','host_episode_updates','host_started_rate','host_queued_rate','host_dropped_rate']
rows=[]
for name in names:
    d=pd.read_csv(SRC/f'{name}.csv').copy()
    start=int(d.loc[d.training_stage.eq('routing_train'),'episode'].min())
    d['routing_round']=d.episode-start+1
    for group,select,cols in [('host_pretrain',d.training_stage.eq('host_pretrain'),general+host),
                             ('routing_all',d.routing_round.ge(1),general+routing),
                             ('joint_finetune',d.training_stage.eq('joint_finetune'),general+host+routing)]:
        x=d.loc[select].copy()
        if x.empty:continue
        x['phase_round']=np.arange(1,len(x)+1)
        unique=list(dict.fromkeys(cols))
        print('\nRUN',name,group,'n',len(x))
        out=[]
        for a in range(1,len(x)+1,50):
            y=x.loc[x.phase_round.between(a,a+49)]
            if len(y)<25:continue
            r={'experiment':name,'phase':group,'window':f'{a}-{int(y.phase_round.max())}',
               'original_start':int(y.episode.min()),'original_end':int(y.episode.max()),'n':len(y),
               **{c+'_mean':float(y[c].mean()) if y[c].notna().any() else None for c in unique},
               **{c+'_sd':float(y[c].std(ddof=1)) if y[c].notna().sum()>1 else None for c in unique}}
            rows.append(r)
            out.append({k:v for k,v in r.items() if k in ['window','original_start','original_end','n'] or (k.endswith('_mean') and k[:-5] in unique)})
        print(pd.DataFrame(out).round(4).to_string(index=False))
        print('LAST100_TREND')
        tail=x.tail(100)
        trends={}
        for c in unique:
            good=tail[c].notna()
            if good.sum()>=20:
                slope=np.polyfit(tail.loc[good,'phase_round'],tail.loc[good,c],1)[0]
                first=tail[c].iloc[:50].mean();last=tail[c].iloc[50:].mean()
                trends[c]={'first50':round(first,5),'last50':round(last,5),'slope_x100':round(slope*100,5)}
        print(trends)
pd.DataFrame(rows).to_csv(OUT/'phase_50_round_statistics.csv',index=False,encoding='utf-8-sig')

# Learning curves are diagnostic evidence, not a proof of parameter convergence.
from reportlab.graphics.shapes import Drawing, String, Line, Rect
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics import renderSVG
from reportlab.lib.colors import HexColor
import math

data={name:pd.read_csv(SRC/f'{name}.csv').copy() for name in names}
colors=['#2563EB','#D97706','#059669']
for name,d in data.items():
    start=int(d.loc[d.training_stage.eq('routing_train'),'episode'].min())
    d['routing_round']=d.episode-start+1
    d['joint_round']=d.episode-400

def nice(values):
    lo,hi=min(values),max(values)
    span=max(hi-lo,abs(hi)*.03,.001)
    raw=span/5;p=10**math.floor(math.log10(raw))
    step=next(k*p for k in [1,2,2.5,5,10] if k*p>=raw)
    return math.floor((lo-.02*span)/step)*step,math.ceil((hi+.02*span)/step)*step,step

def plot_diagnostics():
    draw=Drawing(1500,1120)
    draw.add(Rect(0,0,1500,1120,fillColor=HexColor('#FFFFFF'),strokeColor=None))
    draw.add(String(50,1080,'Learning diagnostics by actively trained layer',fontName='Helvetica-Bold',fontSize=22))
    draw.add(String(50,1054,'Trailing 20-round means; curves exclude frozen stages. Different panels use different training clocks.',fontSize=13))
    for i,(name,color) in enumerate(zip(names,colors)):
        x=50+i*250;draw.add(Line(x,1024,x+30,1024,strokeColor=HexColor(color),strokeWidth=3))
        draw.add(String(x+40,1019,name,fontName='Helvetica-Bold',fontSize=13))
    specs=[('routing','training_episode_reward','Routing training return'),
           ('routing','routing_critic_loss','Routing critic loss'),
           ('routing','routing_policy_entropy','Routing policy entropy'),
           ('pretrain','host_layer_reward_sum','Host pretraining return'),
           ('pretrain','host_alpha_mean','Host pretraining entropy coefficient'),
           ('pretrain','host_policy_entropy','Host pretraining policy entropy'),
           ('joint','host_layer_reward_sum','Host joint-training return'),
           ('joint','host_critic_loss','Host joint-training critic loss'),
           ('joint','host_policy_entropy','Host joint-training policy entropy')]
    for i,(phase,metric,title) in enumerate(specs):
        x=85+(i%3)*487;y=738-(i//3)*314;w=410;h=213
        draw.add(String(x,y+h+32,title,fontName='Helvetica-Bold',fontSize=14))
        series=[];used=[];values=[]
        for name,color in zip(names,colors):
            d=data[name]
            if phase=='routing':
                q=d.loc[d.routing_round.ge(1)];clock='routing_round'
            elif phase=='pretrain':
                q=d.loc[d.training_stage.eq('host_pretrain')];clock='episode'
            else:
                q=d.loc[d.training_stage.eq('joint_finetune')];clock='joint_round'
            if q.empty:continue
            smooth=q[metric].rolling(20,min_periods=20).mean()
            pts=[(float(t),float(v)) for t,v in zip(q[clock],smooth) if np.isfinite(v)]
            series.append(pts);used.append(color);values.extend(v for _,v in pts)
            if metric.endswith('policy_entropy'):
                target='routing_target_entropy' if phase=='routing' else 'host_target_entropy'
                target_smooth=q[target].rolling(20,min_periods=20).mean()
                pts=[(float(t),float(v)) for t,v in zip(q[clock],target_smooth) if np.isfinite(v)]
                series.append(pts);used.append('#64748B');values.extend(v for _,v in pts)
        lo,hi,step=nice(values);xmax={'routing':808,'pretrain':500,'joint':608}[phase]
        for tick in np.arange(lo,hi+step*.1,step):
            yy=y+h*(tick-lo)/(hi-lo)
            draw.add(Line(x,yy,x+w,yy,strokeColor=HexColor('#E3E7ED'),strokeWidth=.5))
        lp=LinePlot();lp.x=x;lp.y=y;lp.width=w;lp.height=h;lp.data=series;lp.joinedLines=1
        lp.xValueAxis.valueMin=1;lp.xValueAxis.valueMax=xmax;lp.xValueAxis.valueStep=200 if phase=='routing' else 100
        lp.yValueAxis.valueMin=lo;lp.yValueAxis.valueMax=hi;lp.yValueAxis.valueStep=step
        lp.xValueAxis.labels.fontSize=10;lp.yValueAxis.labels.fontSize=10;lp.yValueAxis.labelTextFormat='%g'
        lp.xValueAxis.strokeColor=HexColor('#94A3B8');lp.yValueAxis.strokeColor=HexColor('#94A3B8')
        for j,color in enumerate(used):
            lp.lines[j].strokeColor=HexColor(color);lp.lines[j].strokeWidth=1.9
            if color=='#64748B':lp.lines[j].strokeDashArray=[3,3]
        draw.add(lp)
        xlabel={'routing':'Rounds since routing began (first = 1)',
                'pretrain':'Host pretraining rounds (first = 1)',
                'joint':'Rounds since joint training began (first = 1)'}[phase]
        draw.add(String(x+w/2,y-32,xlabel,fontSize=11,textAnchor='middle',fillColor=HexColor('#475569')))
        if phase=='routing':
            xx=x+w*200/(xmax-1)
            draw.add(Line(xx,y,xx,y+h,strokeColor=HexColor(colors[0]),strokeWidth=.8,strokeDashArray=[4,3]))
    draw.add(String(50,53,'Blue vertical line: joint training begins in 200-200-700. Grey dashed lines in entropy panels: target entropy.',fontSize=12))
    draw.add(String(50,30,'A stable entropy alone does not prove a stable policy. Critic loss is assessed together with return, actions and service outcomes.',fontSize=12))
    path=OUT/'layer_learning_diagnostics.svg';renderSVG.drawToFile(draw,str(path))
    svg=path.read_text(encoding='utf-8').replace('font-family: Helvetica-Bold;','font-family: Arial; font-weight: bold;').replace('font-family: Helvetica;','font-family: Arial;')
    path.write_text(svg,encoding='utf-8')

plot_diagnostics()

stats=pd.DataFrame(rows)
def get(name,phase,window,metric):
    return float(stats.loc[(stats.experiment==name)&(stats.phase==phase)&(stats.window==window),metric+'_mean'].iloc[0])

report='''# Routing 与 Host 层的近收敛分析

## 判断口径

这里把“接近收敛的效果”理解为：回报、服务指标和动作统计已进入主要收益实现后的平台区间。用实际更新阶段的连续 50 轮均值判断趋势，并同时看 actor、critic、熵及熵系数。轮次是原始日志 episode，不是梯度更新步数。

这是对当前曲线的平台期估计，不是数学收敛判定或统一阈值检验。没有固定负载的独立评估曲线、参数变化量、梯度范数或多个训练随机种子的证据，因此不能给出精确的“第 N 轮已经收敛”。策略熵靠近目标熵只说明熵调节接近目标，不单独证明策略收敛。

## 判断结果

Routing 的策略与服务表现已经出现明显的近收敛特征。主要提升通常在前 100–200 轮实现，200-800 稍慢，约 150–250 轮接近后期效果。后续仍有小幅改善，critic loss 也继续下降。

Host 的收敛证据较弱。200 轮预训练结束时仍在明显变化；500 轮预训练日志显示约 200–300 轮后服务表现进入平台区间，但熵系数、actor 与 Q 值仍变化。只有 200-200-700 有联合阶段，Host 的动作行为约在联合训练 150–250 轮后接近后期状态，300–400 轮后更稳定；截至实际记录的 608 轮仍不能认定整个 Host 学习器完全收敛。

| 配置 | Routing 接近后期效果的估计 | 对应原始轮次 | Host 判断 |
|---|---|---|---|
| 200-200-700 | 单独训练约 100–150 轮；联合阶段开始后需重新观察，约联合 150–250 轮后动作行为接近后期状态 | Routing 约 300–350；联合约 550–650 | 200 轮预训练尚未充分稳定；联合后有行为平台，内部训练仍变化 |
| 200-800 | Routing 约 150–250 轮，200–300 轮后可视为主要收益已实现 | 约 350–450，保守观察到 400–500 | 仅训练前 200 轮，之后冻结，不能用后续平稳判断 Host 已收敛 |
| 500-500 | Routing 约 100–150 轮已接近后期服务效果，行为比例仍有缓慢调整 | 约 600–650 | 预训练效果约 200–300 轮进入平台，但截至 500 轮内部量仍漂移 |

范围是近似估计。“稳定”也不意味着达到最优表现。联合训练平台的 SLA／时延表现并未优于冻结 Host 的 200-800。

## Routing 的证据

| 配置 | Routing 窗口 | 回报 | 完成率（%） | SLA（%） | 策略熵 | Critic loss |
|---|---|---:|---:|---:|---:|---:|
'''
for name,windows in [('200-200-700',['1-50','51-100','101-150','151-200','751-800']),
                     ('200-800',['1-50','101-150','151-200','201-250','751-800']),
                     ('500-500',['1-50','51-100','101-150','451-500'])]:
    for window in windows:
        vals=[get(name,'routing_all',window,c)*scale for c,scale in [('training_episode_reward',1),('completion_rate',100),('sla_satisfaction_rate',100),('routing_policy_entropy',1),('routing_critic_loss',1)]]
        report+='| '+name+' | '+window+' | '+' | '.join(f'{v:.4f}' if i>=3 else f'{v:.2f}' for i,v in enumerate(vals))+' |\n'
report+='''
Routing 的目标熵为约 0.35835。三组在约 50–100 轮后，平均策略熵已靠近此值；actor loss 和 Q 值主要在 100–250 轮稳定下来。回报达到约 2950–3050 后，进一步训练的收益相对初期跃升明显减小。

但是 entropy 的稳定比任务表现稳定更早，不能把第 50 轮就称为完整收敛。200-200-700 的 critic loss 从 routing 第 151–200 轮的约 0.7895 继续降至第 751–800 轮的约 0.3798；200-800 从约 0.9792 降至 0.6715。因此“回报与服务效果接近稳定”比“所有网络量停止变化”更符合数据。

## Host 预训练的证据

200-800 与 500-500 的前 200 轮 Host 预训练在所核对的指标上相同，可用同一条预训练轨迹观察继续训练到 500 轮的变化。

| 配置 | Host 预训练窗口 | Host 回报 | 完成率（%） | SLA（%） | Host 熵系数 α | 策略熵 | 目标熵 |
|---|---|---:|---:|---:|---:|---:|---:|
'''
for name,windows in [('200-200-700',['101-150','151-200']),('200-800',['101-150','151-200']),('500-500',['151-200','201-250','251-300','351-400','451-500'])]:
    for window in windows:
        vals=[get(name,'host_pretrain',window,c)*scale for c,scale in [('host_layer_reward_sum',1),('completion_rate',100),('sla_satisfaction_rate',100),('host_alpha_mean',1),('host_policy_entropy',1),('host_target_entropy',1)]]
        report+='| '+name+' | '+window+' | '+' | '.join(f'{v:.4f}' if i>=3 else f'{v:.2f}' for i,v in enumerate(vals))+' |\n'
b=get('200-800','host_pretrain','151-200','completion_rate');c=get('500-500','host_pretrain','451-500','completion_rate')
sb=get('200-800','host_pretrain','151-200','sla_satisfaction_rate');sc=get('500-500','host_pretrain','451-500','sla_satisfaction_rate')
report+=f'''
两份 200 轮预训练的日志，最后两个 50 轮窗口仍存在明显提升：完成率提升约 1.6–2.1 个百分点，Host 回报提升约 193–251。Host α 从约 0.09–0.095 降至约 0.059–0.060，策略熵约 0.75–0.76，明显高于目标约 0.593。因此 200 轮更适合作为获得可用初始化的预算，不能称为充分收敛。

从 200-800 的第 151–200 轮到 500-500 的第 451–500 轮，完成率仅增加 {(c-b)*100:.2f} 个百分点，SLA 增加 {(sc-sb)*100:.2f} 个百分点。额外 300 轮预训练的服务收益较小，约 200–300 轮后的曲线已经表现出平台特征。

但 Host 内部量还未完全稳定：第 251–300 轮到 451–500 轮的 α 从约 0.0299 降至 0.0158，策略熵从约 0.688 降至 0.638，仍高于目标熵约 0.593；actor loss 和 Q 值也继续漂移。预训练达到效果平台与网络充分收敛应分别报告。

## Host 联合训练的证据

只有 200-200-700 实际进入此阶段。Host 冻结期间的回报和服务变化，主要体现输入任务／routing 策略变化，不能说明 Host 网络在继续学习。

| 联合训练窗口 | 原始轮次 | Host 立即启动（%） | Host 回报 | Host critic loss | 策略熵 | 目标熵 |
|---|---|---:|---:|---:|---:|---:|
'''
for window in ['1-50','51-100','101-150','151-200','251-300','351-400','451-500','551-600']:
    r=stats.loc[(stats.experiment=='200-200-700')&(stats.phase=='joint_finetune')&(stats.window==window)].iloc[0]
    vals=[float(r[c+'_mean'])*scale for c,scale in [('host_started_rate',100),('host_layer_reward_sum',1),('host_critic_loss',1),('host_policy_entropy',1),('host_target_entropy',1)]]
    report+=f'| {window} | {int(r.original_start)}–{int(r.original_end)} | '+' | '.join(f'{v:.4f}' if i>=2 else f'{v:.2f}' for i,v in enumerate(vals))+' |\n'
report+='''
Host 立即启动率从联合前 50 轮的约 54.85% 降到第 101–150 轮的约 45.71%，之后主要在约 44%–46% 间波动，说明策略经过约 100–200 轮重新适应。熵在后期靠近约 0.600 的目标，而 critic loss 和 Q 值仍持续变化。最近两个完整 50 轮窗口，critic loss 约从 0.543 降至 0.486，Q 均值约从 1.739 增至 1.822，所以还没有所有内部训练量均已稳定的证据。

联合阶段约 150–250 轮可作为接近后期行为的估计，300–400 轮观察更稳妥。这里的平台是当前联合策略的行为平台，并非相比冻结 Host 的性能提升。

![两层学习诊断](layer_learning_diagnostics.png)

## 实验预算上的含义

若目标是接近目前日志的服务效果，可先以 Host 预训练 200–300 轮、Routing 单独训练 200–300 轮作为新的预算参照。Host 若要考察内部学习器是否继续接近收敛，可保留 400–500 轮预训练观察；当前 500 轮日志也仍有漂移，无法确定它之后的完整收敛轮次。

如需要联合训练，可在开始后约 200–300 轮进行固定负载的评估，300–400 轮观察稳定性，而不是因为训练更久就认定更好。上述轮数是基于当前日志的预算估计，改变阶段长度也会改变现有 λ 调度，不能保证截短后复现原曲线。

严谨验证需要在固定评估 seed 上测试保存的策略，冻结评估时的 λ 并报告多训练种子。按指标选择 checkpoint，比只看单轮训练回报或只看 critic loss 更可靠。只有这三份聚合日志，不能证明每个 DC 的 Host 都已稳定，聚合均值可能遮盖个体差异。

数据来源：用户指定的 200-200-700、200-800、500-500 三份 episode 日志。原始数据未修改，50 轮统计保存在 `phase_50_round_statistics.csv`。
'''
(OUT/'convergence_report.md').write_text(report,encoding='utf-8')
print('SAVED',OUT)
