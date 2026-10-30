"""Static scientific plots of phase8 raw measured values; no smoothing."""
import argparse,json,csv,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from phase8_evalmetrics import spectral
from ppo_common import dump_json
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir).resolve();start=time.perf_counter();out=r/'plots'
def read(p):return json.loads(Path(p).read_text())
plan=read(r/'experiment_plan.json');names=plan['order'];colors={'Parent':'black','Q_base':'tab:blue','E_position':'tab:orange','E_omega':'tab:green'}
plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.2,'savefig.dpi':140})
def save(fig,name):fig.tight_layout(rect=(0,0,1,.97));fig.savefig(out/(name+'.png'));plt.close(fig)
def csvsave(name,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (out/(name+'.csv')).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
# Reward shape and exact analytic derivative.
x=np.linspace(-.5,.5,2001);curve=[]
fig,axs=plt.subplots(2,2,figsize=(12,7))
for j,(axis,sig,coef) in enumerate([('xy along X; ey=0',.10,10.),('z',.05,6.)]):
    q=-coef*x*x;e=coef*sig*sig*np.expm1(-x*x/(sig*sig));gq=-2*coef*x;ge=gq*np.exp(-x*x/(sig*sig))
    for ax,y1,y2,label in [(axs[0,j],q,e,'reward / policy step'),(axs[1,j],gq,ge,'d reward / d error [1/m]')]:
        ax.plot(x,y1,label='quadratic');ax.plot(x,y2,label='exponential');ax.set(xlabel=f'{axis} error [m]',ylabel=label);ax.legend()
    for i in range(len(x)):curve.append(dict(axis=axis,error_m=x[i],quadratic=q[i],exponential=e[i],quadratic_gradient=gq[i],exponential_gradient=ge[i]))
fig.suptitle('Phase8 position penalties: same curvature at zero; different far-error scale');save(fig,'reward_curves');csvsave('reward_curves',curve)
fig,axs=plt.subplots(2,2,figsize=(12,7))
for j,(axis,sig,coef) in enumerate([('xy along X; ey=0',.10,10.),('z',.05,6.)]):
    xx=np.linspace(-sig,sig,1001);q=-coef*xx*xx;e=coef*sig*sig*np.expm1(-xx*xx/(sig*sig))
    gq=-2*coef*xx;ge=gq*np.exp(-xx*xx/(sig*sig))
    for ax,y1,y2,label in [(axs[0,j],q,e,'Reward / policy step'),(axs[1,j],gq,ge,'Reward derivative [1/m]')]:
        ax.plot(xx,y1,label='quadratic');ax.plot(xx,y2,label='exponential');ax.set(xlabel=axis+' error [m]',ylabel=label);ax.legend()
fig.suptitle('Near-target zoom: error within one sigma; sigma is not a success tolerance');save(fig,'reward_curves_near_target')
history=read(r/'evaluation_history.json')
metrics=[('completed_count','8s completed conditions'),('completed_position_rmse_m','Completed-condition 3D RMSE [m]'),
    ('tail_body_omega_rms_rad_s','Completed tail body omega RMS [rad/s]'),('tail_z_peak_to_peak_m','Completed tail Z peak-to-peak [m]'),
    ('tail_roll_rad_band_10_20_power','Completed tail roll 10-20 Hz power [rad^2]'),('tail_position_mean_m','Completed tail position error [m]')]
for xkey,xlabel in [('additional_transitions','Additional policy transitions'),('additional_updates','Additional actual Adam updates'),('training_elapsed_s','Additional training wall time [s]')]:
    fig,axs=plt.subplots(3,2,figsize=(12,12))
    for ax,(key,label) in zip(axs.flat,metrics):
        for n in names:
            hs=[z for z in history if z['branch']==n];ax.plot([z[xkey] for z in hs],[np.nan if z.get(key) is None else z[key] for z in hs],'.-',label=n,color=colors[n])
        ax.set(xlabel=xlabel,ylabel=label);ax.legend()
    fig.suptitle('Phase8 fixed 8s evaluations; failed episode tail values omitted');save(fig,'evaluation_vs_'+xkey)
training={n:[json.loads(x) for x in (r/n/'metrics.jsonl').read_text().splitlines()] for n in names}
groups={
    'training_reward':[f'RewardStep/{k}' for k in ['xy','z','velocity','tilt','angular_velocity','yaw','termination','total']]+['Monitor/common_reward_A'],
    'training_ppo':['Loss/policy','Loss/value','Loss/entropy_contribution','Loss/ppo_total','PPO/approx_kl','PPO/clip_fraction','Policy/entropy','PPO/learning_rate']+[f'Policy/std_axis_{i}' for i in range(4)],
    'training_physical':['Physical/xy_error_squared_m2','Physical/z_error_squared_m2','Physical/velocity_squared_m2_s2','Physical/body_omega_squared_rad2_s2','Kernel/xy','Kernel/z'],
    'training_performance':['Performance/transitions_per_s','Performance/rollout_s','Performance/update_s','Performance/cuda_device_used_MiB','Performance/torch_peak_allocated_MiB','Episode/length_s','Episode/failure_fraction','Actuator/action_clipping_fraction','Actuator/allocation_clipping_fraction']}
for name,tags in groups.items():
    fig,axs=plt.subplots(len(tags),1,figsize=(12,2.1*len(tags)),squeeze=False);csvrows=[]
    for ax,tag in zip(axs.flat,tags):
        for n in names:
            rr=training[n];x=[z['global_step'] for z in rr];y=[z.get(tag,np.nan) for z in rr]
            ax.plot(x,y,label=n,color=colors[n],lw=.8)
        ax.set(ylabel=tag);ax.legend(loc='best',fontsize=7)
    axs[-1,0].set_xlabel('Additional policy transitions');fig.suptitle('Phase8 raw iteration aggregates; reward and PPO objectives are distinct')
    save(fig,name)
    for n in names:
        for z in training[n]:csvrows.append(dict(branch=n,additional_transitions=z['global_step'],**{k:z.get(k) for k in tags}))
    csvsave(name,csvrows)
selected=read(r/'selected_summary.json')
conditions=['nominal','x_plus','x_minus','y_plus','y_minus']
def getrows(n,kind,horizon,cond):
    if n=='Parent':path=r/f'common/evaluation_{horizon}s/initial'/(cond+'.json')
    else:
        k=kind
        path=r/n/f'evaluation/reload_{horizon}s'/k/(cond+'.json')
        if not path.exists():path=r/n/f'evaluation/reload_{horizon}s/final'/(cond+'.json')
    return read(path)['trajectory']
signals=['ex','ey','ez','vx','vy','vz','position_error_m','roll_deg','pitch_deg','wx','wy','wz','raw_action_0','action_0','actual_0','total_thrust_N']
blockrows=[];spectralrows=[]
for horizon in [8,30]:
    for kind in ['best','final']:
        data={(n,c):getrows(n,kind,horizon,c) for n in ['Parent',*names] for c in conditions}
        tag=f'{horizon}s_{kind}'
        flat=[]
        for (n,c),rs in data.items():
            for row in rs:flat.append(dict(branch=n,condition=c,time_s=row['time_s'],**{k:row[k] for k in signals}))
        csvsave('trajectories_'+tag,flat)
        fig,axs=plt.subplots(5,3,figsize=(14,15),sharex=True)
        for j,c in enumerate(conditions):
            for i,k in enumerate(['ex','ey','ez']):
                ax=axs[j,i]
                for n in ['Parent',*names]:
                    rs=data[n,c];ax.plot([z['time_s'] for z in rs],[100*z[k] for z in rs],color=colors[n],lw=.8,label=n)
                ax.set(title=c,ylabel=f'{k} [cm]',xlim=(0,horizon))
                if j==4:ax.set_xlabel('Episode time [s]')
        for i in range(3):
            limits=[ax.get_ylim() for ax in axs[:,i]];low=min(x[0] for x in limits);high=max(x[1] for x in limits)
            for ax in axs[:,i]:ax.set_ylim(low,high)
        axs[0,0].legend(fontsize=7);fig.suptitle(f'{tag}: link-origin position error = world position - env origin - target');save(fig,'position_'+tag)
        # Zoom panels explicitly distinguish Euler angle, body rate, action, and applied COM torque.
        intervals=[(0,2),(horizon-1,horizon)]
        specs=[('roll_deg','Euler roll [deg]'),('wx','Body roll rate [rad/s]'),('raw_action_0','Raw actor mean roll [normalized]'),('action_0','Clipped roll action [normalized]'),('actual_0','Applied COM roll torque [N m]')]
        fig,axs=plt.subplots(len(specs),2,figsize=(14,13))
        for j,(lo,hi) in enumerate(intervals):
            for i,(k,label) in enumerate(specs):
                ax=axs[i,j]
                for n in ['Parent',*names]:
                    rs=[z for z in data[n,'nominal'] if lo<z['time_s']<=hi];ax.plot([z['time_s'] for z in rs],[z[k] for z in rs],color=colors[n],lw=.8,label=n)
                ax.set(xlim=(lo,hi),ylabel=label,xlabel='Time [s]',title=f'Nominal, ({lo}, {hi}] s')
        for i in range(len(specs)):
            limits=[ax.get_ylim() for ax in axs[i]];lo=min(v[0] for v in limits);hi=max(v[1] for v in limits)
            for ax in axs[i]:ax.set_ylim(lo,hi)
        axs[0,0].legend(fontsize=7);fig.suptitle(f'{tag}: roll waveform, initial return and final 1-second zoom');save(fig,'roll_zoom_'+tag)
        fig,axs=plt.subplots(3,2,figsize=(14,9))
        for j,(lo,hi) in enumerate([(0,horizon),(horizon-(2 if horizon==8 else 10),horizon)]):
            for i,(k,label) in enumerate([('ez','Z error [m]'),('vz','Z velocity [m/s]'),('total_thrust_N','Total actual rotor thrust [N]')]):
                ax=axs[i,j]
                for n in ['Parent',*names]:
                    rs=[z for z in data[n,'nominal'] if lo<z['time_s']<=hi];ax.plot([z['time_s'] for z in rs],[z[k] for z in rs],color=colors[n],lw=.8,label=n)
                ax.set(xlim=(lo,hi),xlabel='Time [s]',ylabel=label)
        axs[0,0].legend();fig.suptitle(f'{tag}: nominal Z waveform; whole episode and final interval');save(fig,'z_waveform_'+tag)
        fig,axs=plt.subplots(4,3,figsize=(15,13))
        for j,segment in enumerate(['initial_2s','full','tail']):
            for i,(key,unit,xmax) in enumerate([('roll_rad','rad^2/Hz',25),('wx','(rad/s)^2/Hz',25),('ez','m^2/Hz',10),('vz','(m/s)^2/Hz',10)]):
                ax=axs[i,j]
                for n in ['Parent',*names]:
                    rs=data[n,'nominal'];complete=len(rs)==horizon*100 and not rs[-1]['terminated']
                    seq=rs[:200] if segment=='initial_2s' else rs if segment=='full' else rs[-(200 if horizon==8 else 1000):] if complete else []
                    if not seq:continue
                    sp=spectral([z[key] for z in seq])
                    if sp is None:continue
                    ax.semilogy(sp['frequency_hz'],np.maximum(sp['psd'],1e-20),label=n,color=colors[n])
                    for f,power in zip(sp['frequency_hz'],sp['psd']):spectralrows.append(dict(branch=n,kind=kind,horizon_s=horizon,condition='nominal',segment=segment,signal=key,frequency_hz=f,psd=power,units=unit))
                ax.set(xlim=(0,xmax),xlabel='Frequency [Hz]',ylabel=key+' PSD ['+unit+']',title=segment)
        axs[0,0].legend(fontsize=7);fig.suptitle(f'{tag}: 100 Hz policy sampling, mean detrend, Hann, one-sided density');save(fig,'spectra_'+tag)
        fig,axs=plt.subplots(4,1,figsize=(13,12))
        for ax,(key,unit) in zip(axs,[('roll_rad','rad^2/Hz'),('wx','(rad/s)^2/Hz'),('ez','m^2/Hz'),('vz','(m/s)^2/Hz')]):
            for n in ['Parent',*names]:
                rs=data[n,'nominal']
                if len(rs)!=horizon*100 or rs[-1]['terminated']:continue
                seq=rs[-(200 if horizon==8 else 1000):]
                sp=spectral([z[key] for z in seq])
                ax.semilogy(sp['frequency_hz'],np.maximum(sp['psd'],1e-20),label=n,color=colors[n])
            ax.set(xlim=(0,50),xlabel='Frequency [Hz]',ylabel=key+' PSD ['+unit+']')
            ax.legend(fontsize=7)
        fig.suptitle(f'{tag}: complete nominal tail, full measurable 0-50 Hz band');save(fig,'spectra_fullband_'+tag)
        if horizon==30:
            for (n,c),rs in data.items():
                for lo in [0,10,20]:
                    block=[z for z in rs if lo<z['time_s']<=lo+10]
                    if not block:continue
                    e=np.array([[z[k] for k in ['ex','ey','ez']] for z in block]);omega=np.array([[z[k] for k in ['wx','wy','wz']] for z in block])
                    blockrows.append(dict(branch=n,kind=kind,condition=c,start_s=lo,actual_end_s=block[-1]['time_s'],complete_10s=len(block)==1000,
                        ex_mean_m=float(e[:,0].mean()),ey_mean_m=float(e[:,1].mean()),ez_mean_m=float(e[:,2].mean()),
                        position_rmse_m=float(np.sqrt((e*e).sum(1).mean())),omega_rms_rad_s=float(np.sqrt((omega*omega).sum(1).mean())),
                        z_peak_to_peak_m=float(np.ptp(e[:,2]))))
csvsave('spectral_density_nominal',spectralrows);csvsave('30s_time_blocks',blockrows)
fig,axs=plt.subplots(3,2,figsize=(12,10))
for ax,(key,label) in zip(axs.flat,[('ex_mean_m','X bias [m]'),('ey_mean_m','Y bias [m]'),('ez_mean_m','Z bias [m]'),('position_rmse_m','3D RMSE [m]'),('omega_rms_rad_s','Body omega RMS [rad/s]'),('z_peak_to_peak_m','Original Z peak-to-peak [m]')]):
    for n in ['Parent',*names]:
        xs=[z for z in blockrows if z['kind']=='final' and z['condition']=='nominal' and z['branch']==n and z['complete_10s']]
        ax.plot([z['start_s']+5 for z in xs],[z[key] for z in xs],'.-',color=colors[n],label=n)
    ax.set(xlabel='Center of measured 10-second block [s]',ylabel=label);ax.legend(fontsize=7)
fig.suptitle('30s nominal final-policy bias and oscillation by complete 10-second blocks');save(fig,'30s_block_changes')
fig,axs=plt.subplots(2,2,figsize=(12,9))
for j,h in enumerate([8,30]):
    for i,(key,label) in enumerate([('tail_body_omega_rms_rad_s','Tail body omega RMS [rad/s]'),('tail_roll_rad_band_10_20_power','Tail roll 10-20 Hz power [rad^2]')]):
        ax=axs[i,j]
        for z in selected:
            if z['horizon_s']!=h or z['completed_count']!=5:continue
            ax.scatter(100*z['position_rmse_m'],z[key],color=colors[z['branch']],marker='*' if z['kind']=='best' else 's' if z['kind']=='final' else 'o',s=70,label=z['branch']+' '+z['kind'])
        ax.set(xlabel='Mean position RMSE [cm]',ylabel=label,title=f'{h}s: only 5/5 completed candidates')
        if i==1:ax.set_yscale('log')
        ax.legend(fontsize=6)
fig.suptitle('Position accuracy and oscillation are separate outcomes');save(fig,'position_oscillation_tradeoff')
dump_json(r/'plot_generation.json',dict(success=True,wall_s=time.perf_counter()-start,pngs=[p.name for p in out.glob('*.png')],raw_no_smoothing=True,frequency_analysis='100Hz; constant detrend; symmetric Hann; density normalized'))
print('PLOTS_COMPLETE',flush=True)

