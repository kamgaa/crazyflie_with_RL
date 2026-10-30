"""Aggregate two fixed-budget reward ablations and unchanged circle evaluations."""
import argparse,csv,hashlib,json,math,shutil,socket,time,sys
from pathlib import Path
import numpy as np,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from ppo_common import ROOT,dump_json,sha256,protected_hashes
from phase11_reward import TERMS,components
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir).resolve();t0=time.perf_counter();plan=json.loads((r/'experiment_plan.json').read_text());base=Path(plan['baseline']);plots=r/'plots'
def read(p):return json.loads(Path(p).read_text())
def csvout(p,rows):
 keys=list(dict.fromkeys(k for x in rows for k in x))
 with Path(p).open('w',newline='') as f:w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def csvread(p):
 with Path(p).open() as f:return [{k:float(v) if v not in ['True','False',''] else (v=='True' if v else None) for k,v in z.items()} for z in csv.DictReader(f)]
def mean(ds,k):return float(np.mean([x[k] for x in ds])) if ds else None
runs={'phase11':base,**{k:Path(v) for k,v in plan['branches'].items()}}
labels={'phase11':'phase11 기준 0.15','phase12-1':'phase12-1 자세 벌점 0.5배 (0.075)','phase12-2':'phase12-2 자세 벌점 2배 (0.30)','PID':'기존 PID'}
colors={'phase11':'#009E73','phase12-1':'#D55E00','phase12-2':'#CC79A7','PID':'#333333'}
checks={};training=[];hover=[];candidates=[];datasets={};detailed=[]
for name,d in runs.items():
 tr=read(d/'training_result.json');sel=read(d/'best_selection.json');training.append(dict(condition=name,historical_reused=name=='phase11',roll_pitch_weight=read(d/'reward_config.json')['axis_weights']['attitude'][0],iterations=tr['iterations'],transitions=tr['transitions'],optimizer_updates=tr['optimizer_updates'],adam_step=tr['adam_steps'][0],training_s=tr['training_wall_s'],transitions_per_s=tr['transitions_per_s'],initialization_s=tr['initialization_s'],initial_evaluation_s=tr['initial_evaluation_wall_s'],intermediate_evaluation_s=tr['intermediate_evaluation_wall_s'],device_peak_sampled_MiB=tr['cuda_device_used_peak_sampled_MiB'],torch_peak_allocated_MiB=tr['memory_final']['torch_peak_allocated_MiB'] if 'torch_peak_allocated_MiB' in tr['memory_final'] else tr['memory_final'].get('torch_max_allocated_MiB'),best_iteration=sel['selected']['additional_iteration'],best_sha256=sha256(d/'best.pt'),final_sha256=sha256(d/'final.pt')))
 for it in range(0,513,64):
  ds=read(d/'evaluation'/f'validation_{it:03d}'/'results.json');done=[x for x in ds if x['completed']]
  candidates.append(dict(condition=name,iteration=it,transitions=it*32768,completed=len(done),total=len(ds),mean_duration_s=mean(ds,'duration_s'),completed_only_position_rmse_m=mean(done,'position_rmse_m'),completed_only_tail_omega_rms_rad_s=mean(done,'tail_body_omega_rms_rad_s'),completed_only_tail_action_saturation=mean(done,'tail_applied_action_saturation_fraction')))
 for model in ['best','final']:
  for suite in ['fixed','random20']:
   ds=read(d/'evaluation'/f'{model}_{suite}_30s'/'results.json');datasets[(name,model,suite)]=ds;done=[x for x in ds if x['completed']];settled=[x for x in done if x['condition']!='nominal' and x['settling_time_1cm_1s_s'] is not None];cross=[max(x['early0_5_cross_axis_max_m'].values()) for x in done if x['early0_5_cross_axis_max_m']]
   hover.append(dict(condition=name,model=model,suite=suite,checkpoint_iteration=ds[0]['iteration'],completed=len(done),total=len(ds),mean_duration_s=mean(ds,'duration_s'),completed_only_position_rmse_m=mean(done,'position_rmse_m'),completed_only_tail_speed_rms_m_s=mean(done,'tail_speed_rms_m_s'),completed_only_tail_omega_rms_rad_s=mean(done,'tail_body_omega_rms_rad_s'),completed_only_tail_roll_rms_rad=mean(done,'tail_roll_rms_rad'),completed_only_tail_pitch_rms_rad=mean(done,'tail_pitch_rms_rad'),completed_only_tail_action_saturation=mean(done,'tail_applied_action_saturation_fraction'),mean_non_nominal_settling_s=mean(settled,'settling_time_1cm_1s_s'),non_nominal_settled_count=len(settled),early_cross_worst_m=max(cross,default=None)))
   detailed.extend(dict(experiment=name,model=model,suite=suite,**x) for x in ds)
 if name!='phase11':
  checks[name+'_branch_verification']=read(d/'verification.json')['success']
  checks[name+'_training_budget']=(tr['iterations'],tr['transitions'],tr['optimizer_updates'],tr['adam_steps'])==(512,16777216,163840,[163840])
  checks[name+'_latest_best_selected_by_hover']=not sel['circle_used_for_selection'] and sha256(d/'best.pt')==sel['best_sha256']
# Counterfactual reward validation and first stochastic rollout parity, before any update.
first={name:torch.load(d/'first_rollout.pt',weights_only=True,map_location='cpu') for name,d in runs.items()};parity={}
for name in ['phase12-1','phase12-2']:
 proof={k:dict(equal=torch.equal(first['phase11'][k],first[name][k]),max_abs=float((first['phase11'][k].double()-first[name][k].double()).abs().max())) for k in ['input_obs','sampled_action','next_obs','obs','raw_action','action','thrust','omega','request','actual','world_position','terminated','truncated']}
 parity[name]=proof;checks[name+'_first_rollout_physics_actions_equal']=all(v['equal'] for v in proof.values())
 cfg=read(r/name/'reward_config.json');ref=components(first[name]['obs'].reshape(-1,15),first[name]['terminated'].reshape(-1),cfg).reshape_as(first[name]['terms']);checks[name+'_reward_from_declared_config']=float((ref-first[name]['terms']).abs().max())<1e-5
 del ref
 del first[name]
del first
checks['common_initialization']=read(r/'common_initialization_verification.json')['success'];dump_json(r/'first_rollout_comparison.json',parity)
checks['only_requested_reward_weights_changed']=all((lambda c: c==read(base/'reward_config.json'))({**read(r/name/'reward_config.json'),'axis_weights':{**read(r/name/'reward_config.json')['axis_weights'],'attitude':[.15,.15,1.]}}) for name in ['phase12-1','phase12-2'])
checks['identical_agent_configs']=all(read(r/name/'agent_config.json')==read(base/'agent_config.json') for name in ['phase12-1','phase12-2'])
# Circle comparisons. Historical files are reused with explicit provenance.
old=ROOT/'runs/circle_phase11_best_20261014_214656';pidroot=ROOT/'runs/circle_pid_qe_20261013_160146';circle=[];seqs={};circle_scalars=[];circle_sources=[]
for name in ['PID','phase11','phase12-1','phase12-2']:
 for T in [10,5]:
  case=pidroot/'cases'/f'PID_T{T}' if name=='PID' else old/'cases'/f'phase11_best_T{T}' if name=='phase11' else r/name/'circle/cases'/f'phase12_best_T{T}'
  z=read(case/'result.json');rows=read(case/'trajectory.json')['trajectory'];m=z['metrics'];seqs[(name,T)]=rows
  circle.append(dict(condition=name,historical_reused=name in ['PID','phase11'],period_s=T,requested_s=2*T,completed=m['completed_two_reference_laps'],duration_s=m['duration_s'],reasons=','.join(m['reasons']),position_rmse_m=m['position_rmse_m'],radial_rmse_m=m['radial_error_rmse_m'],phase_lag_mean_deg=m['phase_lag_mean_deg'],max_tilt_deg=m['max_tilt_deg'],omega_rms_rad_s=m['body_omega_rms_rad_s'],action_clip_fraction=m['action_clipping_fraction'],allocation_clip_fraction=m['allocation_clipping_fraction'],reference_laps=m['reference_laps_elapsed'],physical_azimuth_revolutions=m['measured_azimuth_revolutions']))
  circle_sources.append(dict(condition=name,period_s=T,result=str(case/'result.json'),result_sha256=sha256(case/'result.json'),trajectory=str(case/'trajectory.json'),trajectory_sha256=sha256(case/'trajectory.json')))
  if name in ['phase12-1','phase12-2']:
   reload=read(case/'reload.json');checks[name+f'_circle_T{T}']=z['success'] and z['optimizer_updates']==0 and reload['max_action_error']<=1e-6 and reload['sha256']==sha256(r/name/'best.pt')
   tb=r/'tensorboard_circle'/f'{name}_T{T}';writer=None if any(tb.glob('events.out.tfevents.*')) else SummaryWriter(str(tb))
   for row in rows:
    for key in ['position_error_m','radial_error_m','phase_lag_deg','tilt_deg','roll_deg','pitch_deg','action_clipping_fraction','allocation_clipping_fraction']:
     v=float(np.float32(row[key]));tag='Circle/'+key
     if writer is not None:writer.add_scalar(tag,v,row['step'])
     circle_scalars.append(dict(condition=name,period_s=T,step=row['step'],simulation_time_s=row['time_s'],tag=tag,value=v))
   if writer is not None:writer.flush();writer.close()
   ev=EventAccumulator(str(tb),size_guidance={'scalars':0});ev.Reload()
   checks[name+f'_circle_T{T}_tensorboard']=all([(e.step,e.value) for e in ev.Scalars(tag)]==[(x['step'],x['value']) for x in circle_scalars if x['condition']==name and x['period_s']==T and x['tag']==tag] for tag in ev.Tags()['scalars'])
checks['physics_sources_unchanged']=protected_hashes()==read(r/'evidence/physics_sha256.json')
checks['previous_artifacts_unchanged']=all(sha256(Path(k))==v for k,v in read(r/'evidence/previous_artifacts_sha256.json').items())
checks['existing_source_unchanged']=all(sha256(ROOT/k)==v for k,v in read(r/'evidence/project_before_sha256.json').items())
checks['total_new_budget']=sum(x['transitions'] for x in training if not x['historical_reused'])==33554432 and sum(x['optimizer_updates'] for x in training if not x['historical_reused'])==327680
for fn,data in [('training_comparison',training),('hover_comparison',hover),('checkpoint_comparison',candidates),('circle_comparison',circle),('circle_scalars',circle_scalars)]:csvout(plots/(fn+'.csv'),data);dump_json(r/(fn+'.json'),data)
dump_json(r/'all_hover_metrics.json',detailed);dump_json(r/'circle_sources.json',circle_sources)
# Exact same-window circle comparisons for failures; never rank short failures by full-horizon RMSE.
from circle_eval_common import reference,segment_metrics
common=[]
for T in [10,5]:
 for name in ['phase11','phase12-1','phase12-2']:
  n=min(len(seqs[(name,T)]),len(seqs[('PID',T)]))
  for which in ['PID',name]:
   m=segment_metrics(seqs[(which,T)][:n]);common.append(dict(pair=name+'_vs_PID',condition=which,period_s=T,duration_s=n*.01,position_rmse_m=m['position_rmse_m'],max_tilt_deg=m['max_tilt_deg']))
csvout(plots/'circle_common_interval.csv',common)
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':9,'axes.grid':True,'grid.alpha':.25})
def save(fig,name):fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(plots/(name+'.png'),dpi=160);plt.close(fig)
def vals(rows,key):return np.array([x[key] for x in rows],dtype=float)
fig,axs=plt.subplots(4,1,figsize=(12,12),sharex=True)
for name in runs:
 ds=[x for x in candidates if x['condition']==name];xx=vals(ds,'transitions')/1e6
 for ax,key,title in zip(axs,['completed','completed_only_position_rmse_m','completed_only_tail_omega_rms_rad_s','completed_only_tail_action_saturation'],['30초 완료 조건 수 / 7','완료 조건 평균 3D RMSE (m)','후반 body 각속도 RMS (rad/s)','후반 적용 action 포화 비율']):ax.plot(xx,vals(ds,key),'o-',label=labels[name],color=colors[name]);ax.set_ylabel(title)
axs[0].legend();axs[-1].set_xlabel('fresh-run policy transitions (million)');fig.suptitle('동일 초기화 / 자세 보상 가중치 비교 · 실패의 짧은 RMSE는 별도');save(fig,'01_hover_checkpoint_comparison')
logs={name:[json.loads(x) for x in (d/'metrics.jsonl').read_text().splitlines()] for name,d in runs.items()}
fig,axs=plt.subplots(4,4,figsize=(17,13))
for ax,key in zip(axs.ravel(),list(TERMS)+['total']):
 for name,ds in logs.items():ax.plot(vals(ds,'global_step')/1e6,vals(ds,'Reward/'+key),label=labels[name],color=colors[name],lw=.8)
 ax.set_title('Reward/'+key);ax.set_xlabel('transitions (million)')
for ax in axs.ravel()[14:]:ax.axis('off')
axs[0,0].legend(fontsize=7);fig.suptitle('실제 signed 보상 기여도 · 정의가 다른 total reward로 물리 성능 순위를 정하지 않음');save(fig,'02_training_reward_terms')
fig,axs=plt.subplots(4,2,figsize=(15,12))
keys=['Loss/policy','Loss/value','Loss/entropy_contribution','Loss/ppo_total','PPO/approx_kl','PPO/clip_fraction','Policy/std_mean','Performance/transitions_per_s']
for ax,key in zip(axs.ravel(),keys):
 for name,ds in logs.items():ax.plot(vals(ds,'global_step')/1e6,vals(ds,key),label=labels[name],color=colors[name],lw=.8)
 ax.set_title(key);ax.set_xlabel('transitions (million)')
axs[0,0].legend(fontsize=7);fig.suptitle('실제 PPO 목적함수와 학습 지표 · 시간/처리량은 실행 당시 서버 부하 포함');save(fig,'03_training_ppo_comparison')
fig,axs=plt.subplots(1,2,figsize=(14,7))
for ax,T in zip(axs,[10,5]):
 ref=reference(np.linspace(0,T,600),T)[0];ax.plot(ref[:,0],ref[:,1],'k--',label='목표 원 R=1m')
 for name in ['PID','phase11','phase12-1','phase12-2']:
  ds=seqs[(name,T)];ax.plot(np.r_[1,vals(ds,'x')],np.r_[0,vals(ds,'y')],label=labels[name],color=colors[name]);ok=next(x for x in circle if x['condition']==name and x['period_s']==T)['completed'];ax.scatter(ds[-1]['x'],ds[-1]['y'],marker='o' if ok else 'x',color=colors[name])
 ax.set(aspect='equal',xlabel='X (m)',ylabel='Y (m)',title=f'주기 {T}s / 요청 {2*T}s / × 첫 실패');ax.legend(fontsize=8)
fig.suptitle('최종 선정 best 원 궤적 · 기존 PID/phase11 수치는 저장 자료 재사용');save(fig,'04_circle_xy_comparison')
for T in [10,5]:
 export=[]
 for name in ['PID','phase11','phase12-1','phase12-2']:
  for row in seqs[(name,T)]:
   q=np.array([row[k] for k in ['qw','qx','qy','qz']]);q/=np.linalg.norm(q);Fv=row['actual_3']*(1-2*(q[1]**2+q[2]**2));export.append(dict(condition=name,period_s=T,vertical_thrust_N=Fv,**{k:row[k] for k in ['time_s','x','y','z','ex','ey','ez','vx','vy','vz','roll_deg','pitch_deg','wx','wy','wz','position_error_m','radial_error_m','phase_lag_deg','tilt_deg','action_clipping_fraction','allocation_clipping_fraction','actual_3']}))
 csvout(plots/f'circle_trajectories_T{T}.csv',export)
 fig,axs=plt.subplots(4,1,figsize=(12,12),sharex=True)
 for ax,key,title in zip(axs,['position_error_m','radial_error_m','phase_lag_deg','tilt_deg'],['3D 위치 오차 (m)','반경 오차 (m)','목표 대비 위상 지연 (deg)','기울기 (deg)']):
  for name in ['PID','phase11','phase12-1','phase12-2']:
   ds=seqs[(name,T)];ax.plot(vals(ds,'time_s'),vals(ds,key),label=labels[name],color=colors[name])
  ax.set_ylabel(title);ax.set_xlim(0,2*T)
 axs[0].legend(fontsize=8);axs[-1].set_xlabel('simulation time (s)');fig.suptitle(f'원 궤적 · 주기 {T}s · 첫 실패 이후 연결 없음');save(fig,f'05_circle_errors_T{T}')
 fig,axs=plt.subplots(3,1,figsize=(12,10),sharex=True)
 for ax,key,title in zip(axs,['roll_deg','pitch_deg','vertical_thrust_N'],['실제 roll (deg)','실제 pitch (deg)','실제 수직 추력 T cos roll cos pitch (N)']):
  for name in ['PID','phase11','phase12-1','phase12-2']:
   ds=[x for x in export if x['condition']==name];ax.plot(vals(ds,'time_s'),vals(ds,key),label=labels[name],color=colors[name])
  ax.set_ylabel(title);ax.set_xlim(0,2*T)
 axs[2].axhline(.043384*9.81,color='black',ls=':',label='mg');axs[0].legend(fontsize=8);axs[-1].set_xlabel('simulation time (s)');fig.suptitle(f'주기 {T}s · 실제 자세와 수직 추력 · 저장된 100Hz 표본');save(fig,f'06_circle_attitude_vertical_T{T}')
 fig,axs=plt.subplots(3,1,figsize=(12,10),sharex=True);tt=np.arange(2*T*100+1)*.01;ref=reference(tt,T)[0]
 for i,(ax,key) in enumerate(zip(axs,'xyz')):
  ax.plot(tt,ref[:,i],'k--',label='목표')
  for name in ['PID','phase11','phase12-1','phase12-2']:
   ds=seqs[(name,T)];ax.plot(vals(ds,'time_s'),vals(ds,key),label=labels[name],color=colors[name])
  ax.set_ylabel(key+' (m)')
 axs[0].legend(fontsize=8);axs[-1].set_xlabel('simulation time (s)');fig.suptitle(f'원 추종 XYZ / 주기 {T}s / 각 모델의 최종 선정 best');save(fig,f'07_circle_xyz_T{T}')
# Nominal late attitude and all six initial-return conditions, selected best only.
nominal={};early={}
for name,d in runs.items():
 folder=d/'evaluation/best_fixed_30s/best';nominal[name]=csvread(folder/'nominal.csv')
 for case in ['x_plus','x_minus','y_plus','y_minus','z_plus','z_minus']:early[(name,case)]=csvread(folder/(case+'.csv'))
fig,axs=plt.subplots(5,1,figsize=(12,12),sharex=True)
for ax,key in zip(axs,['roll_deg','pitch_deg','wx','wy','wz']):
 for name,ds in nominal.items():ss=[x for x in ds if x['time_s']>=29];ax.plot(vals(ss,'time_s'),vals(ss,key),label=labels[name],color=colors[name])
 ax.set_ylabel(key+(' (deg)' if 'deg' in key else ' (rad/s)'));ax.set_xlim(29,30)
axs[0].legend(fontsize=8);axs[-1].set_xlabel('simulation time (s)');fig.suptitle('최종 선정 best / nominal 후반 29~30초');save(fig,'08_hover_nominal_late_attitude')
fig,axs=plt.subplots(3,6,figsize=(22,10),sharex=True,sharey='row');cases=['x_plus','x_minus','y_plus','y_minus','z_plus','z_minus']
for j,case in enumerate(cases):
 for i,key in enumerate(['ex','ey','ez']):
  ax=axs[i,j]
  for name in runs:ss=[x for x in early[(name,case)] if x['time_s']<=5];ax.plot(vals(ss,'time_s'),vals(ss,key),label=labels[name],color=colors[name])
  ax.set_xlim(0,5);ax.axhline(0,color='black',ls=':',lw=.5)
  if i==0:ax.set_title(case)
  if j==0:ax.set_ylabel(key+' (m)')
  if i==2:ax.set_xlabel('time (s)')
axs[0,0].legend(fontsize=6);fig.suptitle('최종 선정 best / ±5cm 복귀 / 첫 0~5초');save(fig,'09_hover_return_position')
checks={k:bool(v) for k,v in checks.items()};dump_json(r/'verification.json',dict(success=all(checks.values()),checks=checks,new_training_transitions=33554432,new_optimizer_calls=327680,circle_optimizer_calls=0));assert all(checks.values()),checks
# Concrete report and reproduction commands.
lines=['Phase12 — phase11의 자세 벌점 가중치 비교',str(r),'',
'실행: phase12-1 roll/pitch=0.075(0.5배), phase12-2=0.30(2배). 기존 phase11=0.15는 저장된 기준 결과로 재사용했다. 새 학습은 정확히 두 조건이며 조건별 512회이다.',
'처음부터 학습: seed42, 동일 초기 actor/critic, log_std=-1.5, 새 Adam(빈 state), 초기 카운터0. phase11 및 두 새 실행의 첫 rollout 상태/확률 행동/물리 궤적이 bitwise 일치함을 검증했다. 이후는 보상에 따른 update로 달라진다.',
'기존 phase11 보상 함수/물리/관측/행동/학습 스크립트는 수정하지 않았다. 조건별 reward_config.json의 axis_weights.attitude[0:2]만 변경했다. yaw=1, body omega 각축=.001, 위치/속도/실패 항, 합산/단위/dt 미적용 유지.',
'N1024, rollout32, epochs5, minibatches64(size512), actor/critic[64,64]tanh, lr3e-4 fixed, clip.2,gamma.99,lambda.95,entropy.003,value1,valueclip,actor/critic gradient1,rollout advantage normalization,관측/보상 normalization 없음. dt.002/decimation5,8초 train episode,XYZ독립 uniform±.05 초기 위치,실패 조건과 모터/질량/COM/좌표계 유지.',
'Best: 0/64/.../512의 고정7조건30초 평가. 완료 개수 우선,7 미만 동률은 평균생존시간,7/7 동률은 평균위치RMSE,최종 동률은 이른 checkpoint. 학습을 모두 끝낸 뒤 확정한 가장 마지막 best.pt를 원 궤적에 사용했다. 마지막 iteration 모델 final.pt와 구분한다. 원 평가로 best를 다시 고르지 않았다.',
'학습 예산은 각 512회로 같지만 best가 선택된 iteration은 다를 수 있다. 원 평가 모델의 실제 학습량은 best_selection.json 및 circle의 reload.json 카운터를 따른다. 같은 iteration의 정책만 비교한 실험은 아니다.',
'호버 fixed7 + 기존 random20은 best/final 새 프로세스 재로드.128개 고정 관측 action 검증.기존20개는 이미 사용한 평가 목록이며 OOD/미공개 시험이 아니다. 기존 phase11 비교 평가만 저장 자료 재사용.',
'원 궤적: R=1m,center[0,0,1],CCW,yaw0,T10/5 각2lap분(20/10s),[1,0,1]정지 시작,수평/hover모터. 움직이는 목표 위치만 obs에 반영,원속도/가속도 feedforward 없음.원래 clip/allocation/RPM 지연 적용.첫 실패에 종료.실제 방위각회전량과 목표2회분 비행시간을 구분한다.',
'', '학습 결과 (phase11은 과거 실행):']
for x in training:lines.append(f"{x['condition']} | iteration {x['iterations']} | transitions {x['transitions']} | actual Adam calls {x['optimizer_updates']} | train {x['training_s']:.3f}s | {x['transitions_per_s']:.2f}trans/s | best {x['best_iteration']}회")
lines+=['','새로 수행한 합계: 33,554,432 policy transitions / 327,680 optimizer.step 호출. Adam 파라미터별 step을 합산한 수치가 아니다. phase11 과거 학습량은 합계에서 제외.','학습 시간은 초기화/초기평가/중간평가를 제외한 기존 timer 기준. 세 실행의 당시 서버 부하가 다를 수 있으므로 처리량 차이를 reward 계수만의 효과로 해석하지 않는다. cudaMemGetInfo는 장치 전체 사용량, Torch allocator peak는 해당 프로세스 allocator만 의미한다. 세부값 training_comparison.json 및 branch training_result.json.','','호버 완료율/완료 조건 평균 RMSE:']
for x in hover:lines.append(f"{x['condition']} {x['model']} {x['suite']} | {x['completed']}/{x['total']} | RMSE {x['completed_only_position_rmse_m']}m | tail omega {x['completed_only_tail_omega_rms_rad_s']}rad/s | tail speed {x['completed_only_tail_speed_rms_m_s']}m/s")
lines+=['','원 궤적:']
for x in circle:lines.append(f"{x['condition']} T{x['period_s']} | complete {x['completed']} | {x['duration_s']}/{x['requested_s']}s | {x['reasons']} | position RMSE {x['position_rmse_m']:.6f}m | radial RMSE {x['radial_rmse_m']:.6f}m | phase lag {x['phase_lag_mean_deg']:.3f}deg | tilt {x['max_tilt_deg']:.3f}deg | action clip {x['action_clip_fraction']:.5f} / allocation clip {x['allocation_clip_fraction']:.5f}")
lines+=['','해석: RMSE는 오차의 제곱평균제곱근, RMS는 제곱평균제곱근. 미완료 원 궤적의 짧은 RMSE를 완주와 같은 의미로 순위화하지 않는다. 공통 시간 비교 circle_common_interval.csv를 별도로 제공한다.','단일 seed42, 같은 새 초기화에서의 두 보상 계수 후보이다. 여러 seed의 일반적 우월성이나 계수 최적값으로 해석하지 않는다. reward/total PPO loss는 정의가 달라진 조건 간 물리 성능 순위 기준이 아니다. OOD 질량/고장 시험은 하지 않았다.','기존 phase11보다 호버 위치 오차와 자세 변동, 원 추종 성능이 각각 어떻게 달라졌는지는 hover_comparison/circle_comparison 및 플롯으로 함께 판단한다.','각 branch/circle/attitude_vertical_thrust_*에는 사후 기존 PID 외부루프 기준 목표 자세와 실제 자세, T cos roll cos pitch 그래프도 포함했다. 이 목표는 RL 내부 자세 목표가 아니다. 100Hz 저장 표본의 투영이며 500Hz 전체 force history가 아니다.','기존6006/6007 TensorBoard와 모든 이전 artifact/코드 SHA256 보존 확인. GUI는 실행하지 않았으며 명령만 제공. 추가 학습/튜닝 없음.']
lines+=['','정량 비교 결론 (이 seed 및 현재 평가 조건에 한함):']
bh=next(x for x in hover if x['condition']=='phase11' and x['model']=='best' and x['suite']=='fixed')
for name in ['phase12-1','phase12-2']:
 h=next(x for x in hover if x['condition']==name and x['model']=='best' and x['suite']=='fixed')
 if h['completed']==bh['completed']==7:
  pr=(h['completed_only_position_rmse_m']/bh['completed_only_position_rmse_m']-1)*100
  wr=(h['completed_only_tail_omega_rms_rad_s']/bh['completed_only_tail_omega_rms_rad_s']-1)*100
  lines.append(f"{name} best의 고정7조건 평균 위치 RMSE는 phase11 대비 {pr:+.2f}%, 후반 각속도 RMS는 {wr:+.2f}%. 위치 정확도와 자세 진동의 변화는 따로 판단한다.")
 else:lines.append(f"{name} best의 고정7조건 완료수 {h['completed']}/7; phase11 {bh['completed']}/7. 서로 다른 생존 구간의 RMSE로 단순 순위를 정하지 않는다.")
 for T in [10,5]:
  z=next(x for x in circle if x['condition']==name and x['period_s']==T);b=next(x for x in circle if x['condition']=='phase11' and x['period_s']==T)
  if z['completed'] and b['completed']:
   delta=(z['position_rmse_m']/b['position_rmse_m']-1)*100;lines.append(f"{name} 원 T{T}: 둘 다 예정 시간 비행 유지. phase11 대비 위치 RMSE {delta:+.2f}%, 위상 지연 {z['phase_lag_mean_deg']:.3f}deg, 최대 기울기 {z['max_tilt_deg']:.3f}deg.")
  else:lines.append(f"{name} 원 T{T}: {z['duration_s']:.2f}/{2*T}s, 완료 {z['completed']}, 최초 종료 {z['reasons']}. 조기 실패의 RMSE는 완주한 phase11과 직접 순위화하지 않는다.")
(r/'REPORT_phase12_ko.txt').write_text('\n'.join(lines)+'\n')
for name in ['phase12-1','phase12-2']:
 (r/name/'REPORT_phase12_ko.txt').write_text('\n'.join(lines[:13])+f'\n이 분기: {name}\n상위 REPORT_phase12_ko.txt 및 이 폴더의 evaluation_summary.json/training_result.json/best_selection.json 참조.\n')
free=6008
while True:
 s=socket.socket();busy=s.connect_ex(('127.0.0.1',free))==0;s.close()
 if not busy:break
 free+=1
spec=','.join(f'{name}_train:{d}/tensorboard,{name}_validation:{d}/tensorboard_validation' for name,d in runs.items())
commands=f'''cd {ROOT}\nPY={sys.executable if False else '/home/mrl_6534/miniconda3/envs/isaaclab/bin/python'}\nRUN={r}\n# Actual fresh training entrypoint (already completed; do not rerun in existing directories)\n"$PY" scripts/train_phase11.py --run-dir "$RUN/phase12-1" --iterations 512 --device cuda:0 --headless --physics isaacsim_physx\n"$PY" scripts/train_phase11.py --run-dir "$RUN/phase12-2" --iterations 512 --device cuda:0 --headless --physics isaacsim_physx\n# Evaluation only; choose branch\nBRANCH=phase12-1\n"$PY" scripts/evaluate_phase11.py --checkpoints "$RUN/$BRANCH/best.pt" --run-dir "$RUN/$BRANCH" --output "$RUN/$BRANCH/evaluation/manual_$(date +%Y%m%d_%H%M%S)" --positions "$RUN/$BRANCH/evaluation/initial_positions.json" --suite fixed --horizon 30 --device cuda:0 --headless --physics isaacsim_physx\n"$PY" scripts/evaluate_circle_phase12.py --run-dir "$RUN/$BRANCH/circle" --controller phase12_best --period 10 --output "$RUN/$BRANCH/circle/manual_T10_$(date +%Y%m%d_%H%M%S)" --device cuda:0 --headless --physics isaacsim_physx\n# GUI: same circle, --period 10 or 5. Not visually validated in this experiment.\n"$PY" scripts/evaluate_circle_phase12.py --run-dir "$RUN/$BRANCH/circle" --controller phase12_best --period 5 --output "$RUN/$BRANCH/circle/gui_T5_$(date +%Y%m%d_%H%M%S)" --device cuda:0 --visualizer kit --physics isaacsim_physx\n# Existing TensorBoard services preserved. Port {free} was free when report generated.\n"$PY" -m tensorboard.main --logdir_spec {spec} --host 127.0.0.1 --port {free}\n# Windows PowerShell tunnel\nssh -N -L {free}:127.0.0.1:{free} mrl_6534@100.79.103.47\n# http://127.0.0.1:{free}\nscp "mrl_6534@100.79.103.47:{r}/review.tar.gz" .\n'''
(r/'COMMANDS.txt').write_text(commands)
dump_json(r/'aggregate_postprocessing_time.json',dict(seconds=time.perf_counter()-t0))
print('PHASE12_REPORT_READY',str(r),flush=True)
