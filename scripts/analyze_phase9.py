"""Offline aggregate, verify and plot the fixed-budget Phase9 experiment."""
import argparse,csv,json,math,shutil,time
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter
from ppo_common import ROOT,dump_json,sha256
from phase6_common import preserved
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir).resolve();start=time.perf_counter();plots=r/'plots';plots.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':10})
labels={'parent':'부모 E_position best','control_stop_best':'정지 속도 보상 best','control_stop_final':'정지 속도 보상 final','track_return_best':'복귀 속도 추종 보상 best','track_return_final':'복귀 속도 추종 보상 final','control_stop':'정지 속도 보상','track_return':'복귀 속도 추종 보상'}
colors={'parent':'#555555','control_stop_best':'#0072B2','control_stop_final':'#56B4E9','track_return_best':'#D55E00','track_return_final':'#E69F00','control_stop':'#0072B2','track_return':'#D55E00'}
branches=['control_stop','track_return'];models=['parent']+[b+'_'+k for b in branches for k in ['best','final']]
def load(path):return json.loads(Path(path).read_text())
def write_csv(path,rows):
 keys=list(dict.fromkeys(k for row in rows for k in row))
 with Path(path).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows([{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()} for row in rows])
def read_csv(path):
 with Path(path).open() as f:return [{k:(v=='True' if v in ['True','False'] else float(v)) for k,v in row.items() if v!=''} for row in csv.DictReader(f)]
def mean(xs):return float(np.mean(xs)) if xs else None
all_results=[];summary=[];by={}
metrics=['position_rmse_m','speed_rms_m_s','velocity_error_rms_m_s','tail_position_mean_m','tail_speed_mean_m_s','tail_speed_rms_m_s','tail_velocity_error_rms_m_s','tail_body_omega_rms_rad_s','tail_z_std_m','tail_z_peak_to_peak_m','tail_z_mean_m','max_tilt_deg','action_clipping_fraction','allocation_clipping_fraction','tail_roll_rms_rad','tail_pitch_rms_rad']
for model in models:
 for suite,h in [('fixed',8),('fixed',30),('random20',30)]:
  d=r/'evaluation'/f'{model}_{suite}_{h}s';res=load(d/'results.json');by[(model,suite,h)]=res
  for x in res:all_results.append(dict(model=model,korean=labels[model],suite=suite,**x))
  completed=[x for x in res if x['completed']]
  s=dict(model=model,korean=labels[model],suite=suite,horizon_s=h,completed=len(completed),total=len(res),completion_rate=len(completed)/len(res),mean_duration_s=mean([x['duration_s'] for x in res]),failed_conditions=[dict(condition=x['condition'],duration_s=x['duration_s'],reasons=x['reasons']) for x in res if not x['completed']])
  for k in metrics:s[k+'_completed_mean']=mean([x[k] for x in completed if x.get(k) is not None])
  s['tail_bias_xyz_completed_mean']=np.mean([x['tail_error_vector_m'] for x in completed],axis=0).tolist() if completed else None
  summary.append(s)
write_csv(r/'evaluation/all_results.csv',all_results);write_csv(plots/'evaluation_summary.csv',summary);dump_json(r/'evaluation/summary.json',summary)
# Paired, identical initial-condition comparisons; exclude any condition not completed by ALL models.
paired=[]
for suite,h in [('fixed',8),('fixed',30),('random20',30)]:
 common=set.intersection(*[{x['condition'] for x in by[(m,suite,h)] if x['completed']} for m in models])
 for m in models:
  seq=[x for x in by[(m,suite,h)] if x['condition'] in common]
  paired.append(dict(model=m,suite=suite,horizon_s=h,common_completed_count=len(common),conditions=sorted(common),**{k:mean([x[k] for x in seq if x.get(k) is not None]) for k in metrics}))
write_csv(plots/'paired_completed_comparison.csv',paired);dump_json(r/'evaluation/paired_completed_comparison.json',paired)
verification={};tb_details={};training={};validation=[]
for b in branches:
 tr=load(r/b/'training_result.json');training[b]=tr;rows=[json.loads(l) for l in (r/b/'metrics.jsonl').read_text().splitlines()];csvrows=read_csv(r/b/'iterations.csv')
 ev=EventAccumulator(str(r/b/'tensorboard'),size_guidance={'scalars':0});ev.Reload();lookup={tag:{e.step:e.value for e in ev.Scalars(tag)} for tag in ev.Tags()['scalars']}
 checked=0;maxerr=0.
 for row,cr in zip(rows,csvrows):
  for k,v in row.items():
   if '/' not in k:continue
   assert cr[k]==v,(b,k,'CSV');assert row['global_step'] in lookup[k],(b,k,'step');err=abs(lookup[k][row['global_step']]-v);maxerr=max(maxerr,err);assert err==0;checked+=1
 checks=dict(rows_256=len(rows)==256,steps_exact=[x['global_step'] for x in rows]==[(i+1)*32768 for i in range(256)],additional_transitions=tr['additional_transitions']==8388608,actual_optimizer_calls=tr['additional_optimizer_updates']==81920,adam_steps=tr['adam_steps']==[225280],final_iterations=tr['iterations']==1664,final_transitions=tr['transitions']==54525952,tb_csv_json_exact=maxerr==0,reward_sum_residual=max(x['Validation/reward_sum_max_abs_error'] for x in rows)<1e-4,backward_loss_residual=max(x['Validation/ppo_loss_residual'] for x in rows)<1e-6,episode_step_accounting=tr['episode_accounting']['step_residual']==0)
 # Validate signed means and original vs diagnostic total reward.
 for row in rows:
  assert abs(sum(row['Reward/step/'+t] for t in ['xy','z','velocity','tilt','angular_velocity','yaw','termination'])-row['Reward/step/total'])<1e-5
  assert row['Reward/step/total']==row['Reward/common_'+('stop' if b=='control_stop' else 'return')]
 checks['signed_contributions_and_selected_common_reward']=True
 tb_details[b]=dict(checks=checks,checked_scalar_values=checked,max_error=maxerr,first_middle_last_steps=[rows[i]['global_step'] for i in [0,127,255]],tag_count=len(lookup))
 assert all(checks.values()),checks
 writer=SummaryWriter(str(r/b/'tensorboard'));selection=load(r/b/'best_selection.json')
 for c in selection['candidates']:
  res=c['results'];complete=[x for x in res if x['completed']];it=c['additional_iteration'];v=dict(branch=b,additional_iteration=it,additional_transitions=it*32768,completed=len(complete),duration_mean_s=mean([x['duration_s'] for x in res]),position_rmse_m=mean([x['position_rmse_m'] for x in complete]),tail_speed_mean_m_s=mean([x['tail_speed_mean_m_s'] for x in complete]),tail_speed_rms_m_s=mean([x['tail_speed_rms_m_s'] for x in complete]),tail_omega_rms_rad_s=mean([x['tail_body_omega_rms_rad_s'] for x in complete]),tail_z_std_m=mean([x['tail_z_std_m'] for x in complete]));validation.append(v)
  for key,val in v.items():
   if key not in ['branch','additional_transitions'] and val is not None and it*32768 not in lookup.get('EvalValidation/'+key,{}):writer.add_scalar('EvalValidation/'+key,val,it*32768)
 writer.flush();writer.close()
 write_csv(plots/(b+'_training.csv'),rows)
validation_all=[]
for b in branches:
 for c in load(r/b/'best_selection.json')['candidates']:
  validation_all.extend(dict(stage='validation',branch=b,additional_iteration=c['additional_iteration'],**x) for x in c['results'])
write_csv(r/'evaluation/validation_all_results.csv',validation_all)
write_csv(r/'evaluation/all_results.csv',[dict(stage='final',**x) for x in all_results]+validation_all)
checkpoint_checks=[]
for b in branches:
 for it in [64,128,192,256]:
  cp=r/b/f'checkpoint_{it:03d}.pt';state=torch.load(cp,map_location='cpu',weights_only=False);info=state['infos'];steps=sorted(set(int(v['step']) for v in state['optimizer_state_dict']['state'].values()))
  ok=(info['additional_completed_iterations']==it and info['completed_iterations']==1408+it and state['iter']==1408+it-1 and info['collected_transitions']==46137344+it*32768 and info['additional_optimizer_updates']==it*320 and steps==[143360+it*320])
  assert ok;checkpoint_checks.append(dict(branch=b,additional_iteration=it,sha256=sha256(cp),adam_steps=steps,success=ok))
verification['checkpoint_counters']=checkpoint_checks
verification['training_tb']=tb_details;verification['reward_math']=load(r/'reward_ready.json');verification['preflight']=load(r/'comparison_ready.json');verification['first_rollout']=load(r/'track_return/first_rollout_comparison.json')
verification['reloads']=[];verification['terminal_capture']=True
for d in [r/'evaluation'/f'{m}_{s}_{h}s' for m in models for s,h in [('fixed',8),('fixed',30),('random20',30)]]:
 for ck in load(d/'reload_checks.json'):
  assert ck['reference_count']==128 and ck['max_action_error']<=1e-6 and ck['optimizer_updates_in_process']==0;verification['reloads'].append(ck)
 for result in load(d/'results.json'):
  path=d/result['checkpoint_name']/(result['condition']+'.csv');seq=read_csv(path)
  assert not any(x['terminated'] or x['truncated'] for x in seq[:-1])
  if not result['completed']:assert result['tail_speed_rms_m_s'] is None
  assert len(seq)==result['steps']
verification['preserved_project']=preserved(r/'evidence/project_before_sha256.json',ROOT);verification['preserved_prior_runs']=preserved(r/'evidence/prior_runs_sha256.json',ROOT)
assert verification['preserved_project']['success'] and verification['preserved_prior_runs']['success']
verification['success']=True;dump_json(r/'verification.json',verification);write_csv(plots/'validation.csv',validation)

def save(fig,name):fig.tight_layout();fig.savefig(plots/(name+'.png'),dpi=150);plt.close(fig)
def training_plot(name,keys,title):
 fig,axes=plt.subplots(len(keys),1,figsize=(12,max(4,len(keys)*2.5)),sharex=True);axes=np.atleast_1d(axes)
 for ax,(key,unit) in zip(axes,keys):
  for b in branches:
   rows=read_csv(plots/(b+'_training.csv'));ax.plot([x['global_step'] for x in rows],[x.get(key,np.nan) for x in rows],label=labels[b],color=colors[b],lw=1)
  ax.set_ylabel(unit);ax.set_title(key);ax.grid(alpha=.25);ax.legend()
 axes[-1].set_xlabel('이번 추가 policy transitions');fig.suptitle(title+' · 학습 집계(원시 iteration 평균)',y=1.005);save(fig,name)
training_plot('training_reward',[(f'Reward/step/{k}','weighted reward / policy step') for k in ['total','xy','z','velocity','tilt','angular_velocity','yaw','termination']], '정지 속도 보상 / 복귀 속도 추종 보상')
training_plot('common_rewards',[('Reward/common_stop','공통 정지 기준 reward'),('Reward/common_return','공통 복귀 기준 reward')],'진단 전용 common reward (학습 reward에 중복 합산하지 않음)')
training_plot('ppo_loss',[(k,'loss') for k in ['Loss/policy','Loss/value','Loss/entropy_contribution','Loss/ppo_total']],'실제 backward PPO loss')
training_plot('ppo_policy',[(k,u) for k,u in [('PPO/approx_kl','KL'),('PPO/clip_fraction','ratio outside [0.8,1.2]'),('Policy/entropy','entropy'),*[(f'Policy/std_axis_{i}','Gaussian std') for i in range(4)]]],'PPO 비율 clipping과 Gaussian 행동분포')
training_plot('training_flight',[(k,u) for k,u in [('Flight/position_rmse_m','m'),('Velocity/speed_rms_m_s','m/s'),('Velocity/error_rms_m_s','m/s'),('Flight/body_omega_rms_rad_s','rad/s'),('Actuator/action_clipping_fraction','fraction'),('Actuator/allocation_clipping_fraction','fraction')]],'학습 전체 환경의 위치·속도·물리 제한')
training_plot('performance',[(k,u) for k,u in [('Performance/transitions_per_s','transitions/s'),('Performance/training_elapsed_s','s'),('Performance/cuda_device_used_MiB','whole device MiB'),('Performance/torch_peak_allocated_MiB','PyTorch allocator MiB')]],'학습 처리량 및 메모리 (평가 제외)')
fig,axs=plt.subplots(4,1,figsize=(11,12),sharex=True)
for ax,key,u in zip(axs,['completed','position_rmse_m','tail_speed_rms_m_s','tail_z_std_m'],['완료 / 7','RMSE (m), 완료 조건만','후반 속력 RMS (m/s)','후반 Z std (m)']):
 for b in branches:
  seq=[x for x in validation if x['branch']==b];ax.plot([x['additional_transitions'] for x in seq],[x[key] for x in seq],'-o',label=labels[b],color=colors[b])
 ax.set_ylabel(u);ax.grid(alpha=.25);ax.legend()
axs[-1].set_xlabel('이번 추가 policy transitions');fig.suptitle('고정 7조건 · 8초 검증 · best 선택에 사용');save(fig,'validation_progress')
# Full trajectory comparisons, plus early-return and late-segment views.
def trajectory(model,case,h=30):
 d=r/'evaluation'/f'{model}_fixed_{h}s';cp=load(d/'results.json')[0]['checkpoint_name'];return read_csv(d/cp/(case+'.csv'))
for case in ['nominal','x_plus','z_minus']:
 sequences={m:trajectory(m,case) for m in models}
 for m,seq in sequences.items():write_csv(plots/f'trajectory_{case}_{m}_30s.csv',seq)
 for group,fields,ylabels in [('position',[['x'],['y'],['z']],["X (m)","Y (m)","Z (m)"]),('velocity',[[f'v{ax}',f'vdes_{ax}'] for ax in 'xyz'],['X 실제/복귀 속도 (m/s)','Y 실제/복귀 속도 (m/s)','Z 실제/복귀 속도 (m/s)']),('attitude',[['roll_deg','pitch_deg'],['wx','wy','wz'],['ez']],["roll / pitch (deg)","body ω (rad/s)","Z 오차 (m)"])]:
  fig,axs=plt.subplots(len(fields),1,figsize=(13,10),sharex=True)
  for ax,fs,yl in zip(axs,fields,ylabels):
   for m,seq in sequences.items():
    for fi,f in enumerate(fs):ax.plot([x['time_s'] for x in seq],[x[f] for x in seq],label=labels[m]+' '+f,color=colors[m],ls=['-','--',':'][fi],lw=.8,alpha=.8)
   if group=='position':ax.axhline(1 if fs==['z'] else 0,color='black',ls=':',label='고정 목표')
   ax.set_ylabel(yl);ax.grid(alpha=.2);ax.legend(fontsize=7,ncol=3)
  axs[-1].set_xlabel('evaluation simulation time (s)');fig.suptitle(f'부모 / 정지 속도 보상 / 복귀 속도 추종 보상 · best/final · {case} · 30초');save(fig,f'{case}_{group}_30s')
 # Individual actuator plots include every axis and requested/actual physical wrench.
 for m in models:
  seq=sequences[m];tm=[x['time_s'] for x in seq];fig,axs=plt.subplots(5,2,figsize=(14,14),sharex=True)
  for i in range(4):
   axs[i,0].plot(tm,[x[f'raw_action_{i}'] for x in seq],label='raw actor mean');axs[i,0].plot(tm,[x[f'action_{i}'] for x in seq],label='applied normalized action',alpha=.7);axs[i,0].set_ylabel(f'action {i} (무차원)')
   axs[i,1].plot(tm,[x[f'request_{i}'] for x in seq],label='requested');axs[i,1].plot(tm,[x[f'actual_{i}'] for x in seq],label='actual');axs[i,1].set_ylabel(('τ'+str(i)+' (N m)') if i<3 else 'Fz (N)')
  for i in range(4):axs[4,0].plot(tm,[x[f'thrust_{i}'] for x in seq],label=f'motor {i}')
  axs[4,0].set_ylabel('실제 rotor thrust (N)');axs[4,1].plot(tm,[x['allocation_clipping_fraction'] for x in seq]);axs[4,1].set_ylabel('allocation clip fraction')
  for ax in axs.flat:ax.grid(alpha=.2);ax.legend(fontsize=7)
  for ax in axs[-1]:ax.set_xlabel('evaluation simulation time (s)')
  fig.suptitle(f'{labels[m]} · {case} · 30초 · action/물리 wrench/추력');save(fig,f'{case}_{m}_actuators_30s')
# Oscillation and bias: display last second raw waveforms and full final 10s amplitude summaries.
fig,axs=plt.subplots(3,1,figsize=(13,10),sharex=True)
for m in models:
 seq=[x for x in trajectory(m,'nominal') if x['time_s']>29]
 for ax,key,unit in zip(axs,['roll_deg','wx','ez'],['실제 roll (deg)','body roll rate (rad/s)','Z 오차 (m)']):ax.plot([x['time_s'] for x in seq],[x[key] for x in seq],label=labels[m],color=colors[m]);ax.set_ylabel(unit);ax.grid(alpha=.2);ax.legend(ncol=3,fontsize=8)
axs[-1].set_xlabel('evaluation simulation time (s), 29–30초 확대');fig.suptitle('nominal · 부모 / 두 조건 best/final · 원본 진동 파형');save(fig,'nominal_tail_oscillation_30s')
# Completion and accuracy kept together; no short-failure RMSE used as a success metric.
fig,axs=plt.subplots(2,2,figsize=(14,10));x=np.arange(5);w=.25
for j,(suite,h) in enumerate([('fixed',8),('fixed',30),('random20',30)]):
 seq=[next(s for s in summary if s['model']==m and s['suite']==suite and s['horizon_s']==h) for m in models]
 for ax,key,unit in zip(axs.flat,['completion_rate','position_rmse_m_completed_mean','tail_speed_rms_m_s_completed_mean','tail_z_std_m_completed_mean'],['완료율','3D 위치 RMSE (m), 완료만','후반 속력 RMS (m/s), 완료만','후반 Z std (m), 완료만']):ax.bar(x+(j-1)*w,[s[key] if s[key] is not None else np.nan for s in seq],width=w,label=f'{suite} {h}초');ax.set_ylabel(unit);ax.set_xticks(x,[labels[m] for m in models],rotation=15,ha='right');ax.grid(axis='y',alpha=.2);ax.legend()
fig.suptitle('부모 / 정지 속도 보상 / 복귀 속도 추종 보상 · best/final · 고정 및 미선택 random20');save(fig,'final_performance')
dump_json(r/'analysis_result.json',dict(success=True,summary=summary,training=training,plots=len(list(plots.glob('*.png'))),analysis_wall_s=time.perf_counter()-start))
print('ANALYSIS_OK',len(all_results),len(list(plots.glob('*.png'))),flush=True)
