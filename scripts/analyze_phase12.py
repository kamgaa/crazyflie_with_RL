"""Offline statistics, plots, exact scalar audit and phase11 report."""
import argparse,json,csv,math,os,time,shutil,hashlib,subprocess,socket
from pathlib import Path
import numpy as np
import phase10_tables as tables
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter
from ppo_common import ROOT,dump_json,sha256
from phase6_common import preserved
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir).resolve();tick=time.perf_counter();plots=r/'plots'
font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
if font.exists():font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'axes.unicode_minus':False,'font.size':9})
plan=json.loads((r/'experiment_plan.json').read_text());train=json.loads((r/'training_result.json').read_text());selection=json.loads((r/'best_selection.json').read_text());best_it=selection['selected']['additional_iteration']
labels={'phase11':'기존 phase11 best (512회, 계수 0.15)','best':f"{plan['branch']} best / {best_it}회 / 계수 {plan['attitude_weight_roll']}",'final':f"{plan['branch']} final / 512회"}
colors={'phase11':'#555555','best':'#329449','final':'#d67420'}

def flatten(d,prefix=''):
 out={}
 for k,v in d.items():
  key=prefix+k
  if isinstance(v,dict):out.update(flatten(v,key+'/'))
  elif isinstance(v,list):
   if all(isinstance(x,(int,float,bool)) for x in v):out.update({key+f'_{i}':x for i,x in enumerate(v)})
   else:out[key]=json.dumps(v,ensure_ascii=False)
  else:out[key]=v
 return out

def aggregate(data):
 completed=[d for d in data if d['completed']]
 out={'completed_count':len(completed),'conditions':len(data),'completion_fraction':len(completed)/len(data),'mean_duration_s':np.mean([d['duration_s'] for d in data]),'completed_only_mean_position_rmse_m':None,'completed_only_mean_tail_speed_rms_m_s':None}
 for key in ['position_rmse_m','tail_speed_rms_m_s','tail_velocity_error_rms_m_s','tail_body_omega_rms_rad_s','tail_z_std_m','tail_roll_rms_rad','tail_pitch_rms_rad','max_tilt_deg','full_raw_action_exceedance_fraction','tail_raw_action_exceedance_fraction','tail_applied_action_saturation_fraction','tail_allocation_clipping_fraction']:
  out['completed_only_mean_'+key]=float(np.mean([d[key] for d in completed])) if completed else None
 for key in ['tail_error_vector_m','tail_position_error_m_std','tail_attitude_rad_std','tail_thrust_N_std','tail_rpm_std','tail_raw_action_std','tail_applied_action_std','tail_raw_action_rate_rms_s_inv','tail_applied_action_rate_rms_s_inv','tail_actual_wrench_std','tail_requested_wrench_std']:
  out['completed_only_mean_'+key]=np.mean([d[key] for d in completed],axis=0).tolist() if completed else None
 return out
all_rows=[];candidate_rows=[];candidate_details=[]
for it in range(0,513,64):
 data=json.loads((r/'evaluation'/f'validation_{it:03d}'/'results.json').read_text());agg=aggregate(data)
 cp=json.loads((r/'evaluation'/f'validation_{it:03d}'/'reload_checks.json').read_text())[0]
 row=dict(additional_iteration=it,additional_transitions=it*32768,additional_optimizer_updates=it*320,training_elapsed_s=0 if it==0 else cp['loaded_infos']['training_elapsed_s'],**flatten(agg))
 candidate_rows.append(row)
 for d in data:candidate_details.append(dict(additional_iteration=it,**flatten(d)))
tables.Table(candidate_rows).to_csv(plots/'checkpoint_comparison.csv',index=False);tables.Table(candidate_details).to_csv(plots/'checkpoint_conditions.csv',index=False)
summary=[];nominal={};spectra={};final_datasets={}
for label in labels:
 for suite in ['fixed','random20']:
  folder=r/'evaluation'/f'{label}_{suite}_30s';data=json.loads((folder/'results.json').read_text());final_datasets[(label,suite)]=data
  agg=aggregate(data);summary.append(dict(model=label,label=labels[label],suite=suite,**flatten(agg)))
  for d in data:all_rows.append(dict(model=label,suite=suite,**flatten(d)))
  if suite=='fixed':
   stem=Path(data[0]['checkpoint']).stem
   nominal[label]=tables.read_csv(folder/stem/'nominal.csv');spectra[label]=json.loads((folder/stem/'nominal_spectra.json').read_text())
tables.Table(summary).to_csv(plots/'final_comparison.csv',index=False);tables.Table(all_rows).to_csv(plots/'all_evaluation_metrics.csv',index=False)
dump_json(r/'evaluation_summary.json',dict(candidates=candidate_rows,final=summary))
paired=[]
for suite in ['fixed','random20']:
 maps={m:{d['condition']:d for d in final_datasets[(m,suite)]} for m in labels}
 common=[c for c in maps['phase11'] if all(maps[m][c]['completed'] for m in labels)]
 for m in labels:
  ds=[maps[m][c] for c in common];paired.append(dict(model=m,suite=suite,common_completed_conditions=common,**aggregate(ds)) if ds else dict(model=m,suite=suite,common_completed_conditions=[]))
dump_json(r/'paired_completed_comparison.json',paired)
# Validation metrics use added transitions, like training. No historical reward replication.
eval_records=[];writer=None if any((r/'tensorboard_validation').glob('events.out.tfevents.*')) else SummaryWriter(str(r/'tensorboard_validation'))
for row in candidate_rows:
 step=row['additional_transitions']
 for key,value in row.items():
  if isinstance(value,(int,float)) and value is not None and math.isfinite(value):
   v=float(np.float32(value));tag='EvalFixed30/'+key
   if writer is not None:writer.add_scalar(tag,v,step)
   eval_records.append(dict(tag=tag,global_step=step,value=v))
if writer is not None:writer.flush();writer.close()
tables.Table(eval_records).to_csv(r/'evaluation_scalars.csv',index=False);dump_json(r/'evaluation_scalars.json',eval_records)
metrics=[json.loads(s) for s in (r/'metrics.jsonl').read_text().splitlines()];df=tables.Table(metrics)
df.to_csv(plots/'training_metrics.csv',index=False)
def figure(name,nrows,ncols=1,figsize=None):
 fig,axs=plt.subplots(nrows,ncols,figsize=figsize or (11,2.7*nrows),squeeze=False);return fig,axs.ravel()
def save(fig,name):
 fig.tight_layout(rect=(0,0,1,.965));fig.savefig(plots/(name+'.png'),dpi=150,bbox_inches='tight');plt.close(fig)
def training_plot(name,groups):
 fig,axs=figure(name,len(groups))
 for ax,(title,keys) in zip(axs,groups):
  for k in keys:ax.plot(df.global_step/1e6,df[k],lw=.8,label=k)
  ax.set(title=title,xlabel='새 실행 policy transitions [million]');ax.grid(alpha=.25);ax.legend(fontsize=7,ncol=3)
 fig.suptitle('자세 벌점 가중치 비교: 축별 보상과 완화된 자세 벌점 / 원시 iteration 통계',y=1.005);save(fig,name)
training_plot('training_reward_loss', [('보상 기여도 / policy step 평균',[f'Reward/{x}' for x in ['position_x','position_y','position_z','velocity_x','velocity_y','velocity_z','roll','pitch','yaw','omega_x','omega_y','omega_z','failure','total']]),('실제 PPO backward loss',['Loss/policy','Loss/value','Loss/entropy_contribution','Loss/ppo_total']),('PPO KL / probability-ratio clip fraction',['PPO/approx_kl','PPO/clip_fraction']),('Gaussian 행동분포 표준편차',[f'Policy/std_axis_{i}' for i in range(4)]),('Episode 길이 [s] / return',['Episode/length_s','Episode/return'])] if 'Episode/return' in df else [('보상 기여도',[f'Reward/{x}' for x in ['position_x','position_y','position_z','velocity_x','velocity_y','velocity_z','roll','pitch','yaw','omega_x','omega_y','omega_z','failure','total']]),('실제 PPO backward loss',['Loss/policy','Loss/value','Loss/entropy_contribution','Loss/ppo_total']),('PPO KL / ratio clipping',['PPO/approx_kl','PPO/clip_fraction']),('Gaussian std',[f'Policy/std_axis_{i}' for i in range(4)])])
cd=tables.Table(candidate_rows);x=cd.additional_transitions/1e6
for name,groups in [('validation_position',[('30초 완료율','completion_fraction'),('완료 조건 평균 3D RMSE [m]','completed_only_mean_position_rmse_m'),('후반 속력 RMS [m/s]','completed_only_mean_tail_speed_rms_m_s')]),('validation_oscillation',[('후반 body 각속도 RMS [rad/s]','completed_only_mean_tail_body_omega_rms_rad_s'),('후반 roll RMS [rad]','completed_only_mean_tail_roll_rms_rad'),('후반 pitch RMS [rad]','completed_only_mean_tail_pitch_rms_rad'),('후반 적용 action 포화 비율','completed_only_mean_tail_applied_action_saturation_fraction')])]:
 fig,axs=figure(name,len(groups)+1)
 for ax,(title,key) in zip(axs,groups):ax.plot(x,cd[key],'o-');ax.set(title=title,xlabel='새 실행 policy transitions [million]');ax.grid(alpha=.3)
 key='completed_only_mean_tail_error_vector_m' if name=='validation_position' else 'completed_only_mean_tail_thrust_N_std'
 for i in range(3 if name=='validation_position' else 4):axs[-1].plot(x,np.array([z.get(f'{key}_{i}',np.nan) for z in candidate_rows],dtype=float),'o-',label=('XYZ'[i] if name=='validation_position' else f'rotor {i}'))
 axs[-1].set(title='20~30초 위치 편향 [m]' if name=='validation_position' else '20~30초 모터 추력 std [N]',xlabel='새 실행 policy transitions [million]');axs[-1].legend();axs[-1].grid(alpha=.3)
 base=next(z for z in summary if z['model']=='phase11' and z['suite']=='fixed')
 for ax,(title,key) in zip(axs,groups):
  if base.get(key) is not None:ax.axhline(base[key],ls='--',color=colors['phase11'],label='phase11 final 참고');ax.legend(fontsize=7)
 fig.suptitle('자세 벌점 가중치 비교 / 고정 7조건 30초 / 완료 조건 평균·완료율 함께 확인',y=1.005);save(fig,name)
# Common nominal full trajectory data, exact source values.
nominal_csv=tables.concat([d.assign(model=m,label=labels[m]) for m,d in nominal.items()],ignore_index=True);nominal_csv.to_csv(plots/'nominal_30s_comparison.csv',index=False)
fig,axs=figure('nominal_states',3,3,(15,9))
for j,key in enumerate(['ex','ey','ez','vx','vy','vz','roll_deg','pitch_deg','yaw_rad']):
 for m,d in nominal.items():axs[j].plot(d.time_s,d[key],color=colors[m],lw=.65,label=labels[m])
 axs[j].set(title=key+(' [m]' if j<3 else ' [m/s]' if j<6 else ' [deg]' if j<8 else ' [rad]'),xlabel='simulation time [s]');axs[j].grid(alpha=.25);axs[j].set_xlim(0,30)
axs[0].legend(fontsize=7);fig.suptitle('Nominal 0~30초 / 목표 위치 대비 오차·실제 속도·Euler 자세');save(fig,'nominal_states_30s')
fig,axs=figure('zoom',5,1,(12,12))
for ax,key in zip(axs,['roll_deg','pitch_deg','wx','wy','wz']):
 for m,d in nominal.items():z=d[(d.time_s>=29)&(d.time_s<=30)];ax.plot(z.time_s,z[key],lw=.8,color=colors[m],label=labels[m])
 ax.set(title=key+(' [deg]' if 'deg' in key else ' [rad/s]'),xlabel='simulation time [s]',xlim=(29,30));ax.grid(alpha=.3)
axs[0].legend(fontsize=7);fig.suptitle('Nominal 29~30초 확대 / 실제 자세·body 각속도');save(fig,'nominal_attitude_29_30s')
for m,d in nominal.items():
 fig,axs=figure('commands',6,2,(15,16))
 for i in range(4):
  axs[i].plot(d.time_s,d[f'raw_action_{i}'],label='raw actor mean',lw=.7);axs[i].plot(d.time_s,d[f'action_{i}'],label='applied clipped',lw=.7,alpha=.75);axs[i].set_title(f'action {i} [dimensionless]')
  axs[4+i].plot(d.time_s,d[f'request_{i}'],label='requested',lw=.7);axs[4+i].plot(d.time_s,d[f'actual_{i}'],label='actual motor wrench',lw=.7);axs[4+i].set_title(['roll torque [N m]','pitch torque [N m]','yaw torque [N m]','Fz [N]'][i])
 for i in range(4):
  axs[8].plot(d.time_s,d[f'motor_rpm_{i}'],lw=.6,label=f'rotor {i}');axs[9].plot(d.time_s,d[f'thrust_{i}'],lw=.6,label=f'rotor {i}')
  tail=d[(d.time_s>=29)&(d.time_s<=30)];axs[10].plot(tail.time_s,tail[f'raw_action_{i}'],lw=.8,label=f'action {i}');axs[11].plot(tail.time_s,tail[f'thrust_{i}'],lw=.8,label=f'rotor {i}')
 axs[8].set_title('rotor speed [RPM]');axs[9].set_title('rotor thrust [N]');axs[10].set_title('29~30초 raw action');axs[11].set_title('29~30초 rotor thrust [N]')
 for j,ax in enumerate(axs):ax.set_xlabel('simulation time [s]');ax.set_xlim((0,30) if j<10 else (29,30));ax.grid(alpha=.25);ax.legend(fontsize=7)
 fig.suptitle(labels[m]+' / Nominal 30초 / action·요청/실제 wrench·모터');save(fig,f'nominal_actuator_{m}')
fig,axs=figure('spectra',5,1,(12,13));spcsv=[]
for ax,key in zip(axs,['roll_rad','pitch_rad','wx','wy','wz']):
 for m,sp in spectra.items():
  if 'tail' not in sp:continue
  z=sp['tail'][key];f=np.array(z['frequency_hz']);power=np.array(z['psd']);ax.semilogy(f,np.maximum(power,1e-30),label=labels[m],color=colors[m],lw=.8)
  spcsv.extend(dict(model=m,signal=key,frequency_hz=ff,psd=vv) for ff,vv in zip(f,power))
 ax.set(title=key+' PSD ['+('rad²/Hz' if key.endswith('rad') else '(rad/s)²/Hz')+']',xlabel='frequency [Hz]',xlim=(0,50));ax.grid(alpha=.25)
axs[0].legend(fontsize=7);fig.suptitle('Nominal 20~30초 / 100Hz, mean detrend, symmetric Hann, one-sided PSD');save(fig,'nominal_tail_spectra');tables.Table(spcsv).to_csv(plots/'nominal_tail_spectra.csv',index=False)
# Every common scalar is generated from the same float32 record as TB.
ea=EventAccumulator(str(r/'tensorboard'),size_guidance={'scalars':0});ea.Reload();checks={};mismatch=[]
for tag in ea.Tags()['scalars']:
 events=ea.Scalars(tag);expected=[row for row in metrics if tag in row]
 if len(events)!=len(expected):mismatch.append([tag,'count',len(events),len(expected)]);continue
 for ev,row in zip(events,expected):
  if ev.step!=row['global_step'] or ev.value!=row[tag]:mismatch.append([tag,ev.step,'value/step']);break
checks['training_tb_csv_json_all_values']=not mismatch
csv_rows=tables.read_csv(r/'iterations.csv');checks['csv_rows_512']=len(csv_rows)==512
for idx in [0,255,511]:
 for tag in ea.Tags()['scalars']:
  if tag in metrics[idx]:
   assert math.isclose(float(csv_rows.iloc[idx][tag]),metrics[idx][tag],rel_tol=1e-12,abs_tol=1e-12)
checks['first_middle_last_csv_exact_with_decimal_roundtrip_tolerance']=True
checks['steps']=df.global_step.tolist()==list(range(32768,16777216+1,32768))
checks['reward_sum']=float(df['Validation/reward_sum_max_abs_error'].max())<1e-4
checks['actual_backward_loss']=float(df['Validation/ppo_loss_residual'].max())<1e-6
checks['budget']=(train['additional_iterations'],train['additional_transitions'],train['additional_optimizer_updates'],train['adam_steps'])==(512,16777216,163840,[163840])
checks['focused_validation']=json.loads((r/'focused_verification.json').read_text())['success']
checks['reward_unit_tests']=json.loads((r/'reward_ready.json').read_text())['success']
checks['fresh_initialization']=json.loads((r/'fresh_verification.json').read_text())['success'] and not json.loads((r/'fresh_verification.json').read_text())['checkpoint_loaded']
checks['prior_artifacts_preserved']=all(sha256(Path(f))==h for f,h in json.loads((r/'evidence/prior_artifacts_sha256.json').read_text()).items())
checks['existing_sources_preserved']=preserved(r/'evidence/project_before_sha256.json',ROOT)['success']
reloaded=[v for f in (r/'evaluation').glob('*/reload_checks.json') for v in json.loads(f.read_text())]
checks['reload_fixed128']=all(v['reference_count']==128 and v['max_action_error']<=1e-6 and v['optimizer_updates_in_process']==0 for v in reloaded)
checks['evaluation_isolation']=all(all(x['checks'].values()) for x in json.loads((r/'evaluation_isolation.json').read_text()))
check_eval=EventAccumulator(str(r/'tensorboard_validation'),size_guidance={'scalars':0});check_eval.Reload()
checks['evaluation_tb_csv_json']=all([(e.step,e.value) for e in check_eval.Scalars(tag)]==[(x['global_step'],x['value']) for x in eval_records if x['tag']==tag] for tag in check_eval.Tags()['scalars'])
trajectory_events_checked=0
for folder in (r/'evaluation').glob('*/*/tensorboard_time'):
    te=EventAccumulator(str(folder),size_guidance={'scalars':0});te.Reload();cache={}
    for tag in te.Tags()['scalars']:
        case,key=tag.split('/',1)
        if case not in cache:cache[case]=tables.read_csv(folder.parent/(case+'.csv')).set_index('step')
        for event in te.Scalars(tag):
            expected=float(np.float32(cache[case].loc[event.step,key]))
            assert event.value==expected,(folder,tag,event.step,event.value,expected)
            trajectory_events_checked+=1
checks['trajectory_tb_csv_float32']=trajectory_events_checked>0
import torch
saved=torch.load(r/'final.pt',map_location='cpu',weights_only=False)
checks['saved_checkpoint_counts']=(saved['iter'],saved['infos']['completed_iterations'],saved['infos']['collected_transitions'],saved['infos']['optimizer_updates'])==(511,512,16777216,163840)
checks['saved_adam_steps']=sorted(set(int(x['step']) for x in saved['optimizer_state_dict']['state'].values()))==[163840]
ref=torch.load(r/'initial_reference.pt',map_location='cpu',weights_only=True);obs=ref['observations']
checks['reference_observations_valid']=obs.shape==(128,15) and bool(torch.isfinite(obs).all()) and bool(torch.allclose(obs[:,6:10].norm(dim=-1),torch.ones(128),atol=1e-6))

checks={k:bool(v) for k,v in checks.items()};dump_json(r/'verification.json',dict(success=all(checks.values()),checks=checks,mismatches=mismatch,max_reload_action_error=max(v['max_action_error'] for v in reloaded),reward_max_error=float(df['Validation/reward_sum_max_abs_error'].max()),loss_max_residual=float(df['Validation/ppo_loss_residual'].max()),trajectory_scalar_values=trajectory_events_checked,training_scalar_tags=len(ea.Tags()['scalars']),training_scalar_values=sum(len(ea.Scalars(t)) for t in ea.Tags()['scalars'])))
assert all(checks.values()),checks
# Remaining report/commands and packaging delegated to the report entry point.
dump_json(r/'postprocessing_time.json',dict(analysis_and_plots_s=time.perf_counter()-tick))
print('ANALYSIS_DONE',json.dumps(summary,ensure_ascii=False),flush=True)
