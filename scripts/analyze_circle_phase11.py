"""Phase11 circle evaluation report and historical comparison; zero optimizer updates."""
import argparse,json,csv,math,shutil,time,tarfile,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from ppo_common import ROOT,dump_json,sha256,protected_hashes
from circle_eval_common import reference,segment_metrics
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir).resolve();plan=json.loads((r/'experiment_plan.json').read_text());old=Path(plan['historical_run'])
def read(p):return json.loads(Path(p).read_text())
def csvwrite(p,rows):
 keys=list(dict.fromkeys(k for x in rows for k in x))
 with Path(p).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
names=['PID','Q_base','E_position','phase9_best','phase11_best']
labels={'PID':'PID (기존 평가)','Q_base':'phase8 Q best (기존 평가)','E_position':'phase8 E best (기존 평가)','phase9_best':'phase9 복귀 속도 best (기존 평가)','phase11_best':'phase11 축별 보상 best 512회 (새 평가)'}
colors={'PID':'#0072B2','Q_base':'#999999','E_position':'#009E73','phase9_best':'#CC79A7','phase11_best':'#D55E00'}
results={};seqs={};summary=[];common=[];checks={};scalars=[]
for T in [10,5]:
 for name in names:
  folder=(r/'cases'/f'phase11_best_T{T}' if name=='phase11_best' else Path(plan['historical_phase9'])/'cases'/f'track_return_T{T}' if name=='phase9_best' else old/'cases'/f'{name}_T{T}');z=read(folder/'result.json');rows=read(folder/'trajectory.json')['trajectory'];results[(name,T)]=z;seqs[(name,T)]=rows;m=z['metrics']
  summary.append(dict(controller=name,source='new' if name=='phase11_best' else 'historical_reused',period_s=T,requested_s=2*T,completed=m['completed_two_reference_laps'],duration_s=m['duration_s'],reason=','.join(m['reasons']),position_rmse_m=m['position_rmse_m'],radial_rmse_m=m['radial_error_rmse_m'],phase_lag_mean_deg=m['phase_lag_mean_deg'],max_tilt_deg=m['max_tilt_deg'],body_omega_rms_rad_s=m['body_omega_rms_rad_s'],reference_laps=m['reference_laps_elapsed'],measured_azimuth_revolutions=m['measured_azimuth_revolutions'],action_clipping_fraction=m['action_clipping_fraction'],allocation_clipping_fraction=m['allocation_clipping_fraction'],velocity_circle_reference_rmse_m_s=m['velocity_tracking_rmse_m_s']))
  if name!='phase11_best':continue
  checks[f'T{T}_runtime']=z['success'];checks[f'T{T}_optimizer_zero']=z['optimizer_updates']==0;checks[f'T{T}_reload']=read(folder/'reload.json')['bitwise_equal']
  pos=np.array([[x['x'],x['y'],x['z']] for x in rows]);target=np.array([[x['target_x'],x['target_y'],x['target_z']] for x in rows]);err=np.array([[x['ex'],x['ey'],x['ez']] for x in rows])
  checks[f'T{T}_reference']=np.max(np.abs(target-reference(np.array([x['time_s'] for x in rows]),T)[0]))<1e-12
  checks[f'T{T}_observation_origin']=np.max(np.abs(pos-target-err))<2e-6
  checks[f'T{T}_terminal_stop']=not any(x['terminated'] or x['truncated'] for x in rows[:-1])
  with (folder/'trajectory.csv').open() as f:csvrows=list(csv.DictReader(f))
  checks[f'T{T}_csv_json']=len(csvrows)==len(rows) and all(float(cr[k])==float(row[k]) for cr,row in zip(csvrows,rows) for k in row if not isinstance(row[k],bool))
  tbdir=r/'tensorboard'/f'phase11_best_T{T}';writer=None if any(tbdir.glob('events.out.tfevents.*')) else SummaryWriter(str(tbdir))
  fields=['position_error_m','radial_error_m','phase_lag_deg','tilt_deg','roll_deg','pitch_deg','wx','wy','wz','action_clipping_fraction','allocation_clipping_fraction','reward','common_reward_stop','common_reward_return']+[f'{prefix}_{i}' for prefix in ['raw_action','action','actual','thrust'] for i in range(4)]
  fields += [k for k in rows[0] if k.startswith('reward_')]
  for row in rows:
   for k in fields:
    value=float(np.float32(row[k]));tag=('Reward/' if 'reward' in k else 'Evaluation/')+k;
    if writer is not None:writer.add_scalar(tag,value,row['step'])
    scalars.append(dict(period_s=T,tag=tag,step=row['step'],simulation_time_s=row['time_s'],value=value))
  if writer is not None:writer.flush();writer.close()
  ev=EventAccumulator(str(r/'tensorboard'/f'phase11_best_T{T}'),size_guidance={'scalars':0});ev.Reload();lookup={k:{x.step:x.value for x in ev.Scalars(k)} for k in ev.Tags()['scalars']};checks[f'T{T}_tb_exact']=all(lookup[x['tag']][x['step']]==x['value'] for x in scalars if x['period_s']==T)
 n=min(len(seqs[(name,T)]) for name in names)
 for name in names:
  m=segment_metrics(seqs[(name,T)][:n]);common.append(dict(controller=name,period_s=T,common_duration_s=n*.01,position_rmse_m=m['position_rmse_m'],max_tilt_deg=m['max_tilt_deg']))
 # PID vs new model on precisely the same time interval, separate from all-five shortest interval.
 n2=min(len(seqs[('PID',T)]),len(seqs[('phase11_best',T)]))
 for name in ['PID','phase11_best']:
  m=segment_metrics(seqs[(name,T)][:n2]);common.append(dict(controller=name,period_s=T,comparison='PID vs new',common_duration_s=n2*.01,position_rmse_m=m['position_rmse_m'],max_tilt_deg=m['max_tilt_deg']))
first_fields=[f'{prefix}_{i}' for prefix in ['raw_action','action','actual','omega','thrust'] for i in range(4)]+['vx','vy','vz','qw','qx','qy','qz','wx','wy','wz']
checks['same_first_physical_step_both_periods']=all(seqs[('phase11_best',10)][0][k]==seqs[('phase11_best',5)][0][k] for k in first_fields)
import torch
ref=torch.load(r/'models/phase11_best/best_reference.pt',map_location='cpu',weights_only=True)
ref0=ref['deterministic_actions'][0].numpy();initial_action=np.array([seqs[('phase11_best',10)][0][f'raw_action_{i}'] for i in range(4)])
first_action_error=float(np.max(np.abs(initial_action-ref0)));checks['first_action_matches_fixed_nominal']=first_action_error<1e-6
dump_json(r/'first_step_verification.json',dict(initial_action=initial_action.tolist(),saved_nominal_action=ref0.tolist(),max_error=first_action_error,same_physical_first_step_both_periods=checks['same_first_physical_step_both_periods']))
checks['same_initial_state']=results[('phase11_best',10)]['initial']==results[('phase11_best',5)]['initial']==results[('PID',10)]['initial']
checks['physics_preserved']=protected_hashes()==read(r/'evidence/protected_physics_sha256.json')
for label,path in [('existing_sources','evidence/existing_sources_sha256.json'),('historical_results','evidence/historical_files_sha256.json')]:checks[label+'_preserved']=all(sha256(ROOT/k)==v for k,v in read(r/path).items())
checks['checkpoint_preserved']=all(sha256(x['source'])==x['sha256']==sha256(x['copy']) for x in plan['models'].values())
checks={k:bool(v) for k,v in checks.items()}
assert all(checks.values()),checks
dump_json(r/'verification.json',dict(success=True,checks=checks,optimizer_updates=0,tensorboard_scalar_values=len(scalars),case_integrity={str(k):v['checks'] for k,v in results.items() if k[0]=='phase11_best'}))
csvwrite(r/'summary.csv',summary);dump_json(r/'summary.json',summary);csvwrite(r/'common_window_comparison.csv',common);csvwrite(r/'tensorboard_scalars.csv',scalars);dump_json(r/'tensorboard_scalars.json',scalars)
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':10,'axes.grid':True,'grid.alpha':.25})
def save(fig,name):fig.tight_layout();fig.savefig(r/'plots'/(name+'.png'),dpi=160);plt.close(fig)
def vals(rows,key):return np.array([x[key] for x in rows])
fig,axs=plt.subplots(1,2,figsize=(13,6))
for ax,T in zip(axs,[10,5]):
 ref=reference(np.linspace(0,T,600),T)[0];ax.plot(ref[:,0],ref[:,1],'k--',label='원 목표 R=1m')
 for name in names:
  rows=seqs[(name,T)];ax.plot([1]+vals(rows,'x').tolist(),[0]+vals(rows,'y').tolist(),label=labels[name],color=colors[name]);ax.scatter(rows[-1]['x'],rows[-1]['y'],marker='o' if results[(name,T)]['metrics']['completed_two_reference_laps'] else 'x',color=colors[name])
 ax.set(aspect='equal',xlabel='X (m)',ylabel='Y (m)',title=f'주기 {T}s · 목표 두 바퀴 · × 첫 실패');ax.legend(fontsize=8)
save(fig,'xy_paths_comparison')
for T in [10,5]:
 rows=seqs[('phase11_best',T)]
 csvwrite(r/'plots'/f'comparison_trajectories_T{T}.csv',[dict(controller=name,**x) for name in names for x in seqs[(name,T)]])
 fig,axs=plt.subplots(3,1,figsize=(12,10),sharex=True)
 for ax,k in zip(axs,'xyz'):
  tt=np.arange(2*T*100+1)*.01;ref=reference(tt,T)[0];ax.plot(tt,ref[:,list('xyz').index(k)],'k--',label='목표')
  for name in names:ax.plot(vals(seqs[(name,T)],'time_s'),vals(seqs[(name,T)],k),label=labels[name],color=colors[name])
  ax.set_ylabel(k+' (m)');ax.legend(fontsize=8)
 axs[-1].set_xlabel('simulation time (s)');fig.suptitle(f'원 추종 XYZ · 주기 {T}s · 실패 이후 연결 없음');save(fig,f'position_T{T}')
 fig,axs=plt.subplots(5,1,figsize=(12,13),sharex=True)
 for ax,k,u in zip(axs,['position_error_m','radial_error_m','phase_lag_deg','tilt_deg','allocation_clipping_fraction'],['3D 위치오차 (m)','반경오차 (m)','위상 지연 (deg)','기울기 (deg)','allocation 제한 비율']):
  for name in names:ax.plot(vals(seqs[(name,T)],'time_s'),vals(seqs[(name,T)],k),label=labels[name],color=colors[name])
  ax.set_ylabel(u);ax.legend(fontsize=8)
 axs[-1].set_xlabel('simulation time (s)');fig.suptitle(f'원 추종 오차와 실패 · 주기 {T}s');save(fig,f'errors_T{T}')
 fig,axs=plt.subplots(5,2,figsize=(14,14),sharex=True);tm=vals(rows,'time_s')
 for ax,k in zip(axs[0],['roll_deg','pitch_deg']):ax.plot(tm,vals(rows,k));ax.set_ylabel(k+' (deg)')
 for i in range(3):axs[1,0].plot(tm,vals(rows,['wx','wy','wz'][i]),label='xyz'[i]);axs[1,1].plot(tm,vals(rows,f'actual_{i}'),label='xyz'[i])
 axs[1,0].set_ylabel('body ω (rad/s)');axs[1,1].set_ylabel('실제 토크 (N m)')
 for i in range(4):
  axs[2,0].plot(tm,vals(rows,f'raw_action_{i}'),label=str(i));axs[2,1].plot(tm,vals(rows,f'action_{i}'),label=str(i));axs[3,0].plot(tm,vals(rows,f'thrust_{i}'),label=str(i))
 axs[2,0].set_ylabel('raw actor mean');axs[2,1].set_ylabel('applied action');axs[3,0].set_ylabel('모터 추력 (N)');axs[3,1].plot(tm,vals(rows,'request_3'),label='요청');axs[3,1].plot(tm,vals(rows,'actual_3'),label='실제');axs[3,1].set_ylabel('총 추력 (N)')
 for i,k in enumerate('xyz'):
  axs[4,0].plot(tm,vals(rows,'v'+k),label='실제 '+k);axs[4,0].plot(tm,vals(rows,'target_v'+k),'--',label='원 궤적 미분 '+k)
  axs[4,1].plot(tm,vals(rows,'verr_'+k),label='복귀 기준 오차 '+k)
 axs[4,0].set_ylabel('속도 (m/s), 목표미분은 진단용');axs[4,1].set_ylabel('v - clip_norm(-4 e) (m/s)')
 for ax in axs.flat:
  if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=7,ncol=2)
 for ax in axs[-1]:ax.set_xlabel('simulation time (s)')
 fig.suptitle(f'phase11 축별 보상 best · 주기 {T}s · 실제 첫 종료까지');save(fig,f'phase11_best_waveforms_T{T}')
 # Early response, preserve the full requested horizon in other plots.
 fig,axs=plt.subplots(3,1,figsize=(12,9),sharex=True);end=min(3.,2*T)
 for ax,k,u in zip(axs,['position_error_m','tilt_deg','action_clipping_fraction'],['3D 위치오차 (m)','기울기 (deg)','action 제한 비율']):
  for name in names:
   seq=[x for x in seqs[(name,T)] if x['time_s']<=end];ax.plot(vals(seq,'time_s'),vals(seq,k),label=labels[name],color=colors[name])
  ax.set_ylabel(u);ax.set_xlim(0,end);ax.legend(fontsize=8)
 axs[1].axhline(60,ls=':',color='black');axs[-1].set_xlabel('simulation time (s)');fig.suptitle(f'초기 응답 확대 0–{end:g}s · 주기 {T}s');save(fig,f'initial_response_T{T}')
for f in ['summary.csv','common_window_comparison.csv']:shutil.copy2(r/f,r/'plots'/f)
lines=['Phase11 축별 보상 best 원형 경로 평가','실행 경로: '+str(r),'모델: '+plan['models']['phase11_best']['source'],'SHA256: '+plan['models']['phase11_best']['sha256'],'선택: phase11 fresh 512회 checkpoint_512.pt와 동일한 best.pt (이번 새 학습 없음, final과 동일 SHA256).','',
'이전 실험과 동일: 반지름1m, 중심[0,0,1], +Z에서본 반시계방향, yaw0, 주기10s/5s 각두바퀴(20s/10s). 초기[1,0,1], 수평/정지/기존hover RPM. 램프업/settling/목표속도 초기화 없음. 500Hz PhysX,100Hz 정책,기존clip/allocation/모터지연/실패조건 유지. 첫 실패에서 종료한다.',
'정책 입력은15D 위치오차/절대world속도/wxyz/body각속도/yaw sincos. 움직이는 목표의 위치만 갱신하며 원의 접선속도나 가속도 feedforward를 넣지 않았다. v_des=-4e의norm1.5m/s제한은 phase11 학습 reward의 복귀속도 기준이며 이번에도 정책 입력이나 별도 힘 명령이 아니다.',
'원궤적미분 p_dot와 reward용 복귀속도 v_des는 다르다. trajectory의 vex/vey/vez는 v-p_dot, verr_x/y/z는 v-v_des이다. 원궤적 목표속력은 T10에서0.6283m/s,T5에서1.2566m/s이다. reward는 모니터링에만 계산했고 optimizer update=0이다.',
'PID/Q_base/E_position은 circle_pid_qe_20261013_160146, phase9_best는 circle_phase9_track_return_best_20261014_142835 데이터를 재사용했다. 원래 파일 SHA256 및 물리 소스 동일성을 확인했다. 이 네 모델을 이번에 재실행했다고 보고하지 않는다.','',
'모델 | 주기 s | 완료 | 실제시간 s | 종료 | 위치RMSE m | 반경RMSE m | 최대기울기 deg | action clip | allocation clip']
for x in summary:lines.append(f"{x['controller']} ({x['source']}) | {x['period_s']} | {x['completed']} | {x['duration_s']:.2f} | {x['reason']} | {x['position_rmse_m']:.6f} | {x['radial_rmse_m']:.6f} | {x['max_tilt_deg']:.3f} | {x['action_clipping_fraction']:.5f} | {x['allocation_clipping_fraction']:.5f}")
lines+=['','이번 phase11은 두 주기 모두 물리 실패 없이 목표 두 바퀴의 시간까지 비행했다. 이는 목표와 일치하는 정확한 두 바퀴 완주를 의미하지 않는다.']
for T in [10,5]:
 m=results[('phase11_best',T)]['metrics']
 lines.append(f"주기 {T}s: 3D 위치 RMSE {m['position_rmse_m']:.6f}m, 반경 RMSE {m['radial_error_rmse_m']:.6f}m, 접선 방향 오차 RMSE {m['tangential_error_rmse_m']:.6f}m, 평균 위상 지연 {m['phase_lag_mean_deg']:.3f}deg. 원형 경로 근처를 비행하지만 움직이는 목표를 뒤따르는 지연이 위치 오차의 큰 부분을 차지한다.")
lines += ['이 해석은 기록된 반경/접선 오차 분해에 근거한다. 목표 속도 입력 또는 feedforward 부재가 원인이라고 단정한 개입 실험은 아니며, 이번에는 기존 평가 방법을 유지했다.','실제 모니터 reward는 phase11 13개 축별 항의 합이다. common_reward_stop/return은 이전 기준의 진단값이며 실제 reward에 더하지 않는다. 평가에서는 어떠한 reward도 optimizer에 사용하지 않았다.']
lines+=['','RMSE는오차의제곱평균제곱근이다. 조기실패의짧은궤적RMSE를 완주PID의전체RMSE보다좋다고해석하지 않는다. 같은시간비교는 common_window_comparison.csv에별도제공한다. measured_azimuth_revolutions는중심기준방위각의누적변화이지정확한원경로완주증거가아니다.','고정목표±5cm 복귀학습에서얻은모델의새로운이동목표일반화평가이다. 원추종실패가정적호버평가를무효로하지않으며, OOD 질량/고장강인성은시험하지않았다. 단일정책의속도별1회결과이며튜닝/추가학습없음.','',
'검증: 새프로세스2회128개고정관측재로드,동일초기상태,관측오차=localposition-movingtarget,매policy5physics/motorstep,엔진경과시간,action-to-wrench,보상합,finite,terminal이전snapshot,모델/부모/기존코드불변,optimizer0,TB/CSV/JSON일치. GUI는실행하지않았다.','TensorBoard축=evaluation policy step,100Hz. 기존6006/6007서비스는변경하지않았다. 플롯/대응CSV는plots,원시궤적은cases,TensorBoard는tensorboard.']
(r/'REPORT_circle_phase11_ko.txt').write_text('\n'.join(lines)+'\n')
commands=f'''cd {ROOT}\nPY=/home/mrl_6534/miniconda3/envs/isaaclab/bin/python\nRUN={r}\n# Headless replay, period 10 or 5; no training\n"$PY" scripts/evaluate_circle_phase11.py --run-dir "$RUN" --controller phase11_best --period 10 --output "$RUN/manual_T10_$(date +%Y%m%d_%H%M%S)" --device cuda:0 --headless --physics isaacsim_physx\n# GUI replay: replace --headless with --visualizer kit. GUI not verified in this run.\n# TensorBoard on a free port, preserving6006/6007\n"$PY" -m tensorboard.main --logdir "$RUN/tensorboard" --host 127.0.0.1 --port 6008\n# Windows PowerShell\nssh -N -L 6008:127.0.0.1:6008 mrl_6534@100.79.103.47\n# http://127.0.0.1:6008\nscp mrl_6534@100.79.103.47:{r}/review.tar.gz .\n'''
(r/'COMMANDS.txt').write_text(commands)
for root in ['scripts','src']:
 if (ROOT/root).exists():shutil.copytree(ROOT/root,r/'source_snapshot'/root,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
(r/'source_snapshot/README.txt').write_text('Snapshot of scripts and project source for this evaluation. No training. Original phase11 model copied with SHA256 verification.\n')
files=sorted(p for p in r.rglob('*') if p.is_file() and p.name not in ['review.tar.gz','MANIFEST.sha256','review.tar.gz.sha256','archive_verification.json','analysis.log','FILES.txt'] and '__pycache__' not in p.parts)
(r/'FILES.txt').write_text('\n'.join([str(p.relative_to(r)) for p in files]+['FILES.txt','MANIFEST.sha256'])+'\n');files.append(r/'FILES.txt')
manifest={str(p.relative_to(r)):sha256(p) for p in files};(r/'MANIFEST.sha256').write_text(''.join(h+'  '+f+'\n' for f,h in manifest.items()))
with tarfile.open(r/'review.tar.gz','w:gz') as t:
 for p in files+[r/'MANIFEST.sha256']:t.add(p,arcname=str(p.relative_to(r)),recursive=False)
with tarfile.open(r/'review.tar.gz','r:gz') as t:
 for f,h in manifest.items():assert hashlib.sha256(t.extractfile(f).read()).hexdigest()==h,f
archive_hash=sha256(r/'review.tar.gz');(r/'review.tar.gz.sha256').write_text(archive_hash+'  review.tar.gz\n');dump_json(r/'archive_verification.json',dict(success=True,files=len(manifest),sha256=archive_hash,bytes=(r/'review.tar.gz').stat().st_size))
print('DONE',json.dumps([x for x in summary if x['controller']=='phase11_best']),flush=True)
