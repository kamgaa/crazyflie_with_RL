"""Read-only phase10 evaluation plots for the user's four review priorities."""
import argparse,csv,json,hashlib,math,shutil,tarfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir).resolve();out=r/'plots/priority_review';out.mkdir(exist_ok=False)
font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');font_manager.fontManager.addfont(str(font));plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(font)).get_name(),'axes.unicode_minus':False,'font.size':9})
inputs={};labels={'parent':'부모 복귀 속도 / 추가 0회','control_stop':'고정 비교 정지 속도 / phase9 final','final':'최종 복귀 속도 / 추가 512회 (=best)'};colors={'parent':'#525252','control_stop':'#2471a3','final':'#dd7519'}
def remember(p):inputs[str(p.relative_to(r))]=hashlib.sha256(p.read_bytes()).hexdigest()
def read_csv(p):
 remember(p)
 with p.open() as f:return list(csv.DictReader(f))
def load_json(p):remember(p);return json.loads(p.read_text())
def write_csv(name,rows):
 fields=list(dict.fromkeys(k for row in rows for k in row))
 with (out/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def save(fig,name):fig.tight_layout(rect=(0,0,1,.96));fig.savefig(out/name,dpi=150,bbox_inches='tight');plt.close(fig)
# 1: Each point is a saved checkpoint; no interpolation presented as a measurement.
rows=read_csv(r/'plots/checkpoint_comparison.csv');x=np.array([int(q['additional_iteration']) for q in rows]);fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True)
for ax,key,mul,title,log in [(axes[0],'completed_only_mean_position_rmse_m',100,'전체 0~30초 위치 RMSE [cm]',False),(axes[1],'completed_only_mean_tail_body_omega_rms_rad_s',1,'후반 20~30초 body 각속도 RMS [rad/s, log scale]',True),(axes[2],'completed_only_mean_tail_applied_action_saturation_fraction',100,'후반 적용 action 포화 [%; 4축 전체]',False)]:
 y=np.array([float(q[key]) for q in rows])*mul;ax.plot(x,y,'o-',color=colors['final'])
 for xx,yy in zip(x,y):ax.annotate(f'{yy:.4g}',(xx,yy),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
 if log:ax.set_yscale('log')
 ax.set_ylabel(title);ax.grid(alpha=.25)
axes[-1].set_xticks(x);axes[-1].set_xlabel('이번 추가 iterations (부모 누적 1,664회; 64회 간격 평가)');fig.suptitle('우선순위 1 | 복귀 속도 모델 checkpoint 변화 | 고정 7조건 모두 30초 완료, 조건별 지표 평균');save(fig,'01_checkpoint_rmse_omega_saturation.png');write_csv('01_checkpoint_metrics.csv',rows)
# 2: Plot actual Euler angles, body rates, normalized actions, actual torque separately.
nominal={m:read_csv(r/'evaluation'/f'{m}_fixed_30s/final/nominal.csv') for m in ['parent','final']}
nominal={m:[q for q in data if 29<=float(q['time_s'])<=30] for m,data in nominal.items()}
fig,axes=plt.subplots(6,2,figsize=(15,18),sharex=True);axes=axes.ravel()
fields=[('roll_deg','Euler roll [deg]'),('pitch_deg','Euler pitch [deg]'),('wx','body omega X [rad/s]'),('wy','body omega Y [rad/s]'),('wz','body omega Z [rad/s]')]
for ax,(key,title) in zip(axes[:5],fields):
 for m,data in nominal.items():ax.plot([float(q['time_s']) for q in data],[float(q[key]) for q in data],color=colors[m],label=labels[m],lw=1.)
 ax.set_title(title)
for i,ax in enumerate(axes[5:9]):
 for m,data in nominal.items():
  t=[float(q['time_s']) for q in data]
  ax.plot(t,[float(q[f'raw_action_{i}']) for q in data],color=colors[m],label=labels[m]+' raw',lw=1.)
  ax.plot(t,[float(q[f'action_{i}']) for q in data],color=colors[m],label=labels[m]+' applied',ls='--',lw=1.)
 ax.axhline(1,color='red',ls=':',lw=.7);ax.axhline(-1,color='red',ls=':',lw=.7);ax.set_title(['roll','pitch','yaw','collective'][i]+' action [dimensionless; solid raw / dashed clipped]')
for i,ax in enumerate(axes[9:12]):
 for m,data in nominal.items():ax.plot([float(q['time_s']) for q in data],[float(q[f'actual_{i}']) for q in data],color=colors[m],label=labels[m],lw=1.)
 ax.set_title('実際' if False else ['실제 roll torque [N m]','실제 pitch torque [N m]','실제 yaw torque [N m]'][i]);ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,2))
for ax in axes:ax.set_xlim(29,30);ax.grid(alpha=.25);ax.set_xlabel('simulation time [s]')
axes[0].legend(fontsize=8);axes[5].legend(fontsize=7);fig.suptitle('우선순위 2 | Nominal 29~30초 | 부모 vs final(=best) | 자세·행동·실제 토크 구분');save(fig,'02_parent_final_waveforms_29_30s.png')
write_csv('02_parent_final_waveforms_29_30s.csv',[dict(model=m,**q) for m,data in nominal.items() for q in data])
# 3: Six +/-5cm cases; retain XYZ cross-axis response, no condition averaging.
cases=['x_plus','x_minus','y_plus','y_minus','z_plus','z_minus'];ctitles=['X +5cm','X -5cm','Y +5cm','Y -5cm','Z +5cm','Z -5cm'];early={};early_rows=[]
for m in ['parent','final']:
 for case in cases:
  folder=r/'evaluation'/f'{m}_fixed_30s/final';data=[q for q in read_csv(folder/(case+'.csv')) if float(q['time_s'])<=5]
  initial=load_json(folder/(case+'.json'))['initial_observation'];w,qx,qy,qz=initial[6:10]
  first=dict(time_s=0.,ex=initial[0],ey=initial[1],ez=initial[2],vx=initial[3],vy=initial[4],vz=initial[5],roll_deg=math.degrees(math.atan2(2*(w*qx+qy*qz),1-2*(qx*qx+qy*qy))),pitch_deg=math.degrees(math.asin(np.clip(2*(w*qy-qz*qx),-1,1))),tilt_deg=math.degrees(math.acos(np.clip(1-2*(qx*qx+qy*qy),-1,1))))
  keys=list(first);series=[dict(first,source='saved initial_observation JSON')]+[dict({k:float(q[k]) for k in keys},source='saved trajectory CSV') for q in data];early[m,case]=series
  early_rows.extend(dict(model=m,condition=case,**q) for q in series)
for name,keys,unit,title in [('03a_return_position_0_5s.png',['ex','ey','ez'],'cm','목표 대비 XYZ 위치 오차'),('03b_return_velocity_0_5s.png',['vx','vy','vz'],'m/s','실제 world XYZ 속도'),('03c_return_attitude_0_5s.png',['roll_deg','pitch_deg','tilt_deg'],'deg','Euler roll/pitch 및 body Z 기울기')]:
 fig,axes=plt.subplots(3,6,figsize=(22,9),sharex=True,sharey='row');scale=100 if unit=='cm' else 1
 for j,case in enumerate(cases):
  for i,key in enumerate(keys):
   ax=axes[i,j]
   for m in ['parent','final']:
    data=early[m,case];ax.plot([q['time_s'] for q in data],[q[key]*scale for q in data],color=colors[m],label=labels[m],lw=.85)
   ax.axhline(0,color='black',ls=':',lw=.5);ax.set_xlim(0,5);ax.grid(alpha=.25)
   if i==0:ax.set_title(ctitles[j])
   if j==0:ax.set_ylabel(key+' ['+unit+']')
   if i==2:ax.set_xlabel('time [s]')
 axes[0,0].legend(fontsize=7);fig.suptitle('우선순위 3 | ±5cm 복귀 첫 0~5초 | '+title+' | 부모 vs final(=best)');save(fig,name)
write_csv('03_return_0_5s.csv',early_rows)
# 4: Show each of all 27 pre-existing initial conditions; no cancellation through averaging.
per_case=[];cases_all=['nominal',*cases]+[f'random_{i:02d}' for i in range(20)]
for m in ['parent','control_stop','final']:
 for suite in ['fixed','random20']:
  for d in load_json(r/'evaluation'/f'{m}_{suite}_30s/results.json'):
   q=dict(model=m,suite=suite,condition=d['condition'],completed=d['completed'],duration_s=d['duration_s'],tail_position_rmse_m=d['tail_position_rmse_m'],tail_position_error_mean_m=d['tail_position_mean_m'])
   for i,axis in enumerate('xyz'):
    q['bias_'+axis+'_mm']=None if d['tail_error_vector_m'] is None else d['tail_error_vector_m'][i]*1000
    q['std_'+axis+'_mm']=None if d['tail_position_error_m_std'] is None else d['tail_position_error_m_std'][i]*1000
   per_case.append(q)
fig,axes=plt.subplots(4,1,figsize=(18,12),sharex=True);xx=np.arange(27);ticklabels=['N','X+','X-','Y+','Y-','Z+','Z-']+[f'R{i:02d}' for i in range(20)]
for ax,key,title in zip(axes,['bias_x_mm','bias_y_mm','bias_z_mm','tail_position_rmse_m'],['후반 X 편향 [mm]','후반 Y 편향 [mm]','후반 Z 편향 [mm]','후반 3D 위치 RMSE [mm]']):
 for j,m in enumerate(['parent','control_stop','final']):
  lookup={q['condition']:q for q in per_case if q['model']==m};yy=[lookup[c][key] for c in cases_all];yy=[np.nan if v is None else v*(1000 if key=='tail_position_rmse_m' else 1) for v in yy]
  ax.scatter(xx+(j-1)*.18,yy,color=colors[m],label=labels[m],s=24)
 ax.axhline(0,color='black',ls=':',lw=.7);ax.axvline(6.5,color='gray',ls='--',lw=.8);ax.grid(axis='y',alpha=.25);ax.set_ylabel(title)
axes[0].legend(ncol=3);axes[-1].set_xticks(xx,ticklabels,rotation=45);axes[-1].set_xlabel('개별 초기조건: 고정7 | 기존 random20 (평균하지 않은 각 episode)');fig.suptitle('우선순위 4 | 조건별 20~30초 편향과 위치 오차 | 모두 30초 완료; 실패 후반은 N/A 규약');save(fig,'04_conditionwise_tail_xyz_bias_error.png');write_csv('04_conditionwise_tail_xyz_bias_error.csv',per_case)
readme='''phase10 우선순위 검토 그래프 (기존 저장 데이터만 사용; 학습/평가 추가0)
01: checkpoint별 고정7조건 평균. 64회 간격의 측정이며 체크포인트 사이 변화 시점은 확정할 수 없음.
02: Nominal 부모/final 29~30초. raw/applied action은 무차원, 실제 Euler는deg, body각속도rad/s, 실제COM토크Nm.
03a/b/c: 여섯 +/-5cm 조건을 평균 없이 비교. t=0은 저장된 initial_observation JSON, 나머지는 .01초 간격 원시 CSV. 공통 축 범위.
04: 고정7+기존20 각 조건의 후반10초 XYZ 편향 및3D RMSE. 반대 부호 상쇄를 확인하도록 조건별 점으로 표시. CSV에 centered std도 포함.
추가0은 phase9 track_return/final, 추가512는 phase10 final=best. 고정 비교 control_stop은 phase9 final이며 추가 학습하지 않음.
기존 review.tar.gz는 변경하지 않았고 이 추가 그래프는 별도 priority_review.tar.gz에 포함됨.
'''
(out/'README_ko.txt').write_text(readme);(out/'INPUT_SHA256.json').write_text(json.dumps(inputs,indent=2));shutil.copy2(__file__,out/Path(__file__).name)
assert all(hashlib.sha256((r/f).read_bytes()).hexdigest()==h for f,h in inputs.items())
checks=dict(input_files_unchanged=True,optimizer_updates=0,evaluation_steps=0,checkpoint_samples=len(rows),early_cases=len(early),conditionwise_rows=len(per_case),all_tail_conditions_complete=all(q['completed'] for q in per_case))
(out/'verification.json').write_text(json.dumps(checks,indent=2))
files=sorted(p for p in out.iterdir() if p.is_file());hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files};(out/'MANIFEST.sha256').write_text(''.join(f'{h}  {n}\n' for n,h in hashes.items()))
archive=r/'plots/priority_review.tar.gz'
with tarfile.open(archive,'w:gz') as tar:tar.add(out,arcname='priority_review')
with tarfile.open(archive,'r:gz') as tar:
 for n,h in hashes.items():assert hashlib.sha256(tar.extractfile('priority_review/'+n).read()).hexdigest()==h
archive.with_name(archive.name+'.sha256').write_text(hashlib.sha256(archive.read_bytes()).hexdigest()+'  '+archive.name+'\n')
print('DONE',out,archive.stat().st_size)
