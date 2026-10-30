"""Phase7 diagnosis using existing phase6 artifacts; zero optimizer updates."""
import argparse,json,csv,math,sys,time,inspect
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensordict import TensorDict
from rsl_rl.storage import RolloutStorage
from ppo_common import ROOT,dump_json,sha256
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args()
r=Path(a.run_dir);out=r/'diagnosis';plan=json.loads((r/'experiment_plan.json').read_text());parent=Path(plan['parent'])
def read(p):return json.loads(Path(p).read_text())
def csvwrite(path,rs):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,list(dict.fromkeys(k for x in rs for k in x)));w.writeheader();w.writerows(rs)
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
metrics=lines(parent/'metrics.jsonl');assert len(metrics)==768
terms=['xy','z','velocity','tilt','angular_velocity','yaw','termination']
coeff=[10,6,.005,6,.001,1,10]
keys=[f'RewardStep/{k}' for k in terms+['total']]+[f'RewardEpisode/{k}' for k in terms+['total']]
keys+=['Policy/std_axis_'+str(i) for i in range(4)]+['Policy/entropy','PPO/approx_kl','PPO/kl','PPO/clip_fraction','PPO/learning_rate','Loss/policy','Loss/value','Loss/entropy_contribution','Loss/ppo_total',
    'Gradient/actor_preclip_norm','Gradient/critic_preclip_norm','Gradient/actor_clipped_fraction','Gradient/critic_clipped_fraction',
    'Actuator/action_clipping_fraction','Actuator/allocation_clipping_fraction','Episode/length_s','Episode/return']
windows={}
for label,rows in [('first64',metrics[:64]),('last64',metrics[-64:])]:
    vals={}
    for key in keys:
        subset=[x for x in rows if key in x]
        if key.startswith('RewardEpisode/') or key in ['Episode/return','Episode/length_s']:
            denom=sum(x['Episode/completed_count'] for x in subset)
            vals[key]=sum(x[key]*x['Episode/completed_count'] for x in subset)/denom if denom else None
        else:vals[key]=float(np.mean([x[key] for x in subset])) if subset else None
    for k,c in zip(terms,coeff):
        vals['RewardRaw/'+k]=-vals['RewardStep/'+k]/c
    vals['completed_episode_count']=sum(x['Episode/completed_count'] for x in rows)
    windows[label]=vals
csvwrite(out/'training_windows.csv',[dict(metric=k,first64=windows['first64'][k],last64=windows['last64'][k],last_minus_first=windows['last64'][k]-windows['first64'][k] if windows['first64'][k] is not None else None) for k in windows['first64']])
cp=torch.load(parent/'final.pt',map_location='cpu',weights_only=False)
assert cp['iter']==1023 and cp['infos']['collected_transitions']==33554432
assert {int(x['step']) for x in cp['optimizer_state_dict']['state'].values()}=={20480}
assert sha256(parent/'final.pt')==plan['parent_sha256']
cpstd=[]
for path in [parent.parent/'phase5_parallel_20261010_145354/N1024/final.pt',parent/'checkpoint_10485760.pt',parent/'checkpoint_31457280.pt',parent/'final.pt']:
    c=torch.load(path,map_location='cpu',weights_only=False);log=c['actor_state_dict']['distribution.log_std_param']
    cpstd.append(dict(path=str(path),iteration=c['infos']['completed_iterations'],log_std=log.tolist(),std=log.exp().tolist()))
evaldir=parent/'evaluation/final_reload/final';results=read(parent/'evaluation/final_reload/results.json')
trajectory=[];allrows=[]
fig,axs=plt.subplots(5,4,figsize=(16,15))
for j,x in enumerate(results):
    with (evaldir/(x['condition']+'.csv')).open() as f:rows=list(csv.DictReader(f))
    arr=lambda names:np.array([[float(v[k]) for k in names] for v in rows])
    t=arr(['time_s'])[:,0];e=arr(['ex','ey','ez']);vel=arr(['vx','vy','vz']);pos=np.linalg.norm(e,axis=1)
    tail=e[-200:];ts=t[-200:];vv=vel[-200:];slope=np.array([np.polyfit(ts,tail[:,i],1)[0] for i in range(3)])
    clipped=arr(['action_clipping_fraction','allocation_clipping_fraction']);raw=arr(['raw_action_'+str(i) for i in range(4)])
    correlation=[]
    for i in range(2):
        correlation.append(float(np.corrcoef(clipped[:,i],pos)[0,1]) if np.std(clipped[:,i])>0 else None)
    rec=dict(condition=x['condition'],duration_s=x['duration_s'],completed_8s=x['completed_8s'],
        tail2_error_vector_m=tail.mean(0).tolist(),tail2_mean_error_m=float(np.linalg.norm(tail,axis=1).mean()),
        tail2_rmse_m=float(np.sqrt(np.mean(np.sum(tail**2,axis=1)))),tail2_mean_speed_m_s=float(np.linalg.norm(vv,axis=1).mean()),
        tail2_mean_velocity_m_s=vv.mean(0).tolist(),tail2_error_slope_m_s=slope.tolist(),tail2_error_std_m=tail.std(0).tolist(),
        first1s_error_vector_m=e[:100].mean(0).tolist(),last_error_vector_m=e[-1].tolist(),
        max_tilt_deg=x['max_tilt_deg'],action_clip_axis_fraction=(np.abs(raw)>1).mean(0).tolist(),
        action_first_clipping_s=float(t[np.where(clipped[:,0]>0)[0][0]]) if np.any(clipped[:,0]>0) else None,
        allocation_first_clipping_s=float(t[np.where(clipped[:,1]>0)[0][0]]) if np.any(clipped[:,1]>0) else None,
        clip_position_norm_correlation=correlation,
        mean_error_when_action_clipped_m=float(pos[clipped[:,0]>0].mean()) if np.any(clipped[:,0]>0) else None,
        mean_error_without_action_clipping_m=float(pos[clipped[:,0]==0].mean()) if np.any(clipped[:,0]==0) else None)
    rec['time_bins']=[dict(start_s=k,end_s=k+1,mean_error_m=float(pos[(t>k)&(t<=k+1)].mean()),
        action_clip_fraction=float(clipped[(t>k)&(t<=k+1),0].mean()),allocation_clip_fraction=float(clipped[(t>k)&(t<=k+1),1].mean())) for k in range(8)]
    trajectory.append(rec)
    for v in rows:allrows.append(dict(condition=x['condition'],**v))
    for ax,data,names,unit in zip(axs[j],[e,vel,arr(['tilt_deg']),clipped],[['ex','ey','ez'],['vx','vy','vz'],['tilt'],['action','allocation']],['Error m','Velocity m/s','Tilt deg','Clipped component fraction']):
        for k,name in enumerate(names):ax.plot(t,data[:,k],label=name,lw=.7)
        ax.axvspan(6,8,color='gray',alpha=.1);ax.set(xlabel='Time s',ylabel=unit,title=x['condition']);ax.grid(alpha=.2);ax.legend(fontsize=7)
fig.suptitle('Phase6 final: existing recorded trajectories, last 2s shaded')
fig.tight_layout(rect=(0,0,1,.965));fig.savefig(out/'parent_trajectories.png',dpi=140);plt.close(fig)
csvwrite(out/'parent_trajectories.csv',allrows)
csvwrite(out/'tail2_diagnosis.csv',[{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in x.items()} for x in trajectory])
fig,axs=plt.subplots(4,2,figsize=(12,13))
for ax,key in zip(axs.flat,[f'RewardStep/{k}' for k in terms+['total']]):
    ax.plot([x['global_step'] for x in metrics],[x[key] for x in metrics]);ax.set(title=key,xlabel='Cumulative transitions',ylabel='Signed contribution / policy step');ax.grid(alpha=.2)
fig.suptitle('Phase6 reward A: raw iteration means, identical sample scope');fig.tight_layout(rect=(0,0,1,.965));fig.savefig(out/'reward_history.png',dpi=140);plt.close(fig)
csvwrite(out/'reward_history.csv',[{k:x[k] for k in ['global_step','iteration']+[f'RewardStep/{k}' for k in terms+['total']]} for x in metrics])
fig,axs=plt.subplots(3,2,figsize=(12,10))
groups=[['Policy/std_axis_'+str(i) for i in range(4)],['Policy/entropy'],['PPO/approx_kl','PPO/clip_fraction'],['Loss/policy'],['Loss/value','Loss/ppo_total'],['Gradient/actor_preclip_norm','Gradient/critic_preclip_norm']]
for ax,ks in zip(axs.flat,groups):
    for k in ks:ax.plot([x['iteration'] for x in metrics],[x[k] for x in metrics],label=k)
    ax.set_xlabel('Cumulative iterations');ax.grid(alpha=.2);ax.legend(fontsize=7)
fig.suptitle('Phase6 PPO statistics (cached forward std, actual backward loss)');fig.tight_layout(rect=(0,0,1,.965));fig.savefig(out/'ppo_history.png',dpi=140);plt.close(fig)
csvwrite(out/'ppo_history.csv',[{k:x.get(k) for k in ['global_step','iteration']+sum(groups,[])} for x in metrics])
# Exercise actual observation/reward function source independently of any simulator/optimizer.
base=ROOT/'src/crazyflie_isaaclab/tasks/crazyflie';ns={'__name__':'phase7_contract_check'}
exec(compile((base/'frames.py').read_text(),str(base/'frames.py'),'exec'),ns)
source=(base/'observations_rewards.py').read_text().replace('from .frames import xyzw_to_wxyz, inverse_rotate','')
exec(compile(source,str(base/'observations_rewards.py'),'exec'),ns)
checks=[];g=torch.Generator().manual_seed(42)
for n in [1,64,256,1024]:
    origins=torch.zeros(n,3);ids=torch.arange(n);origins[:,0]=(ids%32)*3;origins[:,1]=(ids//32)*3
    error=torch.tensor([.125,-.25,.125]).expand(n,3);target=torch.tensor([0.,0.,1.]);q=torch.tensor([0.,0.,0.,1.]).expand(n,4)
    obs=ns['state_observation'](origins+target+error,torch.ones(n,3)*.1,q,torch.zeros(n,3),origins,target)
    rew=ns['reward'](obs,torch.zeros(n,dtype=torch.bool),'A')
    assert torch.equal(obs,obs[:1].expand_as(obs));assert torch.equal(rew,rew[:1].expand_as(rew))
    varied=torch.randn(n,15,generator=g);varied[:,6]=1;varied[:,14]=1;crash=ids%3==0
    batch=ns['reward'](varied,crash,'A');individual=torch.cat([ns['reward'](varied[i:i+1],crash[i:i+1],'A') for i in range(n)])
    assert torch.equal(batch,individual)
    checks.append(dict(num_envs=n,origin_removed_exact=True,per_environment_reward_equal=True,single_sample_reward=float(rew[0])))
assert len({x['single_sample_reward'] for x in checks})==1
# Actual installed minibatch generator: each index appears once per epoch, 5 times total.
mbchecks=[]
for mb in [4,64]:
    storage=object.__new__(RolloutStorage);storage.training_type='rl';storage.num_envs=1024;storage.num_transitions_per_env=32;storage.device='cpu'
    storage.observations=TensorDict({'policy':torch.arange(32768).reshape(32,1024,1)},batch_size=[32,1024])
    for name in ['actions','values','returns','actions_log_prob','advantages']:setattr(storage,name,torch.zeros(32,1024,1))
    storage.distribution_params=(torch.zeros(32,1024,4),torch.ones(32,1024,4))
    count=torch.zeros(32768,dtype=torch.int64);batches=0
    for b in storage.mini_batch_generator(mb,5):
        idx=b.observations['policy'].flatten();assert len(idx)==32768//mb;count+=torch.bincount(idx,minlength=32768);batches+=1
    assert torch.all(count==5) and batches==mb*5
    mbchecks.append(dict(minibatches=mb,size=32768//mb,actual_yielded_batches=batches,each_sample_uses=5,optimizer_updates=0))
(out/'installed_minibatch_generator.py').write_text(inspect.getsource(RolloutStorage.mini_batch_generator))
maxerr=max(x['Validation/reward_sum_max_abs_error'] for x in metrics)
assert maxerr<1e-4
res=dict(success=True,optimizer_updates=0,existing_data_only=True,parent_sha256=sha256(parent/'final.pt'),
    parent_adam_steps=[20480],windows=windows,checkpoint_distribution=cpstd,trajectory=trajectory,
    coordinate_reward_checks=checks,minibatch_checks=mbchecks,reward_sum_max_abs_error=maxerr,
    report_differences=['No numerical discrepancy in reported phase6 final metrics or checkpoint counters.',
      'Phase6 root report lists aggregate summaries; detailed trajectory provides additional last2 vector/slopes not previously reported.',
      'Iteration log_std is absent; cached std exists and checkpoint log_std is measured only at saved checkpoints.'])
dump_json(out/'diagnosis.json',res)
def table(h,rr):return '\n'.join([' | '.join(h), ' | '.join(['---']*len(h))]+[' | '.join(map(str,x)) for x in rr])
tt=table(['조건','마지막2초 평균 오차벡터[m]','평균크기[m]','RMSE[m]','평균속력[m/s]','축별기울기[m/s]'],[[x['condition'],str(np.round(x['tail2_error_vector_m'],6).tolist()),round(x['tail2_mean_error_m'],6),round(x['tail2_rmse_m'],6),round(x['tail2_mean_speed_m_s'],6),str(np.round(x['tail2_error_slope_m_s'],6).tolist())] for x in trajectory])
wt=table(['지표','추가 첫64회','추가 마지막64회'],[[k,windows['first64'][k],windows['last64'][k]] for k in keys])
report=f"""Phase7 학습 전 진단 — 기존 Phase6 데이터만 사용
서버 MRL-server, 프로젝트 {ROOT}
부모 {parent/'final.pt'} / SHA256 {sha256(parent/'final.pt')}
진단 optimizer update=0. 새 물리 평가 없이 기존 다섯 8초 궤적을 재사용했다.
보고서의 final 5/5완료, RMSE,시간 및 부모1024회/33,554,432전이/Adam20480과 실제 파일 일치.

위치 좌표:
e=world link-origin position - env_origin - target. target=[0,0,1].
e_x<0은 목표보다 -X, e_y>0은 목표보다 +Y, e_z>0은 목표보다 위.
원점 제거 및 reward 환경축 처리를 실제 함수로 N1/64/256/1024에서 검사했다.
모든 환경에 같은 상대상태를 주면 관측/reward가 동일하며, 개별 계산과 배치 계산도 정확히 일치했다.
환경축을 합산해 개별 reward에 섞는 코드는 없다. reward 각 sum(-1)은 상태성분 축이다.

{tt}

다섯 조건 모두 주로 -X,+Y,+Z 방향으로 편향된다. 시작부터 이동하여 끝까지 목표에 수렴하지 않는다.
마지막2초 회귀 slope와 평균 velocity가 0이 아닌 축, 속도/기울기 진동이 함께 나타난다.
8초에 제한된 데이터로 정상상태나 이후 발산 여부를 단정하지 않는다.
시간별 clip 발생시점/축별빈도,오차와 상관계수,1초구간 평균은 diagnosis.json에 저장했다.
Action clipping과 position error는 시간/상태에 함께 의존한다. 상관관계는 인과관계의 증명이 아니다.
원인 후보: 제한된 업데이트/큰 미니배치 구성, reward 내 위치 정밀도와 생존/자세 안정 사이 학습 균형,
커진 행동분포 및 clipped actuator regime. 현재 관찰만으로 어느 하나를 원인으로 확정하지 않는다.

Reward A:
xy=-10||e_xy||²,z=-6ez²,velocity=-.005||v||²,tilt=-6(qx²+qy²),
angular_velocity=-.001||omega_body||²,yaw=-yaw_error²,termination=-10*failed.
signed 기여도이며 dt 배율과 reward 합산 후 clipping 없음.
RewardRaw는 기존 signed contribution / (-계수)로 산출한 수학적 cost이고 과거에 기록된 별도 태그가 아니다.
서로 다른 물리 단위 계수 자체가 아니라 같은 표본의 signed contribution을 비교한다.
Step은1024×32 표본평균, Episode는 해당 window 종료 episode 수로 가중한 episode별 합계 평균.
종료 penalty 합,원래환경reward,timeout bootstrap 보정 reward를 분리한다.
reward 항 합 최대잔차 {maxerr} (float rounding); 검증 통과.

{wt}

정책std/entropy 증가와 tilt·종료 penalty 감소가 같이 나타났지만,XY 기여도는 더 음수가 된 구간이 있다.
모든 항의 정확한 첫/끝 차이는 training_windows.csv. episode return은 길이 변화에 영향받으므로 step평균과 직접 혼동하지 않는다.
Gaussian std는 clipping 전 분포 척도; action의 환경 간 분산과 다른 값이다.
매 iteration log_std 원시로그는 없다. 이를 측정값으로 만들어 채우지 않았다.
checkpoint별 실제 log_std/std는 diagnosis.json의 checkpoint_distribution.
PPO epsilon .2는 고정값, clip_fraction은 |exp(logp_new-logp_old)-1|>.2 비율이다.
환경 action clipping(|raw_action|>1), allocation clipping([0,.289]N 밖)은 별도 지표이다.
Loss/policy와 Loss/value, entropy contribution 및 실제 loss.backward 직전 total을 구분한다.
기존 gradient norm은 원래 clip_grad_norm 반환값; 추가 optimizer update 없이 분석했다.

비교 전제:
좌표/reward/복원 전제를 깨는 문제는 발견하지 못했다.
설치 minibatch generator는32768표본을 한 번 shuffle하고 같은 partition을5epochs 순회한다.
4→64 변경은 minibatch8192→512 및 update20→320/iteration을 함께 바꾼다.
CPU에서 실제 generator를 실행해 모든 표본이 총5번 사용됨을 확인했다. optimizer step0.
부모에 simulator/진행 episode/rollout은 저장되지 않았다. 세 분기 모두 새 reset/빈 rollout이며 RNG는 저장값 복원.
R2는 같은 critic/Adam으로 시작하므로 reward target 변경에 따른 value loss 과도응답 가능성이 있다.
기존 RSL timeout bootstrap의 current-state V 사용은 보존된 제한사항이며 이 실험에서 수정하지 않는다.
진단자료와 source hash는 학습 전 저장; 이후 결과에 맞추어 진단을 바꾸지 않는다.
"""
(out/'diagnosis_before_training_ko.txt').write_text(report);(r/'diagnosis_before_training_ko.txt').write_text(report)
dump_json(r/'diagnosis_ready.json',dict(time=time.time(),success=True,optimizer_updates=0,report_sha256=sha256(r/'diagnosis_before_training_ko.txt'),json_sha256=sha256(out/'diagnosis.json')))
print('DIAGNOSIS_READY',json.dumps(trajectory,ensure_ascii=False),flush=True)

