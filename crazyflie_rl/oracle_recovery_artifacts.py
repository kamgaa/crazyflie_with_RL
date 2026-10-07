"""CSV-only cross-checks and plots for the fixed-controller recovery experiment."""
import numpy as np

from .integral_eval import read_columns
from .oracle_recovery import COMBINATIONS, tilt_deg


def vector(columns, name, n=3):
    return np.column_stack([columns[f'{name}_{i}'] for i in range(n)])


def verify_csv(directory, result):
    """Recompute from saved numeric signals, not the in-memory summary helpers."""
    f=read_columns(directory/(result['key']+'.csv'))
    t=f.time_post; p=vector(f,'position'); target=vector(f,'reference_post'); e=p-target
    velocity=vector(f,'velocity'); omega=vector(f,'omega'); q=vector(f,'quaternion',4)
    np.testing.assert_allclose(e,vector(f,'position_error_world'),atol=1e-15)
    np.testing.assert_array_equal(vector(f,'policy_raw_velocity'),vector(f,'observation',15)[:,3:6])
    np.testing.assert_allclose(vector(f,'p_cmd'),vector(f,'p_target')+vector(f,'xi_t'),atol=1e-15)
    theta=tilt_deg(q); yaw=np.abs(np.degrees(f.yaw_error_rad))
    post=t>=5-1e-9
    measured={}
    for axis,sl in (('xy',slice(0,2)),('z',slice(2,3)),('3d',slice(0,3)),('joint',slice(0,3))):
        good=(np.linalg.norm(e[:,sl],axis=1)<=.005)&(np.linalg.norm(velocity[:,sl],axis=1)<=.02)
        if axis=='joint':good &= (theta<=5)&(yaw<=5)&(np.linalg.norm(omega,axis=1)<=.10)
        times=t[post]; bad=np.flatnonzero(~good[post]); first=bad[-1]+1 if len(bad) else 0
        latency=None
        if result['completed'] and first<len(times) and 60-times[first]>=1-1e-9:
            latency=float(times[first]-5)
        measured[axis]=latency
        expected=result['recovery_position_attitude_s' if axis=='joint' else 'recovery_'+axis+'_s']
        if result['efficiency_percent']==100:
            assert expected is None and result['recovery_reason']=='not_applicable'
        elif latency is None or expected is None:
            assert latency is expected
        else:np.testing.assert_allclose(latency,expected,atol=1e-12)
    assert result['position_criterion_met']==(measured['3d'] is not None)
    assert result['joint_criterion_met']==(measured['joint'] is not None)
    identity_error=0.
    if result['completed']:
        selected=e[t>58+1e-9];mu=selected.mean(0); variance=np.mean((selected-mu)**2,axis=0)
        square=np.mean(selected**2,axis=0);tail=result['tail58_60']
        np.testing.assert_allclose(mu,np.r_[tail['mean_error_xy_m'],tail['mean_error_z_m']],atol=1e-13)
        np.testing.assert_allclose([np.sqrt(square[:2].sum()),np.sqrt(square[2])],[tail['rmse_xy_m'],tail['rmse_z_m']],atol=1e-13)
        np.testing.assert_allclose([np.sqrt(variance[:2].sum()),np.sqrt(variance[2])],
            [tail['sway_xy_rms_m'],tail['sway_z_rms_m']],atol=1e-13)
        identity_error=float(np.max(np.abs(square-mu**2-variance)))
    else:
        assert result['tail58_60'] is None and all(v is None for v in measured.values())
    xi=vector(f,'xi_t');next_xi=vector(f,'xi_next');candidate=xi-(t-f.time)[:,None]*.2*vector(f,'e_true_before')
    np.testing.assert_allclose(candidate,vector(f,'xi_candidate'),atol=1e-15)
    frozen=f.integral_frozen.astype(bool)
    np.testing.assert_array_equal(next_xi[frozen],xi[frozen])
    np.testing.assert_array_equal(xi[1:],next_xi[:-1])
    expected_freeze=f.interval_allocator_clipping.astype(bool)|f.interval_esc_boundary.astype(bool)|f.interval_action_boundary.astype(bool)
    np.testing.assert_array_equal(frozen,expected_freeze)
    assert np.max(np.linalg.norm(next_xi[:,:2],axis=1))<=.40+1e-12 and np.max(np.abs(next_xi[:,2]))<=.15+1e-12
    dt=t-f.time
    np.testing.assert_allclose(dt[frozen].sum(),result['dwell_full']['integral_frozen']['duration_any_s'],atol=1e-12)
    fphys=read_columns(directory/(result['key']+'-physics.csv'))
    pt=fphys.physics_time;eta=vector(fphys,'plant_efficiency',4)
    scheduled=np.ones_like(eta);motor=1 if 'motor2' in result['condition'] else 2
    scheduled[pt>=5-1e-9,motor]=result['efficiency_percent']/100
    np.testing.assert_array_equal(eta,scheduled)
    np.testing.assert_array_equal(eta,vector(fphys,'allocator_efficiency',4))
    for name in ('thrust','reaction'):
        np.testing.assert_array_equal(vector(fphys,'motor_'+name+'_actual',4),eta*vector(fphys,'motor_'+name+'_nominal',4))
    np.testing.assert_allclose(vector(fphys,'allocation_residual_xml',4)+vector(fphys,'actuator_response_residual_xml',4),
                               vector(fphys,'total_rotor_residual_xml',4),atol=1e-15)
    physdt=fphys.physics_time_post-pt;clipped=vector(fphys,'allocator_clipped',4).astype(bool)
    np.testing.assert_allclose(np.sum(clipped*physdt[:,None],axis=0),
        result['dwell_full']['allocator_clipped']['duration_s_per_channel'],atol=1e-12)
    np.testing.assert_allclose(physdt[np.any(clipped,axis=1)].sum(),
        result['dwell_full']['allocator_clipped']['duration_any_s'],atol=1e-12)
    # Check every control interval uses the union of its actual physical commands.
    per_control=round(np.median(dt)/np.median(physdt))
    assert len(pt)==len(t)*per_control
    interval=clipped.any(axis=1).reshape(-1,per_control).any(axis=1)
    np.testing.assert_array_equal(interval,f.interval_allocator_clipping.astype(bool))
    return dict(control_samples=len(t),physics_samples=len(pt),recovery_recomputed_s=measured,
                bias_variance_max_residual_m2=identity_error,tail_null_correct=True,efficiency_single_application=True,
                union_dwell_and_integral_recomputed=True,policy_absolute_velocity_preserved=True)


def save_plots(directory,results,boundaries):
    from .plotting import _pyplot
    plt=_pyplot()
    style={'physical_termination':('x','C3'),'completed_position_unrecovered':('s','C1'),
           'position_only_recovered':('^','C4'),'position_attitude_recovered':('o','C2')}
    for label,name in COMBINATIONS:
        combo=label+'-'+name
        rs=sorted([r for r in results if r['combination']==combo and r['evaluated']],key=lambda r:r['efficiency_percent'])
        def metric_plot(file,descriptors):
            fig,axes=plt.subplots(len(descriptors),1,figsize=(11,3.4*len(descriptors)),sharex=True,squeeze=False)
            for ax,(title,series) in zip(axes[:,0],descriptors):
                for j,(legend,getter) in enumerate(series):
                    values=[getter(r) for r in rs]
                    ax.plot([r['efficiency_percent'] for r in rs],[np.nan if v is None else v for v in values],
                            '-',color=f'C{j}',label=legend,alpha=.75)
                    for r,value in zip(rs,values):
                        if value is not None:
                            marker,color=style[r['category']]
                            ax.scatter(r['efficiency_percent'],value,marker=marker,color=color,s=32,zorder=3)
                # Missing recovery/tail values live in an axis-coordinate status
                # strip, never at a numeric zero value on the data axis.
                for r in rs:
                    text='N/A' if file=='recovery' and r['efficiency_percent']==100 else 'FAIL' if r['terminated'] else 'NR' if not r['position_criterion_met'] else ''
                    if text:ax.text(r['efficiency_percent'],.98,text,transform=ax.get_xaxis_transform(),rotation=90,va='top',ha='center',fontsize=7,color='C3')
                ax.set_xlim(min(r['efficiency_percent'] for r in rs)-1.5,101.5)
                ax.set_ylabel(title);ax.grid(alpha=.25);ax.legend(fontsize=8)
            axes[-1,0].set_xlabel('Target motor effectiveness [%] (independent runs)')
            fig.suptitle(combo+' / fixed oracle + integral 0.20\n'
                         'circles: joint; triangles: position only; squares: unrecovered; x: physical end',fontsize=10)
            fig.tight_layout(rect=(0,0,1,.96));fig.savefig(directory/(combo+'-'+file+'.png'),dpi=130);plt.close(fig)
        metric_plot('recovery', [('Recovery latency after 5s event [s]',[(k,lambda r,k=k:r[k]) for k in
            ('recovery_xy_s','recovery_z_s','recovery_3d_s','recovery_position_attitude_s')])])
        metric_plot('max-errors', [('Post-event XY error [m]',[('max XY (observed; partial on failure)',lambda r:r['post5_observed_metrics']['max_xy_error_m'])]),
                                  ('Post-event |Z error| [m]',[('max |Z| (observed; partial on failure)',lambda r:r['post5_observed_metrics']['max_abs_z_error_m'])])])
        metric_plot('tail', [(title,[(key,lambda r,key=key:r['tail58_60'][key] if r['tail58_60'] else None)]) for title,key in
            [('58-60 XY offset [m]','offset_xy_m'),('58-60 signed Z mean [m]','mean_error_z_m'),
             ('58-60 XY mean-around RMS [m]','sway_xy_rms_m'),('58-60 Z mean-around RMS [m]','sway_z_rms_m')]])
        metric_plot('dynamics', [('Post-event max tilt [deg]',[('tilt',lambda r:r['post5_observed_metrics']['max_tilt_deg'])]),
            ('Post-event max angular speed [rad/s]',[('omega',lambda r:r['post5_observed_metrics']['max_angular_speed_rad_s'])]),
            ('Full observed union duration [s]',[(k,lambda r,k=k:r['dwell_full'][k]['duration_any_s']) for k in
             ('allocator_clipped','esc_boundary_union','policy_action_at_bound','integral_frozen')])])
        # Each transition compares both efficiencies in one shared coordinate
        # system. No independent autoscaling for success and failure traces.
        for boundary in boundaries[combo]['transition_intervals']:
            pair=[next(r for r in rs if r['efficiency_percent']==boundary[k]) for k in ('lower_percent','upper_percent')]
            traces=[(r,read_columns(directory/(r['key']+'.csv'))) for r in pair]
            for zoom,start,end in (('event',4.5,10.),('full',0.,60.)):
                fig,axes=plt.subplots(6,2,figsize=(15,20),sharex=True)
                for color,(r,f) in enumerate(traces):
                    kw=dict(color=f'C{color}',label=f'{r["efficiency_percent"]}% {r["category"]}')
                    e=vector(f,'position_error_world');motor=1 if 'motor2' in name else 2
                    ys=[np.linalg.norm(e[:,:2],axis=1),e[:,2],f.attitude_deg_0,f.attitude_deg_1,
                        tilt_deg(vector(f,'quaternion',4)),np.degrees(f.yaw_error_rad),
                        np.linalg.norm(vector(f,'omega'),axis=1),np.linalg.norm(vector(f,'xi_t')[:,:2],axis=1),
                        f.xi_t_2,f[f'motor_thrust_actual_{motor}'],f.interval_allocator_clipping.astype(float),f.integral_frozen.astype(float)]
                    for j,(ax,y) in enumerate(zip(axes.flat,ys)):
                        clock=f.time if j in (7,8,10,11) else f.motor_sample_time if j==9 else f.time_post
                        ax.plot(clock,y,**kw)
                        if r['terminated']:ax.axvline(r['actual_duration_sec'],color=kw['color'],ls=':',alpha=.6)
                    axes[4,1].plot(f.motor_sample_time,f[f'motor_thrust_nominal_{motor}'],color=kw['color'],ls='--',alpha=.5,label='nominal '+kw['label'])
                titles=['True XY error [m]','Signed true Z error [m]','Roll [deg]','Pitch [deg]','Tilt [deg]','Wrapped yaw error [deg]',
                        'Angular speed [rad/s]','xi XY norm [m]','xi Z [m]',f'Motor {motor+1} actual / dashed nominal [N]','Allocator clip 0/1','Integral frozen 0/1']
                for ax,title in zip(axes.flat,titles):
                    ax.set_ylabel(title);ax.set_xlim(start,end);ax.axvline(5,color='k',ls='--',lw=.8);ax.grid(alpha=.25);ax.legend(fontsize=6)
                for ax in axes[-1]:ax.set_xlabel('Simulation time [s]; input/xi: t, state: t+dt')
                fig.suptitle(combo+f' boundary {boundary["lower_percent"]}%-{boundary["upper_percent"]}% / '+zoom)
                fig.tight_layout(rect=(0,0,1,.98));fig.savefig(directory/(combo+f'-boundary-{boundary["lower_percent"]}-{boundary["upper_percent"]}-{zoom}.png'),dpi=130);plt.close(fig)
