"""Plots on the existing plotting backend; all traces retain actual timestamps."""
import numpy as np
from .integral_eval import read_columns
from .plotting import _pyplot

COLORS={'healthy':'black','blind_fault':'C0','oracle_fault':'C1','estimated_fault':'C2'}
NAMES={'healthy':'Healthy','blind_fault':'Blind fault','oracle_fault':'Oracle fault','estimated_fault':'Estimated fault'}


def plots(directory,results):
    plt=_pyplot()
    for label in ('PID','A_best','B_best'):
        selected=[r for r in results if r['label']==label]
        for kind in ('xy','tracking','efficiency','wrench'):
            n={'xy':1,'tracking':7,'efficiency':4,'wrench':5}[kind]
            fig,axes=plt.subplots(n,2,figsize=(15,5 if n==1 else 3.1*n),squeeze=False,
                                  sharey='row',layout='constrained')
            for col,T in enumerate((10.,5.)):
                group=[r for r in selected if r['period_s']==T]
                fault=6+2*T
                for r in group:
                    c=read_columns(directory/(r['key']+'.csv'));t=c.time_post;u=c.time
                    e=np.array([c[f'position_error_world_{j}'] for j in range(3)]).T
                    color=COLORS[r['variant']];name=NAMES[r['variant']]
                    if kind=='xy':
                        axes[0,col].plot(c.position_0,c.position_1,color=color,label=name)
                    elif kind=='tracking':
                        x=t-fault
                        values=[np.linalg.norm(e[:,:2],axis=1),e[:,2],c.radial_error_m,c.tangential_error_m,
                            np.rad2deg(c.yaw_error_rad),np.linalg.norm(np.array([c[f'omega_{j}'] for j in range(3)]).T,axis=1),c.integral_frozen.astype(float)]
                        for ax,v in zip(axes[:,col],values):ax.plot(x,v,color=color,label=name)
                    elif kind=='efficiency':
                        axes[0,col].step(u-fault,c.truth_efficiency_user_0,where='post',color=color,label=name+' truth')
                        axes[1,col].plot(t-fault,c.estimated_efficiency_user_0,color=color,label=name+' estimated')
                        axes[2,col].step(u-fault,c.allocator_efficiency_user_0,where='post',color=color,label=name+' used')
                        axes[3,col].step(t-fault,c.estimated_motor,where='post',color=color,label=name+' identified user motor')
                    else:
                        x=t-fault
                        axes[0,col].plot(u-fault,c.motor_thrust_unclipped_3,color=color,alpha=.6,label=name+' M1 raw')
                        axes[0,col].plot(u-fault,c.motor_thrust_command_3,color=color,ls='--',label=name+' clipped')
                        axes[1,col].plot(c.motor_sample_time_post-fault,c.motor_thrust_nominal_3,color=color,label=name+' nominal')
                        axes[1,col].plot(c.motor_sample_time_post-fault,c.motor_thrust_actual_3,color=color,ls='--',label=name+' actual')
                        axes[2,col].plot(x,np.linalg.norm(np.array([c[f'allocation_residual_xml_{j}'] for j in range(3)]),axis=0),color=color,label=name)
                        axes[3,col].plot(x,np.linalg.norm(np.array([c[f'actuator_response_residual_xml_{j}'] for j in range(3)]),axis=0),color=color,label=name)
                        axes[4,col].step(u-fault,c.allocator_clipping_union_seconds/.01,where='post',color=color,label=name)
                    if r['terminated'] and kind!='xy':
                        for ax in axes[:,col]:ax.axvline(r['duration_s']-fault,color=color,ls=':',alpha=.7)
                if kind=='xy':
                    theta=np.linspace(0,2*np.pi,501)
                    axes[0,col].plot(np.cos(theta),np.sin(theta),'k--',lw=1,label='Reference')
                    axes[0,col].set_aspect('equal');axes[0,col].set_xlabel('World X (m)');axes[0,col].set_ylabel('World Y (m)')
                else:
                    for ax in axes[:,col]:ax.axvline(0,color='k',ls='--',lw=1)
                    axes[-1,col].set_xlabel('Time from scheduled fault (s); healthy has no injected fault')
                axes[0,col].set_title(f'T = {T:g} s | '+('existing PID' if label=='PID' else label+' PPO + I'))
            labels={'tracking':['XY error (m)','Signed Z error (m)','Radial reference error (m)','Tangential reference error (m)','Yaw error (deg)','Angular speed (rad/s)','Integral frozen'],
                    'efficiency':['True M1 efficiency','Estimated M1 efficiency','Allocator M1 efficiency','Confirmed user motor (0 = none)'],
                    'wrench':['M1 allocation (N)','M1 lagged thrust (N)','Allocation torque residual norm (Nm)','Actuator torque response residual norm (Nm)','Clipping time fraction']}
            for i,row in enumerate(axes):
                for ax in row:
                    if kind!='xy':ax.set_ylabel(labels[kind][i])
                    ax.grid(alpha=.25);ax.legend(fontsize=7,ncol=2)
            fig.savefig(directory/f'{label}-{kind}.png',dpi=135)
            if kind=='wrench':
                for ax in axes.flat:ax.set_xlim(-.5,2.5)
                fig.savefig(directory/f'{label}-wrench-transient.png',dpi=135)
            plt.close(fig)
