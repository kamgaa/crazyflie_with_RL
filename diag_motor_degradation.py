"""Run four PID-only hover experiments; never loads or trains an RL policy."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import yaml
from crazyflie_rl.config import load_config
from crazyflie_rl.motor_degradation import Settings, experiment_config, rollout, summarize, plot_trace
import numpy as np


def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] in ("long-hover", "paired-pulse", "staircase", "study"):
        from crazyflie_rl.motor_study import main as study_main
        return study_main(sys.argv[1:])
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/diagnostics/pid_motor_degradation_diagnostic.yaml')
    parser.add_argument('--output-root',default='artifacts/pid_motor_degradation')
    for key in asdict(Settings()):
        parser.add_argument('--'+key.replace('_','-'),type=str if key=='disturbance_axis' else int if key in ('seed','motor_index') else float)
    args=parser.parse_args()
    config_path=Path(args.config).resolve()
    values=yaml.safe_load(config_path.read_text())
    base=load_config(config_path.parent/values.pop('experiment_config'))
    for key in asdict(Settings()):
        value=getattr(args,key)
        if value is not None: values[key]=value
    settings=Settings(**values); config=experiment_config(base,settings)
    root=Path(args.output_root); root.mkdir(parents=True,exist_ok=True)
    run=Path(tempfile.mkdtemp(prefix='pid_motor_degradation_',dir=root))
    for name in ('metrics','plots','traces','config'): (run/name).mkdir()
    (run/'config/resolved.json').write_text(json.dumps(asdict(config),default=str,indent=2))
    (run/'config/diagnostic.json').write_text(json.dumps(asdict(settings),indent=2))
    summaries={}
    for condition in ('BASE','A','B','C'):
        trace,end=rollout(config,settings,condition)
        np.savez_compressed(run/'traces'/f'{condition}.npz',**trace)
        result=summarize(trace,config,settings,condition,end)
        result['plots']=plot_trace(trace,settings,config,condition,run/'plots')
        (run/'metrics'/f'{condition}.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        summaries[condition]=result
    (run/'metrics/comparison.json').write_text(json.dumps(summaries,indent=2,allow_nan=False))
    print('Condition  pre pos RMS[m] pre att RMS[deg] peak pos[m] recovery pos[s]')
    for name,r in summaries.items():
        print(name,r['pre_disturbance']['position_rms_m'],r['pre_disturbance']['attitude_rms_deg'],r['recovery']['peak_position_error_m'],r['recovery']['position_recovery_time_s'])
    print('\nMetric                                      A                 C')
    def number(value):
        return 'null' if value is None else f'{value:.7g}'
    rows = {
        'Pre position RMS [m]': lambda r: r['pre_disturbance']['position_rms_m'],
        'Pre attitude RMS [deg]': lambda r: r['pre_disturbance']['attitude_rms_deg'],
        'Pre minimum raw margin [N]': lambda r: min(r['pre_disturbance']['motors']['raw_upper_margin_min_n']) if r['pre_disturbance']['motors'] else None,
        'Pre effective headroom [N]': lambda r: min(r['pre_disturbance']['motors']['effective_headroom_min_n']) if r['pre_disturbance']['motors'] else None,
        'Pre worst normalized motor command P99': lambda r: max(r['pre_disturbance']['motors']['motor_command_p99']) if r['pre_disturbance']['motors'] else None,
        'Post any-motor saturation fraction': lambda r: r['post_disturbance_motor_metrics']['any_motor_saturation_fraction'] if r['post_disturbance_motor_metrics'] else None,
    }
    for key in summaries['A']['recovery']:
        rows[key] = lambda r, key=key: r['recovery'][key]
    for label, extract in rows.items():
        print(f'{label:42s} {number(extract(summaries["A"])):>16s} {number(extract(summaries["C"])):>16s}')
    print('Artifacts:',run)


if __name__=='__main__':
    main()
