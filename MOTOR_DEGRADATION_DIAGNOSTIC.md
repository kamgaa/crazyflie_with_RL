# PID motor degradation diagnostic

Run `python diag_motor_degradation.py`. The standalone YAML in
`configs/diagnostics/pid_motor_degradation_diagnostic.yaml` points to an ordinary experiment
config. Existing training/evaluation entry points never import this diagnostic.
Only a diagnostic subclass modifies plant-side force/torque outputs, after the
unchanged nominal allocator, limits, and first-order motor dynamics. Neither PID
nor allocator receives effectiveness. The internal actuator RPM state is nominal;
loss represents downstream aerodynamic effectiveness, not an RPM/gain change.
The supported reaction model is explicitly restricted to `legacy_ratio`:
actual thrust and signed reaction torque are both multiplied by effectiveness.

Four conditions always share nominal initial pose, zero velocity, seed, PID,
limits, and actuator parameters. BASE/A have unit effectiveness; B/C activate the
configured loss. A/C receive the same body-frame torque pulse. The environment's
existing body-to-world transformation applies the torque to MuJoCo. Configs with
randomized payload/actuator or nonzero payload mass are rejected for this test.
No policy is loaded and no learning is performed.

Default pulse is 0.0001 Nm on roll for 0.5 s, starting at 6 s. With current roll
inertia 2.3951e-5 kg m² its open-loop acceleration is about 4.18 rad/s². Nominal
allocator motor thrust increment is about 0.000707 N (arm 0.035355 m), small
relative to 0.20 N motor limit. These figures explain the initial test amplitude;
CLI/config values remain adjustable and no outcome is guaranteed.

Examples:

```bash
python diag_motor_degradation.py --motor-index 0 --effectiveness 0.7 --seed 42
python diag_motor_degradation.py --disturbance-axis=-pitch --disturbance-torque 0.0001
```

Supported axes are roll/+roll/-roll, pitch/+pitch/-pitch, yaw/+yaw/-yaw.
Effectiveness must be in (0,1]. The config contains activation/pulse timing,
experiment duration, pre-window, recovery thresholds and hold duration.
A unique run directory is created atomically; previous artifacts are not reused.
`config/` stores resolved experiment and diagnostic settings, `traces/` stores
lossless NPZ arrays, `metrics/` stores four JSON summaries plus a comparison,
and `plots/` contains four figures per condition.

Samples represent the beginning of each physics substep, AFTER the PID and motor
state update but BEFORE MuJoCo integration. Command/actual wrench thus describe
the inputs for that interval; state describes its beginning. Termination time and
flags are recorded separately. No post-termination samples are fabricated.
PID `_i_vel` and `_i_rate` are copied; internal loop setpoints remain unrecorded
local variables, and requested torque is the rate-loop output. No controller
recomputation or control law modification is used to obtain diagnostics.

Raw demand upper margin is `f_max - raw_request` (may be negative); normalized
margin divides it by `f_max - f_min`. Clipped command headroom is
`f_max - clipped_request`. Ground-truth effective output headroom is
`lambda*f_max - actual_thrust`, including actuator lag. This is a per-motor
quantity, not a directional feasible-wrench set or a validated AMS measure.
Saturation fractions use raw requests at/beyond upper or lower limits.

Pre-window is [pulse_start-pre_window, pulse_start). Peak/integrated errors use
[pulse_start, observed_end). Integrals are physics-step left-rectangle sums of
squared position norm and squared principal identity-relative attitude angle
(radians). Recovery is the earliest below-threshold hold START after pulse END,
reported as elapsed time from pulse end. Zero means already within tolerance;
null means no qualifying observed hold. Position and attitude are independent.
Attitude is SO(3) angle, not independent Euler component errors. Tilt and wrapped
yaw error are also in NPZ traces. Crash results retain truncated observation
horizons, and should not be compared as full-duration integrals.

The default 70% test need not establish equal pre-disturbance tracking. Compare
BASE/A and B/C pre-window equality, then A/C tracking and margins before drawing
any interpretation. No causal or automatic sufficiency judgement is emitted.

## Settled hover and pulse study

```bash
python diag_motor_degradation.py study
python diag_motor_degradation.py long-hover --duration 30 --effectiveness-values 1 .9 .8 .7 .6 .55 .5
python diag_motor_degradation.py paired-pulse --effectiveness .7 --pulse-duration .1
python diag_motor_degradation.py staircase --effectiveness .7 --pulse-amplitudes .00005 .0001 .00015 .0002 .0003 .0004 .0005
```

The old no-subcommand four-condition CLI is preserved. Study subcommands read
`configs/diagnostics/pid_motor_study.yaml`, a standalone configuration, and write
a unique directory under `artifacts/runs` (override with `--output-root`). All
thresholds/timings in this config also accept CLI overrides. `study` executes all
three stages in order with one selected effectiveness. Individual pulse commands
first repeat the long-hover screening with their current config/seed; a manual
`--effectiveness` must be present and eligible in that sweep. No stale cross-run
selection file is implicitly trusted. Unordered amplitude lists are rejected,
not silently reordered. Each amplitude/sign/system uses a fresh environment and
seeded reset; no state or PID integral is carried across trials.

Settling time is the END of the first full `settle_hold_sec` RMS window after
loss activation satisfying position, principal attitude and velocity thresholds.
The last `steady_state_window_sec` window must independently satisfy all three
thresholds and the configured saturation-fraction limit, and the episode must
finish without termination. This empirical rule is not a stability proof. Among
eligible degraded levels the smallest effectiveness is the candidate; nominal
must also pass. A candidate is a diagnostic selection, not an optimal fault level.
If no degraded level passes, no pulse study is attempted. Early-terminated trials
have null final-window metrics rather than fabricated steady-state results.

`--timing-mode individual` uses each system's settle completion + buffer;
`common` uses the later of the two times for all conditions. Pulse times are
aligned upwards to physics substeps; activation and duration must already be on
the physics grid. The initial condition and seed remain identical. The buffer
must accommodate the entire pre-window. A passing settling screen does NOT imply
normal/degraded pre-states are identical: paired JSON includes their actual RMS
differences. This matters when interpreting peak-error differences.

New-study recovery is JOINT pointwise position/attitude/velocity threshold
satisfaction for the full recovery hold after pulse end. Reported latency is the
START of the qualifying hold relative to pulse end. Recording continues through
the complete post window even if recovery latency is zero. Numerical residuals
around 1e-15 s denote zero latency. Integrals use physics-step left rectangles
from pulse start to observed episode end, in m²s or rad²s. Early termination is
explicit and integrals on different observed horizons are not interchangeable.

Critical amplitudes are the first TESTED grid values violating each criterion.
They are not continuous critical thresholds. No event within the sweep produces
`lower_bound_only` with null amplitude/ratio, not infinity or a made-up ratio.
A missing recovery is counted against the time criterion only after the critical
deadline plus hold has actually been observed; termination has a separate flag.
Zero/zero recovery asymmetry is null because its normalization is undefined.

Outputs: `metrics/long_hover_sweep.json`, `paired_roll_pulse.json`,
`pulse_staircase.json`, `summary.json`; three corresponding PNGs in `plots/`;
physics-step NPZ per trial in `traces/`; source experiment and study config in
`config/`. The experiment uses the existing diagnostic residual/PID path and
nominal initial pose; production reward, PID, allocator, actuator and DR code
are unchanged. The default combined study performs 7 long hover + 4 paired +
28 independent staircase trials, without loading or training any RL model.
