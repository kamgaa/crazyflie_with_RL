"""MuJoCo-independent motor actuator models for the Crazyflie environment.

The controller/allocation path works in *requested motor thrust* (newtons).
This module converts that requested thrust into the force and reaction torque
that should be applied to the plant.  Keeping this code free of MuJoCo makes
the numerical behaviour easy to unit test and keeps the environment focused on
simulator I/O.

``InstantaneousActuatorModel`` remains a mathematical compatibility reference:
the allocated command is applied without a delay and its reaction torque is
the legacy constant-ratio value. Runtime configurations select
``Cf21bFirstOrderActuatorModel``. Its thrust polynomial is a paper-candidate
parameterisation; this module does not claim that it is validated for a
physical vehicle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


MOTOR_COUNT = 4
DEFAULT_MOTOR_DIRECTION = (1.0, -1.0, 1.0, -1.0)
DEFAULT_THRUST_POLYNOMIAL = (-0.23, 0.562, -0.043)
DEFAULT_TORQUE_POLYNOMIAL = (-3.4, 8.7, 2.9)
DEFAULT_OMEGA_REFERENCE_RAD_S = 2900.0


def _as_motor_vector(value: Sequence[float] | float, name: str) -> np.ndarray:
    """Return a finite four-motor vector, broadcasting one scalar if needed."""

    result = np.asarray(value, dtype=float)
    if result.ndim == 0:
        result = np.full(MOTOR_COUNT, float(result), dtype=float)
    if result.shape != (MOTOR_COUNT,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite scalar or shape ({MOTOR_COUNT},)")
    return result.copy()


def _positive_scalar(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _finite_scalar(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_range(
    value: Sequence[float], name: str, *, positive: bool = True
) -> tuple[float, float]:
    result = np.asarray(value, dtype=float)
    if result.shape != (2,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain exactly two finite values")
    lower, upper = float(result[0]), float(result[1])
    if lower > upper or (positive and lower <= 0.0):
        qualifier = "positive " if positive else ""
        raise ValueError(f"{name} must be an ordered {qualifier}range")
    return lower, upper


@dataclass(frozen=True)
class ActuatorOutput:
    """Per-physics-substep actuator values.

    ``f_cmd`` is the clipped requested thrust held by the allocator, while
    ``f_actual`` and ``q_actual`` are what should be applied to the plant.
    ``motor_command`` and ``omega`` are ``NaN`` for the legacy instantaneous
    model because it has no ESC or rotor-speed state.

    Wrench fields are available only when an allocation matrix was supplied to
    the model.  The actual yaw entry uses ``sum(q_actual)`` so a non-legacy
    reaction-torque law is represented correctly.
    """

    f_cmd: np.ndarray
    f_actual: np.ndarray
    motor_command: np.ndarray
    omega: np.ndarray
    q_actual: np.ndarray
    wrench_cmd: np.ndarray | None = None
    wrench_actual: np.ndarray | None = None
    allocation_error: np.ndarray | None = None


@dataclass(frozen=True)
class ActuatorParameters:
    """Current, potentially episode-sampled first-order motor parameters."""

    time_constant_s: np.ndarray
    steady_state_gain_rad_s: np.ndarray


class _BaseActuatorModel:
    """Shared validation, diagnostics, and legacy reaction-torque helpers."""

    def __init__(
        self,
        *,
        motor_direction: Sequence[float] = DEFAULT_MOTOR_DIRECTION,
        thrust_min: float = 0.0,
        thrust_max: float = 0.20,
        legacy_ratio_m: float = 0.00594,
        allocation_matrix: Sequence[Sequence[float]] | None = None,
    ) -> None:
        self.motor_direction = _as_motor_vector(motor_direction, "motor_direction")
        self.thrust_min = _finite_scalar(thrust_min, "thrust_min")
        self.thrust_max = _positive_scalar(thrust_max, "thrust_max")
        if self.thrust_min < 0.0 or self.thrust_max <= self.thrust_min:
            raise ValueError("thrust_max must exceed non-negative thrust_min")
        self.legacy_ratio_m = _positive_scalar(legacy_ratio_m, "legacy_ratio_m")

        if allocation_matrix is None:
            self.allocation_matrix: np.ndarray | None = None
        else:
            allocation = np.asarray(allocation_matrix, dtype=float)
            if allocation.shape != (MOTOR_COUNT, MOTOR_COUNT) or not np.all(
                np.isfinite(allocation)
            ):
                raise ValueError("allocation_matrix must be a finite shape (4, 4) array")
            self.allocation_matrix = allocation.copy()

        self._last_output = self._make_output(
            f_cmd=np.zeros(MOTOR_COUNT),
            f_actual=np.zeros(MOTOR_COUNT),
            motor_command=np.full(MOTOR_COUNT, np.nan),
            omega=np.full(MOTOR_COUNT, np.nan),
            q_actual=np.zeros(MOTOR_COUNT),
        )

    @property
    def last_output(self) -> ActuatorOutput:
        """Most-recent output (arrays are copies to protect internal state)."""

        return self._copy_output(self._last_output)

    @property
    def last_f_cmd(self) -> np.ndarray:
        return self._last_output.f_cmd.copy()

    @property
    def last_f(self) -> np.ndarray:
        return self._last_output.f_actual.copy()

    @property
    def last_motor_cmd(self) -> np.ndarray:
        return self._last_output.motor_command.copy()

    @property
    def last_omega(self) -> np.ndarray:
        return self._last_output.omega.copy()

    @property
    def last_q_actual(self) -> np.ndarray:
        return self._last_output.q_actual.copy()

    @property
    def last_wrench_cmd(self) -> np.ndarray | None:
        return self._copy_optional(self._last_output.wrench_cmd)

    @property
    def last_wrench_actual(self) -> np.ndarray | None:
        return self._copy_optional(self._last_output.wrench_actual)

    @property
    def last_allocation_error(self) -> np.ndarray | None:
        return self._copy_optional(self._last_output.allocation_error)

    def _clip_thrust_command(self, f_cmd: Sequence[float]) -> np.ndarray:
        result = _as_motor_vector(f_cmd, "f_cmd")
        return np.clip(result, self.thrust_min, self.thrust_max)

    def _legacy_reaction_torque(self, f_actual: np.ndarray) -> np.ndarray:
        return self.motor_direction * self.legacy_ratio_m * f_actual

    @staticmethod
    def _copy_optional(value: np.ndarray | None) -> np.ndarray | None:
        return None if value is None else value.copy()

    @classmethod
    def _copy_output(cls, output: ActuatorOutput) -> ActuatorOutput:
        return ActuatorOutput(
            f_cmd=output.f_cmd.copy(),
            f_actual=output.f_actual.copy(),
            motor_command=output.motor_command.copy(),
            omega=output.omega.copy(),
            q_actual=output.q_actual.copy(),
            wrench_cmd=cls._copy_optional(output.wrench_cmd),
            wrench_actual=cls._copy_optional(output.wrench_actual),
            allocation_error=cls._copy_optional(output.allocation_error),
        )

    def _make_output(
        self,
        *,
        f_cmd: np.ndarray,
        f_actual: np.ndarray,
        motor_command: np.ndarray,
        omega: np.ndarray,
        q_actual: np.ndarray,
    ) -> ActuatorOutput:
        command = np.asarray(f_cmd, dtype=float).copy()
        actual = np.asarray(f_actual, dtype=float).copy()
        command_u = np.asarray(motor_command, dtype=float).copy()
        omega_value = np.asarray(omega, dtype=float).copy()
        reaction = np.asarray(q_actual, dtype=float).copy()
        if self.allocation_matrix is None:
            wrench_cmd = wrench_actual = allocation_error = None
        else:
            wrench_cmd = self.allocation_matrix @ command
            wrench_actual = self.allocation_matrix @ actual
            wrench_actual[2] = float(np.sum(reaction))
            allocation_error = wrench_cmd - wrench_actual
        return ActuatorOutput(
            f_cmd=command,
            f_actual=actual,
            motor_command=command_u,
            omega=omega_value,
            q_actual=reaction,
            wrench_cmd=wrench_cmd,
            wrench_actual=wrench_actual,
            allocation_error=allocation_error,
        )


class InstantaneousActuatorModel(_BaseActuatorModel):
    """Ideal compatibility/reference actuator, not a runtime profile choice.

    The model has no rate state.  It preserves the legacy relation exactly for
    an allocator-clipped command: ``f_actual == f_cmd`` and
    ``q_actual = direction * legacy_ratio_m * f_cmd``.
    """

    def __init__(
        self,
        *,
        motor_direction: Sequence[float] = DEFAULT_MOTOR_DIRECTION,
        thrust_min: float = 0.0,
        thrust_max: float = 0.20,
        legacy_ratio_m: float = 0.00594,
        torque_coefficient: float | None = None,
        allocation_matrix: Sequence[Sequence[float]] | None = None,
    ) -> None:
        if torque_coefficient is not None:
            coefficient = _positive_scalar(torque_coefficient, "torque_coefficient")
            if not np.isclose(coefficient, legacy_ratio_m, rtol=0.0, atol=0.0):
                raise ValueError(
                    "instantaneous legacy torque must use the allocator torque "
                    "coefficient"
                )
        super().__init__(
            motor_direction=motor_direction,
            thrust_min=thrust_min,
            thrust_max=thrust_max,
            legacy_ratio_m=legacy_ratio_m,
            allocation_matrix=allocation_matrix,
        )

    def reset(
        self,
        *,
        airborne: bool = False,
        episode_mass: float | None = None,
        gravity_m_s2: float = 9.81,
        **_unused: object,
    ) -> ActuatorOutput:
        """Clear diagnostics; ideal-actuator reset never creates motor state."""

        del airborne, episode_mass, gravity_m_s2
        zero = np.zeros(MOTOR_COUNT)
        self._last_output = self._make_output(
            f_cmd=zero,
            f_actual=zero,
            motor_command=np.full(MOTOR_COUNT, np.nan),
            omega=np.full(MOTOR_COUNT, np.nan),
            q_actual=zero,
        )
        return self.last_output

    def apply(self, f_cmd: Sequence[float]) -> ActuatorOutput:
        command = self._clip_thrust_command(f_cmd)
        self._last_output = self._make_output(
            f_cmd=command,
            f_actual=command,
            motor_command=np.full(MOTOR_COUNT, np.nan),
            omega=np.full(MOTOR_COUNT, np.nan),
            q_actual=self._legacy_reaction_torque(command),
        )
        return self.last_output

    step = apply


class Cf21bFirstOrderActuatorModel(_BaseActuatorModel):
    """Runtime first-order BLDC/ESC actuator model.

    ``apply(f_cmd)`` is called once per physics substep, not once per policy
    action.  It uses a fixed-count bounded bisection of the explicit positive
    thrust branch; consequently no unconstrained root solver is run at runtime.
    The supplied thrust polynomial is intentionally treated as a configurable,
    unverified paper candidate.
    """

    _INVERSE_ITERATIONS = 48
    _THRUST_EPSILON_N = 1e-12

    def __init__(
        self,
        *,
        dt: float,
        motor_direction: Sequence[float] = DEFAULT_MOTOR_DIRECTION,
        thrust_min: float = 0.0,
        thrust_max: float = 0.20,
        time_constant_s: Sequence[float] | float = 0.050,
        steady_state_gain_rad_s: Sequence[float] | float = 2900.0,
        thrust_polynomial_coefficients: Sequence[float] = DEFAULT_THRUST_POLYNOMIAL,
        omega_reference_rad_s: float = DEFAULT_OMEGA_REFERENCE_RAD_S,
        positive_branch_min_ratio: float | None = None,
        max_ratio: float = 1.0,
        reaction_torque_model: str = "legacy_ratio",
        legacy_ratio_m: float = 0.00594,
        torque_polynomial_coefficients: Sequence[float] = DEFAULT_TORQUE_POLYNOMIAL,
        torque_polynomial_scale: float = 1e-4,
        rotor_inertia_kg_m2: float = 0.5e-7,
        include_rotor_acceleration_torque: bool = False,
        allocation_matrix: Sequence[Sequence[float]] | None = None,
    ) -> None:
        super().__init__(
            motor_direction=motor_direction,
            thrust_min=thrust_min,
            thrust_max=thrust_max,
            legacy_ratio_m=legacy_ratio_m,
            allocation_matrix=allocation_matrix,
        )
        self.dt = _positive_scalar(dt, "dt")
        self.nominal_time_constant_s = _as_motor_vector(
            time_constant_s, "time_constant_s"
        )
        self.nominal_steady_state_gain_rad_s = _as_motor_vector(
            steady_state_gain_rad_s, "steady_state_gain_rad_s"
        )
        if np.any(self.nominal_time_constant_s <= 0.0):
            raise ValueError("time_constant_s must be positive")
        if np.any(self.nominal_steady_state_gain_rad_s <= 0.0):
            raise ValueError("steady_state_gain_rad_s must be positive")
        self.time_constant_s = self.nominal_time_constant_s.copy()
        self.steady_state_gain_rad_s = (
            self.nominal_steady_state_gain_rad_s.copy()
        )

        coefficients = np.asarray(thrust_polynomial_coefficients, dtype=float)
        if coefficients.shape != (3,) or not np.all(np.isfinite(coefficients)):
            raise ValueError("thrust_polynomial_coefficients must be three finite values")
        self.thrust_polynomial_coefficients = coefficients.copy()
        self.omega_reference_rad_s = _positive_scalar(
            omega_reference_rad_s, "omega_reference_rad_s"
        )
        self.max_ratio = _positive_scalar(max_ratio, "max_ratio")
        derived_branch = self._derive_positive_branch_start(self.max_ratio)
        if positive_branch_min_ratio is None:
            self.positive_branch_min_ratio = derived_branch
        else:
            explicit_branch = _finite_scalar(
                positive_branch_min_ratio, "positive_branch_min_ratio"
            )
            if explicit_branch < derived_branch - 1e-10:
                raise ValueError(
                    "positive_branch_min_ratio enters the negative-thrust branch"
                )
            self.positive_branch_min_ratio = explicit_branch
        if self.positive_branch_min_ratio > self.max_ratio:
            raise ValueError("positive_branch_min_ratio must not exceed max_ratio")
        self._validate_positive_monotonic_branch()
        self._branch_min_thrust = self._thrust_polynomial(
            self.positive_branch_min_ratio
        )
        self._branch_max_thrust = self._thrust_polynomial(self.max_ratio)
        self._mapping_thrust_max = min(self.thrust_max, self._branch_max_thrust)
        if self._mapping_thrust_max <= self._THRUST_EPSILON_N:
            raise ValueError("thrust mapping produces no positive thrust in its branch")

        normalized_torque_model = str(reaction_torque_model).strip().lower()
        aliases = {
            "legacy_ratio": "legacy_ratio",
            "paper_polynomial": "paper_polynomial",
            "cf21b_torque_poly": "paper_polynomial",
        }
        if normalized_torque_model not in aliases:
            raise ValueError(
                "reaction_torque_model must be 'legacy_ratio' or 'paper_polynomial'"
            )
        self.reaction_torque_model = aliases[normalized_torque_model]
        torque_coefficients = np.asarray(torque_polynomial_coefficients, dtype=float)
        if torque_coefficients.shape != (3,) or not np.all(
            np.isfinite(torque_coefficients)
        ):
            raise ValueError("torque_polynomial_coefficients must be three finite values")
        self.torque_polynomial_coefficients = torque_coefficients.copy()
        self.torque_polynomial_scale = _positive_scalar(
            torque_polynomial_scale, "torque_polynomial_scale"
        )
        self.rotor_inertia_kg_m2 = _positive_scalar(
            rotor_inertia_kg_m2, "rotor_inertia_kg_m2"
        )
        if not isinstance(include_rotor_acceleration_torque, (bool, np.bool_)):
            raise ValueError("include_rotor_acceleration_torque must be a boolean")
        self.include_rotor_acceleration_torque = bool(
            include_rotor_acceleration_torque
        )
        self.omega = np.zeros(MOTOR_COUNT)
        self._last_motor_command = np.zeros(MOTOR_COUNT)

    @property
    def parameters(self) -> ActuatorParameters:
        """Return current sampled parameters, independent for all four motors."""

        return ActuatorParameters(
            time_constant_s=self.time_constant_s.copy(),
            steady_state_gain_rad_s=self.steady_state_gain_rad_s.copy(),
        )

    @property
    def nominal_parameters(self) -> ActuatorParameters:
        """Return the profile values before per-episode randomisation."""

        return ActuatorParameters(
            time_constant_s=self.nominal_time_constant_s.copy(),
            steady_state_gain_rad_s=self.nominal_steady_state_gain_rad_s.copy(),
        )

    @property
    def sampled_time_constant_s(self) -> np.ndarray:
        """Current per-motor time constants, suitable for a run manifest."""

        return self.time_constant_s.copy()

    @property
    def sampled_steady_state_gain_rad_s(self) -> np.ndarray:
        """Current per-motor gains, suitable for a run manifest."""

        return self.steady_state_gain_rad_s.copy()

    @property
    def decay(self) -> np.ndarray:
        """Exact-discretisation coefficient ``exp(-dt / T_i)``."""

        return np.exp(-self.dt / self.time_constant_s)

    def set_parameters(
        self,
        *,
        time_constant_s: Sequence[float] | float | None = None,
        steady_state_gain_rad_s: Sequence[float] | float | None = None,
    ) -> ActuatorParameters:
        """Set explicit episode parameters without drawing from any RNG stream."""

        if time_constant_s is not None:
            result = _as_motor_vector(time_constant_s, "time_constant_s")
            if np.any(result <= 0.0):
                raise ValueError("time_constant_s must be positive")
            self.time_constant_s = result
        if steady_state_gain_rad_s is not None:
            result = _as_motor_vector(
                steady_state_gain_rad_s, "steady_state_gain_rad_s"
            )
            if np.any(result <= 0.0):
                raise ValueError("steady_state_gain_rad_s must be positive")
            self.steady_state_gain_rad_s = result
        return self.parameters

    def restore_nominal_parameters(self) -> ActuatorParameters:
        """Restore the constructor values after a randomised episode."""

        self.time_constant_s = self.nominal_time_constant_s.copy()
        self.steady_state_gain_rad_s = self.nominal_steady_state_gain_rad_s.copy()
        return self.parameters

    def sample_parameters(
        self,
        rng: np.random.Generator,
        *,
        enabled: bool,
        time_constant_range: Sequence[float] | None = None,
        steady_state_gain_range: Sequence[float] | None = None,
    ) -> ActuatorParameters:
        """Sample independent motor parameters from a caller-owned RNG.

        The caller owns the RNG stream so actuator sampling cannot alter the
        existing payload/reset draw order.  Passing ``enabled=False`` performs
        no draw and restores nominal values.
        """

        if not isinstance(enabled, (bool, np.bool_)):
            raise ValueError("enabled must be a boolean")
        if not enabled:
            return self.restore_nominal_parameters()
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        if time_constant_range is None or steady_state_gain_range is None:
            raise ValueError("randomization ranges are required when enabled")
        tau_min, tau_max = _validate_range(time_constant_range, "time_constant_range")
        gain_min, gain_max = _validate_range(
            steady_state_gain_range, "steady_state_gain_range"
        )
        self.time_constant_s = rng.uniform(tau_min, tau_max, size=MOTOR_COUNT)
        self.steady_state_gain_rad_s = rng.uniform(
            gain_min, gain_max, size=MOTOR_COUNT
        )
        return self.parameters

    def reset(
        self,
        *,
        airborne: bool = False,
        episode_mass: float | None = None,
        gravity_m_s2: float = 9.81,
        rng: np.random.Generator | None = None,
        randomize: bool = False,
        time_constant_range: Sequence[float] | None = None,
        steady_state_gain_range: Sequence[float] | None = None,
    ) -> ActuatorOutput:
        """Reset rotor state for either a ground or an airborne episode.

        Ground starts use zero rotor speed.  Airborne starts initialise each
        motor at the positive-branch speed producing ``episode_mass * g / 4``.
        Randomisation is explicitly opt-in and requires a caller-owned RNG.
        """

        if randomize:
            if rng is None:
                raise ValueError("rng is required when randomize=True")
            self.sample_parameters(
                rng,
                enabled=True,
                time_constant_range=time_constant_range,
                steady_state_gain_range=steady_state_gain_range,
            )
        if not isinstance(airborne, (bool, np.bool_)):
            raise ValueError("airborne must be a boolean")
        if airborne:
            if episode_mass is None:
                raise ValueError("episode_mass is required for an airborne reset")
            mass = _positive_scalar(episode_mass, "episode_mass")
            gravity = _positive_scalar(gravity_m_s2, "gravity_m_s2")
            hover_thrust = mass * gravity / MOTOR_COUNT
            command = np.full(
                MOTOR_COUNT,
                min(max(hover_thrust, self.thrust_min), self._mapping_thrust_max),
            )
            self.omega = self.inverse_thrust(command)
            motor_command = np.clip(
                self.omega / self.steady_state_gain_rad_s, 0.0, 1.0
            )
        else:
            command = np.zeros(MOTOR_COUNT)
            self.omega = np.zeros(MOTOR_COUNT)
            motor_command = np.zeros(MOTOR_COUNT)
        self._last_motor_command = motor_command.copy()
        actual = self.thrust_from_omega(self.omega)
        reaction = self._reaction_torque(actual, self.omega, np.zeros(MOTOR_COUNT))
        self._last_output = self._make_output(
            f_cmd=command,
            f_actual=actual,
            motor_command=motor_command,
            omega=self.omega,
            q_actual=reaction,
        )
        return self.last_output

    def thrust_from_omega(self, omega: Sequence[float] | float) -> np.ndarray:
        """Forward candidate mapping with explicit low/high branch saturation."""

        speed = _as_motor_vector(omega, "omega")
        ratio = np.clip(speed / self.omega_reference_rad_s, 0.0, self.max_ratio)
        thrust = self._thrust_polynomial(ratio)
        thrust = np.where(ratio <= self.positive_branch_min_ratio, 0.0, thrust)
        thrust = np.clip(thrust, 0.0, self._mapping_thrust_max)
        return thrust

    def inverse_thrust(self, f_cmd: Sequence[float] | float) -> np.ndarray:
        """Bounded fixed-iteration inverse on the positive monotonic branch."""

        command = _as_motor_vector(f_cmd, "f_cmd")
        command = np.clip(command, self.thrust_min, self._mapping_thrust_max)
        target = np.zeros(MOTOR_COUNT)
        active = command > max(self._THRUST_EPSILON_N, self._branch_min_thrust)
        if not np.any(active):
            return target

        lower = np.full(MOTOR_COUNT, self.positive_branch_min_ratio)
        upper = np.full(MOTOR_COUNT, self.max_ratio)
        # Fixed-count bisection makes execution deterministic and bounded.  It
        # intentionally does not call a general-purpose root solver per step.
        for _ in range(self._INVERSE_ITERATIONS):
            midpoint = 0.5 * (lower + upper)
            value = self._thrust_polynomial(midpoint)
            lower = np.where((value < command) & active, midpoint, lower)
            upper = np.where((value >= command) & active, midpoint, upper)
        ratio = np.where(active, 0.5 * (lower + upper), 0.0)
        return ratio * self.omega_reference_rad_s

    def apply(self, f_cmd: Sequence[float]) -> ActuatorOutput:
        """Advance rotor dynamics by one fixed physics substep for ``f_cmd``."""

        command = self._clip_thrust_command(f_cmd)
        target_omega = self.inverse_thrust(command)
        motor_command = np.clip(
            target_omega / self.steady_state_gain_rad_s, 0.0, 1.0
        )
        return self._advance(command, motor_command)

    step = apply

    def apply_motor_command(self, motor_command: Sequence[float]) -> ActuatorOutput:
        """Advance one substep from an explicit normalized ESC command.

        This method is mainly useful for diagnostics and unit tests.  Normal
        environment integration should call :meth:`apply` with allocator thrust
        commands so the inverse mapping remains part of the actuator path.
        """

        command_u = np.clip(_as_motor_vector(motor_command, "motor_command"), 0.0, 1.0)
        target_omega = command_u * self.steady_state_gain_rad_s
        requested = self.thrust_from_omega(target_omega)
        return self._advance(requested, command_u)

    def _advance(
        self, f_cmd: np.ndarray, motor_command: np.ndarray
    ) -> ActuatorOutput:
        omega_before = self.omega.copy()
        decay = self.decay
        self.omega = decay * self.omega + (1.0 - decay) * (
            self.steady_state_gain_rad_s * motor_command
        )
        actual = self.thrust_from_omega(self.omega)
        omega_dot = (self.omega - omega_before) / self.dt
        reaction = self._reaction_torque(actual, self.omega, omega_dot)
        self._last_motor_command = motor_command.copy()
        self._last_output = self._make_output(
            f_cmd=f_cmd,
            f_actual=actual,
            motor_command=motor_command,
            omega=self.omega,
            q_actual=reaction,
        )
        return self.last_output

    def _reaction_torque(
        self, f_actual: np.ndarray, omega: np.ndarray, omega_dot: np.ndarray
    ) -> np.ndarray:
        if self.reaction_torque_model == "legacy_ratio":
            return self._legacy_reaction_torque(f_actual)
        ratio = np.clip(omega / self.omega_reference_rad_s, 0.0, self.max_ratio)
        a3, a2, a1 = self.torque_polynomial_coefficients
        magnitude = self.torque_polynomial_scale * (
            a3 * ratio**3 + a2 * ratio**2 + a1 * ratio
        )
        magnitude = np.maximum(magnitude, 0.0)
        if self.include_rotor_acceleration_torque:
            magnitude = magnitude + self.rotor_inertia_kg_m2 * omega_dot
        return self.motor_direction * magnitude

    def _thrust_polynomial(self, ratio: np.ndarray | float) -> np.ndarray:
        a3, a2, a1 = self.thrust_polynomial_coefficients
        value = np.asarray(ratio, dtype=float)
        return a3 * value**3 + a2 * value**2 + a1 * value

    def _thrust_derivative(self, ratio: np.ndarray | float) -> np.ndarray:
        a3, a2, a1 = self.thrust_polynomial_coefficients
        value = np.asarray(ratio, dtype=float)
        return 3.0 * a3 * value**2 + 2.0 * a2 * value + a1

    def _derive_positive_branch_start(self, max_ratio: float) -> float:
        """Find the non-zero polynomial root which starts positive thrust."""

        a3, a2, a1 = self.thrust_polynomial_coefficients
        tolerance = 1e-14
        roots: list[float] = []
        if abs(a3) > tolerance:
            discriminant = a2 * a2 - 4.0 * a3 * a1
            if discriminant >= 0.0:
                root_scale = float(np.sqrt(discriminant))
                roots.extend(
                    [
                        (-a2 + root_scale) / (2.0 * a3),
                        (-a2 - root_scale) / (2.0 * a3),
                    ]
                )
        elif abs(a2) > tolerance:
            roots.append(-a1 / a2)
        else:
            raise ValueError("thrust polynomial must contain a quadratic or cubic term")

        candidates = sorted(
            root
            for root in roots
            if np.isfinite(root)
            and root >= 0.0
            and root <= max_ratio
            and self._thrust_derivative(root) > tolerance
        )
        if not candidates:
            raise ValueError(
                "thrust polynomial has no increasing positive-thrust branch "
                "within max_ratio"
            )
        return float(candidates[0])

    def _validate_positive_monotonic_branch(self) -> None:
        """Reject configurations whose inverse interval is not monotonic."""

        lower = self.positive_branch_min_ratio
        upper = self.max_ratio
        a3, a2, _a1 = self.thrust_polynomial_coefficients
        points = [lower, upper]
        if abs(a3) > 1e-14:
            vertex = -a2 / (3.0 * a3)
            if lower < vertex < upper:
                points.append(vertex)
        derivative = self._thrust_derivative(np.asarray(points))
        if np.any(derivative <= 1e-12):
            raise ValueError(
                "thrust mapping must be strictly increasing on its positive branch"
            )
        if self._thrust_polynomial(upper) <= self._thrust_polynomial(lower):
            raise ValueError("thrust mapping branch must produce increasing thrust")


__all__ = [
    "ActuatorOutput",
    "ActuatorParameters",
    "Cf21bFirstOrderActuatorModel",
    "DEFAULT_MOTOR_DIRECTION",
    "DEFAULT_OMEGA_REFERENCE_RAD_S",
    "DEFAULT_THRUST_POLYNOMIAL",
    "DEFAULT_TORQUE_POLYNOMIAL",
    "InstantaneousActuatorModel",
    "MOTOR_COUNT",
]
