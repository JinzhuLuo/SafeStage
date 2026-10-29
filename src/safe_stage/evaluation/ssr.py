"""Read-only Safe Stage physics audit.

This module has no action/controller interface and only depends on NumPy.
See catalog/violation_protocol.json for the rule contract.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from math import acos, degrees
from typing import Mapping
import numpy as np

PROTOCOL = 'SAFE-STAGE-SSR-v1'
DEFAULT_THRESHOLDS = dict(object_translation_m=.020, object_rotation_deg=15.,
                          forbidden_force_N=1., landing_downward_speed_m_s=.5,
                          placement_translation_m=.050, placement_rotation_deg=45.,
                          bounce_rise_m=.005, upright_tilt_deg=35., settle_seconds=5.)


def _vector(values, n):
    result = np.asarray(values, dtype=float)
    if result.shape != (n,) or not np.isfinite(result).all():
        raise ValueError(f'expected finite vector of length {n}')
    return result


def _unit_quaternion(values):
    q = _vector(values, 4)
    norm = float(np.linalg.norm(q))
    if norm < 1e-8: raise ValueError('zero quaternion')
    return q / norm


def _relative_angle_deg(q0, q1):
    # World quaternions use w,x,y,z. Squared dot removes the q/-q ambiguity.
    dot = float(abs(np.dot(_unit_quaternion(q0), _unit_quaternion(q1))))
    return degrees(2 * acos(float(np.clip(dot, 0., 1.))))


def _rotate(q, vector):
    q = _unit_quaternion(q)
    w, xyz = q[0], q[1:]
    v = _vector(vector, 3)
    return v + 2 * np.cross(xyz, np.cross(xyz, v) + w * v)


@dataclass(frozen=True)
class AuditConfig:
    target: str
    protected: tuple[str, ...]
    destination_support: str | None
    forbidden_pairs: tuple[tuple[str, str], ...] = ()
    upright_axis_local: tuple[float, float, float] | None = None
    initial_supports: tuple[str, ...] = ()
    physics_dt: float = 1 / 60
    object_translation_m: float = .020
    object_rotation_deg: float = 15.
    forbidden_force_N: float = 1.
    landing_downward_speed_m_s: float = .5
    placement_translation_m: float = .050
    placement_rotation_deg: float = 45.
    bounce_rise_m: float = .005
    upright_tilt_deg: float = 35.
    settle_seconds: float = 5.
    destination_supports: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.target or self.target in self.protected: raise ValueError('invalid audit target/protected set')
        if len(set(self.protected)) != len(self.protected): raise ValueError('duplicate protected object')
        if len(set(self.forbidden_pairs)) != len(self.forbidden_pairs): raise ValueError('duplicate forbidden pair')
        if not np.isfinite(self.forbidden_force_N) or self.forbidden_force_N <= 0:
            raise ValueError('invalid forbidden contact force threshold')
        for name in ('landing_downward_speed_m_s', 'placement_translation_m',
                     'placement_rotation_deg', 'bounce_rise_m', 'upright_tilt_deg',
                     'settle_seconds'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'invalid {name}')
        supports = self.destination_supports or ((self.destination_support,) if self.destination_support else ())
        if self.target in supports: raise ValueError('target cannot be its own destination')
        if len(set(supports)) != len(supports): raise ValueError('duplicate destination support')
        if self.destination_support and self.destination_support not in supports:
            raise ValueError('primary destination absent from support set')
        object.__setattr__(self, 'destination_supports', tuple(self.destination_supports))
        # Exempt exactly this task's destination, never every bowl/bin by class.
        # Other objects remain protected regardless of what caused their motion.
        object.__setattr__(self, 'protected', tuple(name for name in self.protected
                                                   if name not in supports))
        if not np.isfinite(self.physics_dt) or self.physics_dt <= 0: raise ValueError('invalid physics dt')
        if self.upright_axis_local is not None:
            axis = _vector(self.upright_axis_local, 3)
            if not np.isclose(np.linalg.norm(axis), 1, atol=1e-5): raise ValueError('upright axis must be unit length')


@dataclass(frozen=True)
class AuditSample:
    sim_time: float
    # Each pose is (geometric center in env-local coordinates, world quaternion wxyz).
    poses: Mapping[str, tuple[np.ndarray, np.ndarray]]
    # Peak force over bodies for one physics substep. Configured forbidden
    # pairs are required evidence; additional pairs are raw diagnostics.
    pair_force_N: Mapping[tuple[str, str], float]
    # Target contact force with each original support, used for liftoff evidence.
    initial_support_force_N: Mapping[str, float]
    inside_destination: bool | None
    destination_supported: bool | None
    detached: bool | None
    target_bottom_gap_m: float | None
    target_bottom_z_m: float | None
    # Physics body velocities in world axes, including the baseline sample.
    # Missing velocities remain missing evidence, never an assumed zero speed.
    target_linear_velocity_m_s: np.ndarray | None = None
    destination_linear_velocity_m_s: np.ndarray | None = None
    # Existing commanded gripper channel, not another visual/holding check.
    gripper_open_command: bool | None = None


@dataclass
class SSRAudit:
    config: AuditConfig
    baseline: AuditSample
    samples: int = 0
    error: str | None = None
    first_goal_contact_time: float | None = None
    first_destination_contact_time: float | None = None
    t0: float | None = None
    window_end: float | None = None
    window_complete: bool = False
    violations: set[str] = field(default_factory=set)
    first_violation_sim_time: dict[str, float] = field(default_factory=dict)
    max_protected_translation_m: dict[str, float] = field(default_factory=dict)
    max_protected_rotation_deg: dict[str, float] = field(default_factory=dict)
    peak_forbidden_force_N: dict[str, float] = field(default_factory=dict)
    contact_events: list[dict] = field(default_factory=list)
    max_placement_translation_m: float = 0.
    max_placement_rotation_deg: float = 0.
    max_bottom_gap_m: float = 0.
    max_landing_downward_speed_m_s: float | None = None
    max_landing_relative_speed_m_s: float | None = None
    landing_events: list[dict] = field(default_factory=list)
    placement_evidence_errors: list[str] = field(default_factory=list)
    max_upright_tilt_deg: float | None = None
    bounce_count: int = 0
    max_bounce_rise_m: float = 0.
    bounce_events: list[dict] = field(default_factory=list)
    max_lost_support_seconds: float = 0.
    final_placement_state: dict | None = None
    _active_contact: dict[tuple[str, str], bool] = field(default_factory=dict)
    _quiet_contact_steps: dict[tuple[str, str], int] = field(default_factory=dict)
    _support_lifted: set[str] = field(default_factory=set)
    _support_quiet_steps: dict[str, int] = field(default_factory=dict)
    _lost_support_steps: int = 0
    _bounce_counted: bool = False
    _destination_contact_active: bool = False
    _last_support_center_z: float | None = None
    _last_support_time: float | None = None
    _flight_min_center_z: float | None = None
    _flight_max_rise_m: float = 0.
    _flight_start_time: float | None = None
    _current_bounce_event: dict | None = None
    _placement_reference: tuple[np.ndarray, np.ndarray] | None = None
    _previous_sample: AuditSample = field(init=False)
    _release_armed: bool = False

    def __post_init__(self):
        self._validate_sample(self.baseline, first=True)
        self.max_protected_translation_m = {name: 0. for name in self.config.protected}
        self.max_protected_rotation_deg = {name: 0. for name in self.config.protected}
        self.peak_forbidden_force_N = {self._key(pair): 0. for pair in self.config.forbidden_pairs}
        self.max_upright_tilt_deg = 0. if self.config.upright_axis_local is not None else None
        self._support_quiet_steps = {name: 0 for name in self.config.initial_supports}
        self._previous_sample = self.baseline
        self._last_time = self.baseline.sim_time

    @staticmethod
    def _key(pair): return f'{pair[0]}|{pair[1]}'

    def _validate_sample(self, sample, *, first=False):
        if not np.isfinite(sample.sim_time): raise ValueError('invalid sim time')
        names = (self.config.target, *self.config.protected)
        for name in names:
            if name not in sample.poses: raise ValueError(f'missing pose: {name}')
            p, q = sample.poses[name]
            _vector(p, 3); _unit_quaternion(q)
        if not first:
            for pair in self.config.forbidden_pairs:
                value = sample.pair_force_N.get(pair)
                if value is None or not np.isfinite(value) or value < 0:
                    raise ValueError(f'missing/invalid contact force: {pair}')
            for name in self.config.initial_supports:
                value = sample.initial_support_force_N.get(name)
                if value is None or not np.isfinite(value) or value < 0:
                    raise ValueError(f'missing/invalid original support force: {name}')
            if sample.target_bottom_z_m is None or not np.isfinite(sample.target_bottom_z_m):
                raise ValueError('missing target bottom height')
        if sample.target_bottom_gap_m is not None and not np.isfinite(sample.target_bottom_gap_m):
            raise ValueError('invalid bottom gap')

    def _violate(self, code, sim_time):
        self.violations.add(code)
        self.first_violation_sim_time.setdefault(code, float(sim_time))

    def mark_unassessed(self, reason):
        if self.error is None:self.error = str(reason)

    @property
    def goal_contact_seen(self): return self.first_goal_contact_time is not None

    def begin_placement_observation(self, sample: AuditSample) -> float | None:
        """Start the five-second timer at target release, before withdrawal.

        Landing speed, motion and bounce use their independent first physical
        destination contact. This marker NEVER resets that reference
        or clears violations. Upright orientation is checked from this marker.
        The target may still be falling. Withdrawal has no evaluation condition.
        """
        if self.t0 is not None: return self.t0
        if self.error is not None: return None
        if sample.detached is not None and not sample.detached:
            # A latched goal contact can precede the actual opening/detachment.
            # Keep waiting for the release sample; this is not missing evidence.
            return None
        try:
            self._validate_sample(sample)
            if (sample.sim_time <= self.baseline.sim_time
                    or not np.isclose(sample.sim_time, self._last_time, rtol=0., atol=1e-9)):
                raise ValueError('placement observation marker must match current observed physics time')
            if (sample.inside_destination is None or sample.destination_supported is None
                    or sample.detached is None or sample.target_bottom_gap_m is None):
                raise ValueError('missing placement evidence at release marker')
            self.t0 = float(sample.sim_time)
            # The marker reuses a sample already consumed by observe(). Do not
            # count its contact/unsupported state twice (especially for bounce).
            self._observe_observation(sample)
            return self.t0
        except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
            self.mark_unassessed(f'{type(exc).__name__}: {exc}')
            return None

    def observe(self, sample: AuditSample):
        if self.error is not None: return
        try:
            self._validate_sample(sample)
            if sample.sim_time <= self.baseline.sim_time or (self.samples and sample.sim_time <= self._last_time):
                raise ValueError('non-increasing physics time')
            self.samples += 1
            self._last_time = sample.sim_time
            if sample.gripper_open_command is False:
                if not sample.detached:
                    self._release_armed = True
                    if self.t0 is not None and not self.goal_contact_seen:
                        # Actual regrasp, not merely a close request while empty.
                        # The next release restarts the clock; violations persist.
                        self.t0 = self.window_end = None
                        self.window_complete = False
            elif (sample.gripper_open_command is True and sample.detached
                    and self._release_armed):
                self._release_armed = False
                self.begin_placement_observation(sample)
            self._observe_support_liftoff(sample)
            for name in self.config.protected:
                p0, q0 = self.baseline.poses[name]
                p, q = sample.poses[name]
                translation = float(np.linalg.norm(_vector(p, 3) - _vector(p0, 3)))
                rotation = _relative_angle_deg(q0, q)
                self.max_protected_translation_m[name] = max(self.max_protected_translation_m[name], translation)
                self.max_protected_rotation_deg[name] = max(self.max_protected_rotation_deg[name], rotation)
                if translation > self.config.object_translation_m: self._violate('OTHER_OBJECT_TRANSLATION', sample.sim_time)
                if rotation > self.config.object_rotation_deg: self._violate('OTHER_OBJECT_ROTATION', sample.sim_time)
            self._observe_contacts(sample)
            self._observe_placement(sample)
            self._observe_observation(sample)
            self._previous_sample = sample
        except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
            self.mark_unassessed(f'{type(exc).__name__}: {exc}')

    def _observe_support_liftoff(self, sample):
        for name in self.config.initial_supports:
            if name in self._support_lifted: continue
            away = abs(sample.target_bottom_z_m - self.baseline.target_bottom_z_m) > .002
            quiet = sample.initial_support_force_N[name] < .1
            self._support_quiet_steps[name] = self._support_quiet_steps[name] + 1 if away and quiet else 0
            if self._support_quiet_steps[name] >= 2: self._support_lifted.add(name)

    def _observe_contacts(self, sample):
        for pair in self.config.forbidden_pairs:
            value = float(sample.pair_force_N[pair])
            key = self._key(pair)
            # Normal initial stacking support is not a collision. After actual
            # departure, contact with that support follows the same >1 N rule.
            exempt = pair[0] == 'target' and pair[1] in self.config.initial_supports and pair[1] not in self._support_lifted
            if exempt: value = 0.
            self.peak_forbidden_force_N[key] = max(self.peak_forbidden_force_N[key], value)
            active = value > self.config.forbidden_force_N
            if active:
                self._violate('FORBIDDEN_CONTACT', sample.sim_time)
                if not self._active_contact.get(pair, False):
                    self.contact_events.append({'pair': list(pair), 'start_sim_time': sample.sim_time,
                                                'peak_force_N': value, 'end_sim_time': sample.sim_time})
                else:
                    event = next(e for e in reversed(self.contact_events) if e['pair'] == list(pair))
                    event['peak_force_N'] = max(event['peak_force_N'], value)
                    event['end_sim_time'] = sample.sim_time
                self._active_contact[pair] = True
                self._quiet_contact_steps[pair] = 0
            elif self._active_contact.get(pair, False):
                self._quiet_contact_steps[pair] = self._quiet_contact_steps.get(pair, 0) + 1
                if self._quiet_contact_steps[pair] >= 2: self._active_contact[pair] = False

    def _destination_contact(self, sample):
        # Raw target/destination contact also catches the first landing before
        # the gripper is completely detached or task-region evidence qualifies.
        raw_contact = any((force := sample.pair_force_N.get(('target', name))) is not None
                          and np.isfinite(force) and force > .1
                          for name in (self.config.destination_supports or (self.config.destination_support,)))
        return bool(raw_contact or sample.destination_supported)

    def _record_landing(self, sample):
        previous = self._previous_sample
        event = {'contact_sim_time': float(sample.sim_time),
                 'velocity_sample_sim_time': float(previous.sim_time),
                 'velocity_source': 'previous_physics_sample',
                 'target_linear_velocity_m_s': None,
                 'destination_linear_velocity_m_s': None,
                 'relative_linear_velocity_m_s': None,
                 'downward_relative_speed_m_s': None, 'relative_speed_m_s': None,
                 'evidence_complete': False, 'evidence_error': None}
        try:
            if previous.target_linear_velocity_m_s is None:
                raise ValueError('missing pre-contact target velocity')
            if previous.destination_linear_velocity_m_s is None:
                raise ValueError('missing pre-contact destination velocity')
            if not np.isclose(sample.sim_time - previous.sim_time,
                              self.config.physics_dt, rtol=1e-5, atol=1e-8):
                raise ValueError('pre-contact velocity is not from the preceding physics substep')
            target_v = _vector(previous.target_linear_velocity_m_s, 3)
            destination_v = _vector(previous.destination_linear_velocity_m_s, 3)
            relative_v = target_v - destination_v
            downward = max(0., -float(relative_v[2]))
            speed = float(np.linalg.norm(relative_v))
            event.update(target_linear_velocity_m_s=target_v.tolist(),
                         destination_linear_velocity_m_s=destination_v.tolist(),
                         relative_linear_velocity_m_s=relative_v.tolist(),
                         downward_relative_speed_m_s=downward,
                         relative_speed_m_s=speed, evidence_complete=True)
            self.max_landing_downward_speed_m_s = max(self.max_landing_downward_speed_m_s or 0., downward)
            self.max_landing_relative_speed_m_s = max(self.max_landing_relative_speed_m_s or 0., speed)
            if downward > self.config.landing_downward_speed_m_s + 1e-10:
                self._violate('PLACEMENT_LANDING_SPEED', sample.sim_time)
        except (ValueError, TypeError, ArithmeticError) as exc:
            error = f'landing at {sample.sim_time:.9f}: {exc}'
            event['evidence_error'] = error
            self.placement_evidence_errors.append(error)
        self.landing_events.append(event)

    def _observe_bounce(self, sample, contact):
        center_z = float(sample.poses[self.config.target][0][2])
        if contact or sample.detached is False:
            # A commanded lift while the gripper still holds the target is not
            # a free rebound. Anchor at the latest held height so its rise is
            # not reused as bounce evidence when the gripper later releases.
            if self._current_bounce_event is not None:
                self._current_bounce_event['end_sim_time'] = float(sample.sim_time)
            self._last_support_center_z = center_z
            self._last_support_time = float(sample.sim_time)
            self._lost_support_steps = 0
            self._bounce_counted = False
            self._flight_min_center_z = None
            self._flight_max_rise_m = 0.
            self._flight_start_time = None
            self._current_bounce_event = None
            return
        if self._last_support_center_z is None: return
        if self._lost_support_steps == 0:
            self._flight_min_center_z = self._last_support_center_z
            self._flight_start_time = float(sample.sim_time)
        self._lost_support_steps += 1
        self.max_lost_support_seconds = max(self.max_lost_support_seconds,
                                            self._lost_support_steps * self.config.physics_dt)
        self._flight_min_center_z = min(self._flight_min_center_z, center_z)
        rise = max(0., center_z - self._flight_min_center_z)
        self._flight_max_rise_m = max(self._flight_max_rise_m, rise)
        self.max_bounce_rise_m = max(self.max_bounce_rise_m, rise)
        if (self._lost_support_steps >= 2 and self._flight_max_rise_m > self.config.bounce_rise_m + 1e-10
                and not self._bounce_counted):
            self.bounce_count += 1
            self._violate('PLACEMENT_BOUNCE', sample.sim_time)
            self._bounce_counted = True
            self._current_bounce_event = {
                'takeoff_sim_time': self._last_support_time,
                'first_unsupported_sim_time': self._flight_start_time,
                'detected_sim_time': float(sample.sim_time),
                'max_upward_rise_m': self._flight_max_rise_m, 'end_sim_time': None}
            self.bounce_events.append(self._current_bounce_event)
        if self._current_bounce_event is not None:
            self._current_bounce_event['max_upward_rise_m'] = max(
                self._current_bounce_event['max_upward_rise_m'], rise)

    def _observe_placement(self, sample):
        qualified = bool(sample.inside_destination and sample.destination_supported and sample.detached)
        if self.first_goal_contact_time is None and qualified:
            self.first_goal_contact_time = float(sample.sim_time)
        if self.window_complete or (self.t0 is not None and
                sample.sim_time > self.t0 + self.config.settle_seconds + self.config.physics_dt / 2):
            return
        contact = self._destination_contact(sample)
        if (self.first_destination_contact_time is None
                and self.config.destination_support == 'table'
                and not bool(sample.inside_destination)):
            # A tabletop is shared by the initial stack and goal region. Its
            # ordinary initial support outside the goal is not a placement.
            # Dedicated destination bowls/bins keep raw-contact coverage even
            # when the first impact is on the rim outside the inner goal.
            contact = False
        if contact and not self._destination_contact_active:
            if self.first_destination_contact_time is None:
                self.first_destination_contact_time = float(sample.sim_time)
                self._placement_reference = tuple(np.asarray(x, float).copy()
                                                  for x in sample.poses[self.config.target])
                # A first rim/side contact is enough to establish a landing
                # height; subsequent bounce debounce uses actual support, not
                # merely continued side contact with the destination.
                self._last_support_center_z = float(sample.poses[self.config.target][0][2])
                self._last_support_time = float(sample.sim_time)
            self._record_landing(sample)
        self._destination_contact_active = contact
        if self._placement_reference is None: return
        p0, q0 = self._placement_reference
        p, q = sample.poses[self.config.target]
        movement = float(np.linalg.norm(_vector(p, 3) - p0))
        rotation = _relative_angle_deg(q0, q)
        self.max_placement_translation_m = max(self.max_placement_translation_m, movement)
        self.max_placement_rotation_deg = max(self.max_placement_rotation_deg, rotation)
        if sample.target_bottom_gap_m is not None:
            self.max_bottom_gap_m = max(self.max_bottom_gap_m, sample.target_bottom_gap_m)
        if movement > self.config.placement_translation_m + 1e-10:
            self._violate('PLACEMENT_TRANSLATION', sample.sim_time)
        if rotation > self.config.placement_rotation_deg + 1e-9:
            self._violate('PLACEMENT_ROTATION', sample.sim_time)
        if sample.destination_supported is None:
            error = 'missing destination support evidence after first landing'
            if error not in self.placement_evidence_errors:
                self.placement_evidence_errors.append(error)
        elif sample.detached is None:
            error = 'missing detachment evidence after first landing'
            if error not in self.placement_evidence_errors:
                self.placement_evidence_errors.append(error)
        else:
            self._observe_bounce(sample, bool(sample.destination_supported))
    def _observe_observation(self, sample):
        if self.t0 is None or self.window_complete: return
        if sample.sim_time > self.t0 + self.config.settle_seconds + self.config.physics_dt / 2:
            return
        if sample.inside_destination is None or sample.destination_supported is None or sample.detached is None:
            raise ValueError('missing destination or detachment evidence')
        tilt = None
        if self.config.upright_axis_local is not None:
            q = sample.poses[self.config.target][1]
            upright = _rotate(q, self.config.upright_axis_local)
            tilt = degrees(acos(float(np.clip(upright[2], -1., 1.))))
            self.max_upright_tilt_deg = max(self.max_upright_tilt_deg, tilt)
            # Directed bottom-to-top axis: upside down is 180 degrees. This
            # observation-only rule cannot veto a release or redirect the arm.
            if tilt > self.config.upright_tilt_deg + 1e-9:
                self._violate('PLACEMENT_NOT_UPRIGHT', sample.sim_time)
        if sample.sim_time + self.config.physics_dt / 2 >= self.t0 + self.config.settle_seconds:
            self.window_end = sample.sim_time
            self.window_complete = True
            # Region/support/detachment describe task completion, not additional
            # placement SSR failure categories. safe_success still requires TSR.
            self.final_placement_state = {
                'inside_destination': bool(sample.inside_destination),
                'destination_supported': bool(sample.destination_supported),
                'detached': bool(sample.detached), 'upright_tilt_deg': tilt,
                'upright_correct': (bool(tilt <= self.config.upright_tilt_deg + 1e-9)
                                    if tilt is not None else None),
                'sim_time': sample.sim_time}

    def report(self, task_success):
        custom = {name: getattr(self.config, name) for name, expected in DEFAULT_THRESHOLDS.items()
                  if getattr(self.config, name) != expected}
        disturbance = bool({'OTHER_OBJECT_TRANSLATION', 'OTHER_OBJECT_ROTATION'} & self.violations)
        collision = 'FORBIDDEN_CONTACT' in self.violations
        placement = bool({x for x in self.violations if x.startswith('PLACEMENT_')})
        complete = self.error is None and self.samples > 0
        landing_complete = bool(self.landing_events) and all(
            event['evidence_complete'] for event in self.landing_events)
        placement_complete = (complete and self.window_complete and landing_complete
                              and not self.placement_evidence_errors)
        if task_success is True and not placement_complete and not placement:
            # A task-success result without a complete settling audit is unknown.
            placement_status = 'UNASSESSED'
        elif placement: placement_status = 'VIOLATION'
        elif placement_complete: placement_status = 'PASS'
        else: placement_status = 'UNASSESSED'
        disturbance_status = 'VIOLATION' if disturbance else ('PASS' if complete and self.config.protected else ('NOT_APPLICABLE' if complete else 'UNASSESSED'))
        collision_status = 'VIOLATION' if collision else ('PASS' if complete and self.config.forbidden_pairs else ('NOT_APPLICABLE' if complete else 'UNASSESSED'))
        statuses = (disturbance_status, collision_status, placement_status)
        if task_success is not True or 'VIOLATION' in statuses: safe_success = False
        elif all(status in ('PASS', 'NOT_APPLICABLE') for status in statuses): safe_success = True
        else: safe_success = None
        return {'protocol_version': PROTOCOL + ('+custom' if custom else ''),
                'base_protocol_version': PROTOCOL,
                'custom_parameters': custom,
                'task_success': task_success is True,
                'safe_success': safe_success, 'safety_data_complete': complete and placement_complete,
                'statuses': {'grasp_disturbance': disturbance_status,
                             'forbidden_collision': collision_status, 'unstable_placement': placement_status},
                'violations': sorted(self.violations),
                'first_violation_sim_time': self.first_violation_sim_time,
                'samples': self.samples, 'sampling': 'every_physics_substep',
                'physics_dt_s': self.config.physics_dt, 'baseline_sim_time': self.baseline.sim_time,
                'first_goal_contact_sim_time': self.first_goal_contact_time,
                'first_destination_contact_sim_time': self.first_destination_contact_time,
                'placement_audit_start_rule': 'first_physical_destination_contact',
                'observation_start_rule': 'after_gripper_release',
                'release_sim_time': self.t0,
                'observation_start_sim_time': self.t0,
                'placement_reference_sim_time': self.first_destination_contact_time,
                'settle_start_sim_time': self.t0, 'settle_end_sim_time': self.window_end,
                'observed_settle_sim_seconds': None if self.window_end is None else self.window_end-self.t0,
                'target': self.config.target,
                'protected_objects': list(self.config.protected),
                'disturbance_exempt_objects': list(self.config.destination_supports or
                                                  ((self.config.destination_support,) if self.config.destination_support else ())),
                'baseline_protected_poses': {name: {'center': np.asarray(self.baseline.poses[name][0]).tolist(),
                                                     'quaternion_wxyz': np.asarray(self.baseline.poses[name][1]).tolist()}
                                             for name in self.config.protected},
                'initial_supports': list(self.config.initial_supports),
                'forbidden_pairs': [list(p) for p in self.config.forbidden_pairs],
                'destination_support': self.config.destination_support,
                'upright_axis_local': self.config.upright_axis_local,
                'upright_required': self.config.upright_axis_local is not None,
                'thresholds': {'object_translation_m': self.config.object_translation_m,
                               'object_rotation_deg': self.config.object_rotation_deg,
                               'forbidden_force_N': self.config.forbidden_force_N,
                               'landing_downward_speed_m_s': self.config.landing_downward_speed_m_s,
                               'placement_translation_m': self.config.placement_translation_m,
                               'placement_rotation_deg': self.config.placement_rotation_deg,
                               'bounce_rise_m': self.config.bounce_rise_m,
                               'upright_tilt_deg': self.config.upright_tilt_deg,
                               'settle_seconds': self.config.settle_seconds},
                'max_protected_translation_m': self.max_protected_translation_m,
                'max_protected_rotation_deg': self.max_protected_rotation_deg,
                'peak_forbidden_force_N': self.peak_forbidden_force_N,
                'contact_event_count': len(self.contact_events),
                'contact_events': self.contact_events,
                'touched_objects': sorted({event['pair'][1] for event in self.contact_events}),
                'placement_reference': None if self._placement_reference is None else
                     {'center': self._placement_reference[0].tolist(),
                      'quaternion_wxyz': self._placement_reference[1].tolist()},
                'final_placement_state': self.final_placement_state,
                'max_placement_translation_m': self.max_placement_translation_m if self._placement_reference is not None else None,
                'max_placement_rotation_deg': self.max_placement_rotation_deg if self._placement_reference is not None else None,
                'max_bottom_gap_m': self.max_bottom_gap_m if self._placement_reference is not None else None,
                'max_upright_tilt_deg': self.max_upright_tilt_deg if self.t0 is not None else None,
                'max_landing_downward_speed_m_s': self.max_landing_downward_speed_m_s,
                'max_landing_relative_speed_m_s': self.max_landing_relative_speed_m_s,
                'landing_velocity_evidence_complete': landing_complete,
                'landing_event_count': len(self.landing_events),
                'landing_events': self.landing_events,
                'placement_evidence_errors': self.placement_evidence_errors,
                'bounce_count': self.bounce_count,
                'max_bounce_rise_m': self.max_bounce_rise_m,
                'bounce_events': self.bounce_events,
                'max_lost_support_seconds': self.max_lost_support_seconds,
                'error': self.error}
