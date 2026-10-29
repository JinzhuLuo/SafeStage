"""Evaluate normalized, complete physics evidence without changing the trace."""
from __future__ import annotations

from dataclasses import fields
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ..catalog import Dataset, SafeStageError
from .ssr import AuditConfig, AuditSample, PROTOCOL, SSRAudit


def _names(value, label):
    if not isinstance(value, (list, tuple)) or any(not isinstance(x, str) or not x for x in value):
        raise SafeStageError(f'{label} must be a list of nonempty object names')
    return tuple(value)


def _config(payload, protocol):
    if not isinstance(payload, dict):
        raise SafeStageError('Trace config must be an object')
    allowed = {field.name for field in fields(AuditConfig)}
    unknown = set(payload) - allowed
    if unknown:
        raise SafeStageError('Unknown audit configuration fields: ' + ', '.join(sorted(unknown)))
    for key in ['target', 'protected', 'destination_support', 'forbidden_pairs', 'physics_dt']:
        if key not in payload:
            raise SafeStageError(f'Missing explicit audit scope: config.{key}')
    result = {**protocol['thresholds'], **payload}
    for key in ['target', 'destination_support']:
        if not isinstance(result[key], str) or not result[key]:
            raise SafeStageError(f'config.{key} must identify the task instance')
    result['protected'] = _names(result['protected'], 'config.protected')
    for key in ['initial_supports', 'destination_supports']:
        if key in result:
            result[key] = _names(result[key], 'config.' + key)
    pairs = result['forbidden_pairs']
    if not isinstance(pairs, list):
        raise SafeStageError('config.forbidden_pairs must explicitly list monitored pairs')
    result['forbidden_pairs'] = tuple(_names(pair, 'forbidden pair') for pair in pairs)
    if any(len(pair) != 2 or pair[0] == pair[1] for pair in result['forbidden_pairs']):
        raise SafeStageError('Each forbidden pair must contain two distinct names')
    if result.get('upright_axis_local') is not None:
        if not isinstance(result['upright_axis_local'], (list, tuple)):
            raise SafeStageError('upright_axis_local must be a three-component unit vector')
        result['upright_axis_local'] = tuple(result['upright_axis_local'])
    for key in [*protocol['thresholds'], 'physics_dt']:
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise SafeStageError(f'config.{key} must be finite and positive')
    try:
        return AuditConfig(**result)
    except (ValueError, TypeError) as exc:
        raise SafeStageError(f'Invalid task scope: {exc}') from exc


def _sample(row):
    if not isinstance(row, dict):
        raise SafeStageError('Every physics sample must be an object')
    for key in ['sim_time', 'poses', 'pair_force_N', 'initial_support_force_N',
                'inside_destination', 'destination_supported', 'detached',
                'target_bottom_gap_m', 'target_bottom_z_m']:
        if key not in row:
            raise SafeStageError(f'Missing physics field: {key}; use null for unavailable evidence')
    for key in ['inside_destination', 'destination_supported', 'detached', 'gripper_open_command']:
        if row.get(key) is not None and not isinstance(row[key], bool):
            raise SafeStageError(f'{key} must be a boolean or null')
    if not isinstance(row['poses'], dict):
        raise SafeStageError('poses must map object names to center/quaternion observations')
    poses = {}
    for name, pose in row['poses'].items():
        if not isinstance(pose, dict) or 'center_m' not in pose or 'quaternion_wxyz' not in pose:
            raise SafeStageError('Every pose requires center_m and quaternion_wxyz')
        try:
            poses[name] = (np.asarray(pose['center_m'], dtype=float),
                           np.asarray(pose['quaternion_wxyz'], dtype=float))
        except (TypeError, ValueError) as exc:
            raise SafeStageError(f'Invalid pose array: {name}') from exc
    if not isinstance(row['pair_force_N'], list):
        raise SafeStageError('pair_force_N must be a list of {first, second, force_N} records')
    forces = {}
    for record in row['pair_force_N']:
        if not isinstance(record, dict) or any(k not in record for k in ['first', 'second', 'force_N']):
            raise SafeStageError('Every contact record requires first, second, force_N')
        pair = record['first'], record['second']
        if not all(isinstance(name, str) and name for name in pair) or pair in forces:
            raise SafeStageError('Contact pairs must be named and unique per sample')
        forces[pair] = record['force_N']
    if not isinstance(row['initial_support_force_N'], dict):
        raise SafeStageError('initial_support_force_N must be an object')
    velocities = {}
    for key in ['target_linear_velocity_m_s', 'destination_linear_velocity_m_s']:
        value = row.get(key)
        velocities[key] = None if value is None else np.asarray(value, dtype=float)
    return AuditSample(row['sim_time'], poses, forces, row['initial_support_force_N'],
                       row['inside_destination'], row['destination_supported'], row['detached'],
                       row['target_bottom_gap_m'], row['target_bottom_z_m'],
                       **velocities, gripper_open_command=row.get('gripper_open_command'))


def evaluate_trace(payload: dict[str, Any], dataset: Dataset | None = None) -> dict[str, Any]:
    """Score explicitly scoped per-physics-step observations, never a summary.

    This does not infer contact forces from motion or fill missing velocities.
    Gaps and missing required observations cannot produce a safe-pass result.
    """
    dataset = dataset or Dataset()
    protocol = dataset.violation_protocol()
    if protocol.get('protocol_version') != PROTOCOL:
        raise SafeStageError(f'This evaluator implements {PROTOCOL}, not the selected rule version')
    if not isinstance(payload, dict) or payload.get('schema_version') != 1:
        raise SafeStageError('Normalized trace must use schema_version 1')
    if payload.get('protocol_version', PROTOCOL) != PROTOCOL:
        raise SafeStageError(f'Trace must declare protocol_version {PROTOCOL}')
    if type(payload.get('task_success')) is not bool:
        raise SafeStageError('task_success must be an explicit independent boolean task result')
    if not isinstance(payload.get('samples'), list):
        raise SafeStageError('Trace samples must be a chronological list of physical substeps')
    for key in ['contact_pairs_not_applicable', 'synthetic_example']:
        if key in payload and type(payload[key]) is not bool:
            raise SafeStageError(f'{key} must be an explicit boolean')
    if payload.get('scene_id') is not None and not isinstance(payload['scene_id'], str):
        raise SafeStageError('scene_id must be a string or null')
    if payload.get('task_scope') is not None and (not isinstance(payload['task_scope'], str) or not payload['task_scope'].strip()):
        raise SafeStageError('task_scope must be a nonempty description when provided')
    scene = dataset.get(payload['scene_id']) if payload.get('scene_id') is not None else None
    config = _config(payload.get('config'), protocol)
    if scene is not None:
        objects = {row['instance_name'] for row in scene.spec()['objects']}
        supports = set(config.destination_supports or (config.destination_support,))
        configured = {config.target, *config.protected, *supports, *config.initial_supports}
        if config.target not in objects or configured - (objects | {'table'}):
            raise SafeStageError('Audit scope names objects absent from the selected scene')
        expected_protected = objects - {config.target, *supports, 'table'}
        if set(config.protected) != expected_protected:
            raise SafeStageError('All scene objects other than the current target, destination support(s), and fixed table must be protected')
        if config.target != scene.metadata['target_object'] and not payload.get('task_scope'):
            raise SafeStageError('A relocation target requires an explicit task_scope describing the subtask')
    try:
        baseline = _sample(payload.get('baseline'))
        audit = SSRAudit(config, baseline)
    except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
        raise SafeStageError(f'Invalid initial stable baseline: {exc}') from exc
    errors = []
    # A manipulation rollout needs its monitored robot/object contact pairs.
    # A deliberately empty pair set must be explicitly justified by the adapter.
    if not config.forbidden_pairs and not payload.get('contact_pairs_not_applicable', False):
        errors.append('No forbidden contact pairs were configured for this manipulation trace')
    previous_time = float(baseline.sim_time)
    for index, row in enumerate(payload['samples']):
        try:
            sample = _sample(row)
        except (SafeStageError, ValueError, TypeError) as exc:
            errors.append(f'Invalid sample {index}: {exc}')
            continue
        try:
            delta = float(sample.sim_time) - previous_time
            if not math.isclose(delta, config.physics_dt, rel_tol=1e-5, abs_tol=1e-8):
                errors.append(f'Sample {index} is not the next physics substep')
            previous_time = float(sample.sim_time)
        except (TypeError, ValueError):
            errors.append(f'Sample {index} has an invalid physics timestamp')
        audit.observe(sample)
    # Preserve observed violations, even when a separate gap invalidates a pass.
    if errors:
        audit.mark_unassessed('; '.join(errors[:12]))
    report = audit.report(payload['task_success'])
    report.update(schema_version=1, scene_id=payload.get('scene_id'),
                  task_scope=payload.get('task_scope'),
                  input_format='safe-stage-normalized-physics-trace-v1',
                  input_physics_samples=len(payload['samples']), trace_evidence_errors=errors,
                  rules_file='catalog/violation_protocol.json',
                  rules_sha256=hashlib.sha256(dataset.path('catalog/violation_protocol.json').read_bytes()).hexdigest(),
                  is_violation=True if report['violations'] else False if report['safety_data_complete'] else None,
                  source_evidence_modified=False,
                  synthetic_example=bool(payload.get('synthetic_example', False)))
    return report


def evaluate_trace_file(path: str | Path, dataset: Dataset | None = None) -> dict[str, Any]:
    path = Path(path)
    data = path.read_bytes()
    try:
        payload = json.loads(data)
    except (ValueError, UnicodeError) as exc:
        raise SafeStageError(f'Invalid normalized trace JSON: {exc}') from exc
    report = evaluate_trace(payload, dataset)
    report['input_sha256'] = hashlib.sha256(data).hexdigest()
    return report
