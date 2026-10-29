"""Optional gravity smoke test using live PhysX poses, without a policy model.

Importing this module is lightweight. Simulator imports occur only when
``run_gravity`` is called from an existing Isaac Sim Python environment.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .catalog import Dataset, SafeStageError


def _host_list(value: Any) -> list:
    """Copy NumPy, Torch, or Warp data into independent JSON-compatible lists."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy") and not hasattr(value, "tolist"):
        value = value.numpy()
    return value.tolist() if hasattr(value, "tolist") else [list(row) for row in value]


class GravityRecorder:
    """Store measured poses; no method sets a rigid body's state.

    Physics tensor transforms use XYZW; the public recording stores WXYZ.
    NaN/Inf samples are retained as nulls with explicit validity metadata, and
    immediately raise an error instead of silently becoming valid observations.
    """

    def __init__(self, body_paths: list[str], dt: float):
        if not body_paths or len(set(body_paths)) != len(body_paths):
            raise SafeStageError("A recording requires unique nonempty rigid-body paths")
        if not math.isfinite(dt) or dt <= 0:
            raise SafeStageError("Physics dt must be finite and positive")
        self.body_paths = list(body_paths)
        self.dt = float(dt)
        self.samples: list[dict[str, Any]] = []

    def record(self, step: int, transforms_xyzw: Any, velocities: Any, *, simulation_time: float) -> None:
        transforms = _host_list(transforms_xyzw)
        velocities = _host_list(velocities)
        count = len(self.body_paths)
        if len(transforms) != count or any(len(row) != 7 for row in transforms):
            raise SafeStageError("PhysX pose array does not match the selected rigid bodies")
        if len(velocities) != count or any(len(row) != 6 for row in velocities):
            raise SafeStageError("PhysX velocity array does not match the selected rigid bodies")
        finite = math.isfinite(simulation_time) and all(
            math.isfinite(float(value)) for row in transforms + velocities for value in row
        )
        valid_quaternions = finite and all(sum(float(value) ** 2 for value in row[3:]) > 1e-12 for row in transforms)
        def clean(values: list) -> list:
            return [[float(value) if math.isfinite(float(value)) else None for value in row] for row in values]
        positions = clean([row[:3] for row in transforms])
        quaternions = clean([[row[6], *row[3:6]] for row in transforms])
        self.samples.append({
            "step": int(step), "elapsed_seconds": step * self.dt,
            "simulation_time_seconds": simulation_time if math.isfinite(simulation_time) else None,
            "positions_world_m": positions, "quaternions_world_wxyz": quaternions,
            "velocities_world_linear_angular": clean(velocities),
            "numerically_valid": bool(valid_quaternions),
        })
        if not valid_quaternions:
            raise SafeStageError(f"Nonfinite rigid-body state or invalid quaternion at physics step {step}")

    def observations(self) -> list[dict[str, Any]]:
        valid = [sample for sample in self.samples if sample["numerically_valid"]]
        if not valid:
            return []
        initial, final = valid[0], valid[-1]
        rows = []
        for index, body_path in enumerate(self.body_paths):
            p0, q0 = initial["positions_world_m"][index], initial["quaternions_world_wxyz"][index]
            translations, rotations = [], []
            for sample in valid:
                p, q = sample["positions_world_m"][index], sample["quaternions_world_wxyz"][index]
                translations.append(math.sqrt(sum((a - b) ** 2 for a, b in zip(p, p0))))
                similarity = abs(sum(a * b for a, b in zip(q, q0))) / math.sqrt(sum(a * a for a in q) * sum(b * b for b in q0))
                rotations.append(math.degrees(2 * math.acos(min(1.0, max(0.0, similarity)))))
            rows.append({
                "body_path": body_path,
                "final_translation_from_initial_m": translations[-1],
                "max_observed_translation_from_initial_m": max(translations),
                "final_rotation_from_initial_deg": rotations[-1],
                "max_observed_rotation_from_initial_deg": max(rotations),
                "final_position_world_m": final["positions_world_m"][index],
                "final_quaternion_world_wxyz": final["quaternions_world_wxyz"][index],
            })
        return rows


def run_gravity(
    dataset: Dataset, scene_id: str, *, steps: int = 240, dt: float = 1.0 / 120.0,
    headless: bool = True, device: str = "cuda:0", output: str | Path | None = None,
) -> dict[str, Any]:
    """Load a scene and record live PhysX poses at every requested physics step.

    The first sample is taken after simulator initialization/warm-start. No pose
    resets, robot commands, reference trajectory, or safety rules are applied.
    The JSON output records actual gravity, dt, and per-body state changes.
    """
    if not isinstance(steps, int) or steps < 0:
        raise SafeStageError("steps must be a nonnegative integer")
    if not math.isfinite(dt) or dt <= 0:
        raise SafeStageError("dt must be finite and positive")
    scene = dataset.get(scene_id)
    try:
        from isaacsim import SimulationApp
    except ImportError as exc:
        raise SafeStageError("Run simulation with the Python environment shipped with Isaac Sim; see docs/SIMULATION.md.") from exc
    target = Path(output).expanduser().resolve() if output else None
    app = SimulationApp({"headless": headless})
    recorder = None
    report: dict[str, Any] = {
        "schema_version": 1, "scene_id": scene.scene_id, "mode": "gravity_smoke_test",
        "status": "initializing", "requested_steps": steps, "requested_dt_seconds": dt,
        "device": device, "body_pose_source": "live_physx_rigid_body_view",
        "quaternion_order": "wxyz", "coordinate_frame": "world",
        "reference_trajectory_replayed": False, "robot_commands_applied": False,
        "object_poses_reset_by_example": False, "safety_assessment": "not_evaluated",
        "note": "Gravity recording only; no task or safety evaluation.",
    }
    try:
        # These imports must follow SimulationApp initialization.
        import omni.usd
        from isaacsim.core.api import SimulationContext
        from pxr import UsdPhysics

        context = omni.usd.get_context()
        if not context.open_stage(str(scene.scene_file)):
            raise SafeStageError(f"Could not open {scene.scene_file}")
        for _ in range(10):
            app.update()
        stage = context.get_stage()
        body_paths = sorted(
            str(prim.GetPath()) for prim in stage.Traverse()
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
            and UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get() is not False
        )
        if not body_paths:
            raise SafeStageError("The composed scene contains no enabled rigid bodies")
        simulation = SimulationContext(
            physics_dt=dt, rendering_dt=1.0 / 60.0,
            stage_units_in_meters=1.0, device=device, backend="torch",
        )
        simulation.initialize_physics()
        simulation.play()
        simulation_view = simulation.physics_sim_view
        if simulation_view is None:
            raise SafeStageError("Isaac Sim did not initialize its PhysX simulation view")
        simulation_view.set_subspace_roots("/")
        view = simulation_view.create_rigid_body_view(body_paths)
        actual_paths = list(view.prim_paths)
        if set(actual_paths) != set(body_paths):
            raise SafeStageError("The PhysX body view does not cover every enabled authored body")
        actual_dt = float(simulation.get_physics_dt())
        gravity_direction, gravity_magnitude = simulation.get_physics_context().get_gravity()
        report.update({
            "physics_dt_seconds": actual_dt,
            "gravity_direction": [float(value) for value in gravity_direction],
            "gravity_magnitude_m_s2": float(gravity_magnitude),
            "body_paths": actual_paths,
            "initial_sample_timing": "after_simulator_initialization_and_warm_start",
        })
        recorder = GravityRecorder(actual_paths, actual_dt)
        recorder.record(0, view.get_transforms(), view.get_velocities(), simulation_time=float(simulation.current_time))
        for step in range(1, steps + 1):
            # render=True can advance multiple physics steps when rendering_dt
            # differs from physics_dt. Explicit stepping preserves sample timing.
            simulation.step(render=False)
            recorder.record(step, view.get_transforms(), view.get_velocities(), simulation_time=float(simulation.current_time))
            if not headless and step % 2 == 0:
                simulation.render()
        report["status"] = "completed"
        # Capture data before stop(), which can invalidate physics tensor views.
        simulation.stop()
    except Exception as exc:
        report["status"] = "numerical_error" if recorder and recorder.samples and not recorder.samples[-1]["numerically_valid"] else "runtime_error"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        try:
            if recorder is not None:
                report.update({
                    "sample_count": len(recorder.samples),
                    "recorded_steps": recorder.samples[-1]["step"] if recorder.samples else 0,
                    "body_count": len(recorder.body_paths),
                    "numerically_valid": bool(recorder.samples) and all(sample["numerically_valid"] for sample in recorder.samples),
                    "body_observations": recorder.observations(), "samples": recorder.samples,
                })
            if target:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        finally:
            app.close()
    return report
