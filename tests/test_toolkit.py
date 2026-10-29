"""Behavioral checks for portable scene access and optional dependencies.

Run with ``python -m unittest discover -s tests -v`` after installation, or with
``PYTHONPATH=src`` in a checkout. No simulator, GPU, or third-party assets needed.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from safe_stage import Dataset, SafeStageError, check_dataset
from safe_stage.cli import main, usd_info
from safe_stage.validation import GENERIC_SUFFIX
from safe_stage.simulation import GravityRecorder


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def fixture(root: Path) -> None:
    rows = []
    prompts = {condition: {} for condition in ("default", "generic", "specific")}
    for scene_id, category in (("ISH-N01", "ish"), ("P25", "placement")):
        row = {
            "scene_id": scene_id, "category": category,
            "scene_file": f"data/scenes/{category}/{scene_id}.usda",
            "task_file": f"data/tasks/{category}/{scene_id}_task.py",
            "spec_file": f"data/specs/{category}.json",
            "preview_file": f"media/previews/{scene_id}.jpg",
        }
        for key in ("scene_file", "task_file", "preview_file"):
            path = root / row[key]
            path.parent.mkdir(parents=True, exist_ok=True)
            content = {
                "scene_file": '#usda 1.0\n(\n subLayers = [@../../objects/cube.usda@]\n)\n',
                "task_file": 'raise RuntimeError("This file must not be executed by file checks")\n',
                "preview_file": "fixture preview",
            }[key]
            path.write_text(content, encoding="utf-8")
        dump(root / row["spec_file"], [{"scene_id": scene_id, "objects": []}])
        goal = f"Move the target in {scene_id} onto the plate."
        prompts["default"][scene_id] = goal
        prompts["generic"][scene_id] = goal + " " + GENERIC_SUFFIX
        prompts["specific"][scene_id] = goal + " Relocate the supported bottle first."
        rows.append(row)
    dump(root / "catalog/scenes.json", {
        "schema_version": 1, "scene_count": 2,
        "category_counts": {"ish": 1, "placement": 1}, "scenes": rows,
    })
    for condition, prompts_map in prompts.items():
        dump(root / f"prompts/{condition}.json", prompts_map)
    asset = root / "data/objects/cube.usda"
    asset.parent.mkdir(parents=True, exist_ok=True)
    asset.write_text('#usda 1.0\ndef Xform "Cube" {}\n', encoding="utf-8")
    dump(root / "catalog/assets.json", {"schema_version": 1, "assets": [{
        "path": asset.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
        "size_bytes": asset.stat().st_size,
    }]})
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and "objects" not in path.parts:
            files.append({"path": path.relative_to(root).as_posix(),
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "size_bytes": path.stat().st_size})
    dump(root / "catalog/files.json", {"schema_version": 1, "files": files})


class ToolkitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "original checkout"
        fixture(self.root)

    def test_copy_remains_usable_after_original_is_removed(self) -> None:
        relocated = Path(self.temporary.name) / "independent-copy"
        shutil.copytree(self.root, relocated)
        shutil.rmtree(self.root)
        data = Dataset(relocated)
        self.assertEqual(data.get("p25").scene_file.parent.parent.parent.parent, relocated)
        self.assertEqual(data.get("P25").spec()["scene_id"], "P25")
        self.assertTrue(check_dataset(data, assets=True, hashes=True).ok)
        with patch.dict(os.environ, {"SAFE_STAGE_HOME": str(relocated)}):
            self.assertEqual(Dataset().root, relocated)

    def test_core_only_install_does_not_require_optional_assets(self) -> None:
        shutil.rmtree(self.root / "data/objects")
        data = Dataset(self.root)
        self.assertTrue(check_dataset(data).ok)
        report = check_dataset(data, assets=True)
        self.assertFalse(report.ok)
        self.assertIn("missing_asset", {issue.code for issue in report.issues})

    def test_wrong_explicit_root_does_not_fall_back_to_environment(self) -> None:
        with patch.dict(os.environ, {"SAFE_STAGE_HOME": str(self.root)}):
            with self.assertRaises(SafeStageError):
                Dataset(self.root / "absent")

    def test_prompt_conditions_preserve_goals_and_uniform_generic_reminder(self) -> None:
        dataset = Dataset(self.root)
        self.assertEqual(len(dataset.scenes("grasp")), 1)
        self.assertEqual(dataset.get("P25").prompt("generic"),
                         dataset.get("P25").prompt() + " " + GENERIC_SUFFIX)
        values = dataset.prompts("specific")
        values["P25"] = "Move a different object elsewhere."
        dump(self.root / "prompts/specific.json", values)
        report = check_dataset(Dataset(self.root))
        self.assertFalse(report.ok)
        self.assertIn("prompt_goal", {issue.code for issue in report.issues})

    def test_missing_or_extra_prompt_ids_are_errors(self) -> None:
        mapping = Dataset(self.root).prompts("default")
        mapping["NONEXISTENT"] = mapping.pop("P25")
        dump(self.root / "prompts/default.json", mapping)
        report = check_dataset(Dataset(self.root))
        self.assertIn("prompt_coverage", {issue.code for issue in report.issues})

    def test_count_and_duplicate_scene_checks(self) -> None:
        catalog = json.loads((self.root / "catalog/scenes.json").read_text())
        catalog["scene_count"] = 92
        dump(self.root / "catalog/scenes.json", catalog)
        self.assertIn("scene_count", {issue.code for issue in check_dataset(Dataset(self.root)).issues})
        catalog["scenes"].append(catalog["scenes"][0])
        dump(self.root / "catalog/scenes.json", catalog)
        with self.assertRaises(SafeStageError):
            Dataset(self.root)

    def test_paths_cannot_escape_to_original_workspace(self) -> None:
        dataset = Dataset(self.root)
        for path in ("/tmp/object.usd", "../object.usd", "data/../../../object.usd", "C:/object.usd", "data\\object.usd"):
            with self.subTest(path=path), self.assertRaises(SafeStageError):
                dataset.path(path)
        (self.root / "outside").symlink_to(self.root.parent, target_is_directory=True)
        with self.assertRaises(SafeStageError):
            dataset.path("outside/object.usd")

    def test_asset_tampering_is_detected_by_hash_verification(self) -> None:
        asset = self.root / "data/objects/cube.usda"
        content = asset.read_text()
        asset.write_text(content.replace("Cube", "Fake"))
        report = check_dataset(Dataset(self.root), assets=True, hashes=True)
        self.assertFalse(report.ok)
        self.assertIn("hash_mismatch", {issue.code for issue in report.issues})

    def test_task_syntax_is_checked_without_executing_tasks(self) -> None:
        dataset = Dataset(self.root)
        self.assertTrue(check_dataset(dataset).ok)
        dataset.get("P25").task_file.write_text("def broken(:\n")
        report = check_dataset(dataset)
        self.assertIn("task_syntax", {issue.code for issue in report.issues})

    def test_cli_supports_root_before_or_after_subcommand_and_errors_cleanly(self) -> None:
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            self.assertEqual(main(["--root", str(self.root), "list", "--json"]), 0)
        self.assertEqual(len(json.loads(stream.getvalue())), 2)
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            self.assertEqual(main(["prompt", "P25", "--root", str(self.root), "--condition", "specific"]), 0)
        self.assertIn("Relocate the supported bottle first.", stream.getvalue())
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["show", "UNKNOWN", "--root", str(self.root)]), 2)

    def test_import_and_core_checks_do_not_load_gpu_or_numeric_modules(self) -> None:
        program = (
            "import sys; from safe_stage import Dataset, check_dataset; import safe_stage.simulation; "
            "assert check_dataset(Dataset()).ok; "
            "assert not set(['torch','numpy','pxr','isaacsim','omni']).intersection(sys.modules)"
        )
        environment = {**os.environ, "SAFE_STAGE_HOME": str(self.root)}
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
        result = subprocess.run([sys.executable, "-c", program], env=environment,
                                text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_usd_renderer_exceptions_require_exact_manifest_declaration(self) -> None:
        manifest_path = self.root / "catalog/assets.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["unresolved_dependencies"] = [
            {"reference": "https://example.invalid/Appearance.mdl", "status": "external_renderer_resource", "owner": "data/objects/cube.usda"},
            {"reference": "gltf/pbr.mdl", "status": "simulator_material_search_path", "owner": "data/objects/cube.usda"},
        ]
        dump(manifest_path, manifest)
        dependencies = ["https:/example.invalid/Appearance.mdl", "gltf/pbr.mdl"]
        fake = SimpleNamespace(
            Usd=SimpleNamespace(Stage=SimpleNamespace(Open=lambda _: SimpleNamespace(Traverse=lambda: [object()]))),
            UsdUtils=SimpleNamespace(ComputeAllDependencies=lambda _: ([], [], dependencies)),
        )
        with patch.dict(sys.modules, {"pxr": fake}):
            result = usd_info(Dataset(self.root), "P25")
            self.assertTrue(result["ok"])
            self.assertEqual(len(result["renderer_warnings"]), 2)
            dependencies.extend(["https:/example.invalid/missing.png", "https:/example.invalid/Unlisted.mdl"])
            result = usd_info(Dataset(self.root), "P25")
            self.assertFalse(result["ok"])
            self.assertEqual(len(result["unresolved_assets"]), 2)

    def test_physics_recording_uses_live_buffer_copies_and_wxyz(self) -> None:
        recorder = GravityRecorder(["/world/target"], 1 / 120)
        pose = [[1., 2., 3., 0., 0., 0., 1.]]  # PhysX XYZW
        velocity = [[0., 0., 0., 0., 0., 0.]]
        recorder.record(0, pose, velocity, simulation_time=0.)
        pose[0][0] += 0.125
        pose[0][6] = -1.  # Same rotation, opposite quaternion sign.
        recorder.record(1, pose, velocity, simulation_time=1 / 120)
        self.assertEqual(recorder.samples[0]["positions_world_m"], [[1., 2., 3.]])
        self.assertEqual(recorder.samples[0]["quaternions_world_wxyz"], [[1., 0., 0., 0.]])
        observations = recorder.observations()[0]
        self.assertAlmostEqual(observations["max_observed_translation_from_initial_m"], 0.125)
        self.assertEqual(observations["max_observed_rotation_from_initial_deg"], 0.)

    def test_invalid_physics_samples_are_preserved_and_raise(self) -> None:
        recorder = GravityRecorder(["/world/target"], 1 / 120)
        with self.assertRaises(SafeStageError):
            recorder.record(0, [[float("nan"), 0., 0., 0., 0., 0., 1.]], [[0.] * 6], simulation_time=0.)
        self.assertFalse(recorder.samples[-1]["numerically_valid"])
        self.assertIsNone(recorder.samples[-1]["positions_world_m"][0][0])
        json.dumps(recorder.samples, allow_nan=False)  # Strict valid JSON.
        with self.assertRaises(SafeStageError):
            recorder.record(1, [[0.] * 7], [[0.] * 6], simulation_time=1 / 120)
        self.assertFalse(recorder.samples[-1]["numerically_valid"])


if __name__ == "__main__":
    unittest.main()
