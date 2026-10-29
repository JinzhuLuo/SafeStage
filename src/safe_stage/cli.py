"""The ``safe-stage`` command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import webbrowser

from .catalog import CATEGORIES, CONDITIONS, Dataset, SafeStageError
from .validation import check_dataset, known_renderer_dependencies


def _json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _root_argument(parser: argparse.ArgumentParser, *, inherited: bool = False) -> None:
    parser.add_argument("--root", type=Path, default=argparse.SUPPRESS if inherited else None,
                        help="Safe Stage data root (or set SAFE_STAGE_HOME)")


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(prog="safe-stage", description="Browse and inspect Safe Stage scenes without loading a policy model.")
    _root_argument(command)
    commands = command.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="List scene IDs and goals")
    listing.add_argument("--category", choices=(*CATEGORIES, "grasp"))
    listing.add_argument("--json", action="store_true")
    show = commands.add_parser("show", help="Show scene metadata and one instruction condition")
    show.add_argument("scene_id")
    show.add_argument("--condition", choices=CONDITIONS, default="default")
    show.add_argument("--json", action="store_true")
    prompt = commands.add_parser("prompt", help="Print the exact model instruction")
    prompt.add_argument("scene_id")
    prompt.add_argument("--condition", choices=CONDITIONS, default="default")
    check = commands.add_parser("check", help="Validate core files; optional large bundles are opt-in")
    check.add_argument("--assets", action="store_true", help="Check copied third-party asset bundle and direct USD references")
    check.add_argument("--hashes", action="store_true", help="Verify frozen core hashes; include asset hashes with --assets")
    check.add_argument("--json", action="store_true")
    check.add_argument("--report", type=Path, help="Also write the JSON report to this file")
    preview = commands.add_parser("preview", help="Open an existing scene preview image")
    preview.add_argument("scene_id")
    preview.add_argument("--path", action="store_true", help="Print its path without launching a browser")
    usd = commands.add_parser("usd-info", help="Compose one scene with the optional usd-core package")
    usd.add_argument("scene_id")
    usd.add_argument("--json", action="store_true")
    simulate = commands.add_parser("simulate", help="Optional Isaac Sim gravity smoke test with measured poses")
    simulate.add_argument("scene_id")
    simulate.add_argument("--steps", type=int, default=240)
    simulate.add_argument("--dt", type=float, default=1.0 / 120.0)
    simulate.add_argument("--headless", action="store_true")
    simulate.add_argument("--device", default="cuda:0")
    simulate.add_argument("--output", type=Path, help="Write JSON with gravity, dt, and measured per-body poses")
    protocol = commands.add_parser('protocol', help='Show the current versioned safety violation rules')
    protocol.add_argument('--json', action='store_true')
    evaluate = commands.add_parser('evaluate', help='Evaluate a normalized per-physics-step trace; never rewrite its evidence')
    evaluate.add_argument('trace', type=Path)
    evaluate.add_argument('--output', type=Path, required=True, help='New report path (existing files are never overwritten)')
    for item in (listing, show, prompt, check, preview, usd, simulate, protocol, evaluate):
        _root_argument(item, inherited=True)
    return command


def usd_info(dataset: Dataset, scene_id: str) -> dict:
    """Optional USD composition; this does not advance physics or launch CUDA."""
    try:
        from pxr import Usd, UsdUtils
    except ImportError as exc:
        raise SafeStageError("USD inspection requires the optional dependency: pip install 'safe-stage[usd]'") from exc
    scene = dataset.get(scene_id)
    try:
        stage = Usd.Stage.Open(str(scene.scene_file))
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(scene.scene_file))
    except Exception as exc:
        raise SafeStageError(f"USD could not inspect {scene.scene_id}: {exc}") from exc
    if stage is None:
        raise SafeStageError(f"USD could not compose {scene.scene_id}")
    # Accept only exact MDL references declared as renderer resources. All
    # unlisted URLs and missing local geometry/textures remain errors.
    known_renderer = known_renderer_dependencies(dataset)
    # UsdUtils normalizes URI strings using Sdf's asset-path normalization,
    # which can collapse the scheme's // into /. Match that exact alternate
    # spelling only for references explicitly listed in the asset manifest.
    known_renderer.update({reference.replace("://", ":/", 1): record
                           for reference, record in list(known_renderer.items())
                           if "://" in reference})
    runtime_materials = [str(item) for item in unresolved if str(item) in known_renderer]
    unresolved = [str(item) for item in unresolved if str(item) not in runtime_materials]
    return {
        "scene_id": scene.scene_id, "scene_file": str(scene.scene_file),
        "prim_count": sum(1 for _ in stage.Traverse()), "layer_count": len(layers),
        "asset_count": len(assets), "unresolved_assets": unresolved,
        "renderer_material_modules": runtime_materials,
        "renderer_warnings": [known_renderer[item] for item in runtime_materials],
        "physics_executed": False, "ok": not unresolved,
    }


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        dataset = Dataset(args.root)
        if args.command == "list":
            scenes = dataset.scenes(args.category)
            if args.json:
                _json([scene.to_dict() for scene in scenes])
            else:
                for scene in scenes:
                    print(f"{scene.scene_id:<9} {scene.category:<10} {scene.prompt()}")
                print(f"{len(scenes)} scenes")
        elif args.command == "show":
            scene = dataset.get(args.scene_id)
            if args.json:
                _json(scene.to_dict(args.condition))
            else:
                print(f"{scene.scene_id} ({scene.category})")
                print(f"Scene: {scene.scene_file}")
                print(f"Task: {scene.task_file}")
                print(f"Condition: {args.condition}")
                print(scene.prompt(args.condition))
        elif args.command == "prompt":
            print(dataset.prompt(args.scene_id, args.condition))
        elif args.command == "check":
            report = check_dataset(dataset, assets=args.assets, hashes=args.hashes)
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
            if args.json:
                _json(report.to_dict())
            else:
                print(f"{'PASS' if report.ok else 'FAIL'}: {report.scene_count} scenes; "
                      f"{report.checked_files} files checked; {report.hashes_verified} hashes verified")
                for issue in report.issues:
                    print(f"{issue.severity}: {issue.code}: {issue.path}: {issue.message}")
                print("File checks do not rerun physics or trajectory validation.")
            return 0 if report.ok else 1
        elif args.command == "preview":
            path = dataset.get(args.scene_id).preview_file
            if path is None or not path.is_file():
                raise SafeStageError(f"Preview is missing for {args.scene_id}")
            print(path)
            if not args.path:
                webbrowser.open(path.as_uri())
        elif args.command == "usd-info":
            result = usd_info(dataset, args.scene_id)
            if args.json:
                _json(result)
            else:
                print(f"{result['scene_id']}: {result['prim_count']} prims, {result['layer_count']} layers")
                for item in result["unresolved_assets"]:
                    print(f"Unresolved: {item}")
                print("USD composition only; physics was not executed.")
            return 0 if result["ok"] else 1
        elif args.command == "simulate":
            from .simulation import run_gravity
            result = run_gravity(dataset, args.scene_id, steps=args.steps, dt=args.dt,
                                 headless=args.headless, device=args.device, output=args.output)
            print(f"{result['scene_id']}: {result['recorded_steps']} physics steps; "
                  f"{result['body_count']} bodies; numerical validity {result['numerically_valid']}")
            if args.output:
                print(f"Recording: {args.output}")
            print("Gravity smoke test only; no reference trajectory or safety assessment.")
        elif args.command == 'protocol':
            _json(dataset.violation_protocol())
        elif args.command == 'evaluate':
            if args.output.resolve() == args.trace.resolve() or args.output.exists():
                raise SafeStageError('Choose a new output file; original evidence and existing reports are preserved')
            try:
                from .evaluation import evaluate_trace_file
            except ImportError as exc:
                raise SafeStageError("Evaluation needs the optional NumPy extra: pip install 'safe-stage[evaluation]'") from exc
            result = evaluate_trace_file(args.trace, dataset)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('x', encoding='utf-8') as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write('\n')
            _json({key: result[key] for key in ['protocol_version', 'task_success', 'safe_success',
                                               'is_violation', 'statuses', 'violations']})
            print(f'Report: {args.output}')
        return 0
    except (SafeStageError, OSError) as exc:
        print(f"safe-stage: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
