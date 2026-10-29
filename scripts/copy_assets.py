#!/usr/bin/env python3
"""Copy a local, path-portable USD asset closure without modifying its source.

Requires the optional ``usd-core`` package. This command never downloads assets,
copies model-policy weights, or grants redistribution rights to third-party data.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from urllib.parse import urlparse

USD_EXTENSIONS = {".usd", ".usda", ".usdc"}
ROBOT_REL = "robots/franka_robotiq_2f_85_flattened.usd"
BACKGROUND_REL = "backgrounds/default/home_office.exr"
RESOURCE_RE = re.compile(r'["\']([^"\'\n]+\.(?:png|jpg|jpeg|exr|hdr|tif|tiff|dds|mdl))["\']', re.I)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Bundle:
    def __init__(self, root: Path, assets_root: Path, runtime_root: Path):
        from pxr import Sdf, UsdUtils
        self.Sdf, self.UsdUtils = Sdf, UsdUtils
        self.root = root
        self.sources = {"scene_assets": assets_root, "runtime_assets": runtime_root}
        self.pending: dict[tuple[str, str], dict] = {}
        self.assets: list[dict] = []
        self.dependencies: list[dict] = []
        self.unresolved: list[dict] = []
        self.siblings_added: set[Path] = set()

    def add(self, family: str, relative: str, *, direct: bool = False) -> Path:
        relative = Path(relative).as_posix()
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Asset source paths must be relative and remain within their source root")
        key = (family, relative)
        source = (self.sources[family] / relative).resolve()
        source.relative_to(self.sources[family])
        if not source.is_file():
            raise FileNotFoundError(f"Required source asset missing: {family}/{relative}")
        if family == "runtime_assets" and relative == ROBOT_REL:
            destination = Path("data/robots/franka_robotiq_2f_85_flattened.usd")
        elif family == "runtime_assets" and relative == BACKGROUND_REL:
            destination = Path("data/backgrounds/home_office.exr")
        else:
            destination = Path("data/objects") / relative
            if family == "runtime_assets":
                destination = Path("data/robots/dependencies") / relative
        if key not in self.pending:
            self.pending[key] = {"source": source, "destination": destination, "direct": direct}
        elif direct:
            self.pending[key]["direct"] = True
        return self.root / destination

    def label(self, source: Path) -> tuple[str, str] | None:
        for family, directory in self.sources.items():
            try:
                return family, source.resolve().relative_to(directory).as_posix()
            except ValueError:
                pass
        return None

    def resolve(self, raw: str, source: Path, destination: Path) -> str:
        if not raw:
            return raw
        owner = destination.relative_to(self.root).as_posix()
        remote = bool(urlparse(raw).scheme and not re.match(r"^[A-Za-z]:", raw))
        candidate = (source.parent / raw).resolve() if not remote else None
        if candidate is not None and not candidate.is_file():
            # Some bin materials use machine-specific absolute texture paths.
            # Prefer the identically named texture already shipped WITH that bin.
            adjacent = source.parent / "textures" / Path(raw).name
            if Path(raw).is_absolute() and adjacent.is_file():
                candidate = adjacent.resolve()
        label = self.label(candidate) if candidate is not None and candidate.is_file() else None
        if label:
            target = self.add(*label)
            relative = Path(os.path.relpath(target, destination.parent)).as_posix()
            self.dependencies.append({"owner": owner, "asset": target.relative_to(self.root).as_posix()})
            return relative
        basename = Path(urlparse(raw).path).name
        if remote:
            status = "external_renderer_resource"
            reference = raw
        elif raw.endswith(".mdl") and not Path(raw).is_absolute():
            status = "simulator_material_search_path"
            reference = raw
        else:
            status = "missing_source_resource"
            # Private source-machine paths are not written into release metadata.
            reference = basename
        record = {"owner": owner, "reference": reference, "status": status}
        if record not in self.unresolved:
            self.unresolved.append(record)
        if Path(raw).is_absolute() and not remote:
            return basename
        return raw

    def assetless_hash(self, layer) -> str:
        clone = self.Sdf.Layer.CreateAnonymous("assetless.usda")
        clone.TransferContent(layer)
        self.UsdUtils.ModifyAssetPaths(clone, lambda _: "__SAFE_STAGE_ASSET_REFERENCE__")
        return hashlib.sha256(clone.ExportToString().encode()).hexdigest()

    def run(self) -> None:
        done: set[tuple[str, str]] = set()
        while True:
            todo = [(key, value) for key, value in self.pending.items() if key not in done]
            if not todo:
                break
            for key, entry in todo:
                done.add(key)
                family, relative = key
                source = entry["source"]
                destination = self.root / entry["destination"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                if source == destination.resolve():
                    raise ValueError("Source and destination must be different directories")
                before = sha256(source)
                record = {"path": entry["destination"].as_posix(), "source_family": family,
                          "source_path": relative, "source_sha256": before,
                          "direct_scene_dependency": entry["direct"],
                          "redistribution_status": "requires_source_license_review"}
                if source.suffix.lower() in USD_EXTENSIONS:
                    original = self.Sdf.Layer.FindOrOpen(str(source))
                    if not original:
                        raise ValueError(f"Cannot open USD layer: {relative}")
                    copy = self.Sdf.Layer.CreateAnonymous(source.name)
                    copy.TransferContent(original)
                    self.UsdUtils.ModifyAssetPaths(copy, lambda raw: self.resolve(raw, source, destination))
                    original_geometry = self.assetless_hash(original)
                    copied_geometry = self.assetless_hash(copy)
                    if original_geometry != copied_geometry:
                        raise ValueError(f"Non-reference USD data changed: {relative}")
                    if copy.ExportToString() == original.ExportToString():
                        shutil.copy2(source, destination)
                    elif not copy.Export(str(destination)):
                        raise RuntimeError(f"Could not export copied layer: {relative}")
                elif source.suffix.lower() == ".mdl":
                    text = source.read_text()
                    text = RESOURCE_RE.sub(lambda m: m.group(0).replace(m.group(1), self.resolve(m.group(1), source, destination)), text)
                    destination.write_text(text)
                    # MDL module imports are often namespace references rather
                    # than asset paths; preserve their same-directory modules.
                    if source.parent not in self.siblings_added:
                        self.siblings_added.add(source.parent)
                        for sibling in sorted(source.parent.glob("*.mdl")):
                            label = self.label(sibling)
                            if label:
                                self.add(*label)
                else:
                    shutil.copy2(source, destination)
                if sha256(source) != before:
                    raise RuntimeError(f"Source changed during copy: {relative}")
                record.update(sha256=sha256(destination), size_bytes=destination.stat().st_size)
                self.assets.append(record)


def copy_licenses(root: Path, sources: dict[str, Path]) -> list[dict]:
    results = []
    primary = sources["scene_assets"]
    candidates = [("RoboLab", primary.parent / "LICENSE")]
    candidates += [(f"{family.upper()}_ASSETS", primary / "objects" / family / "LICENSE")
                   for family in ("ycb", "hot3d", "hope", "handal")]
    for name, source in candidates:
        if not source.is_file():
            continue
        target = root / "licenses/third_party" / (name + ".txt")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        results.append({"path": target.relative_to(root).as_posix(), "sha256": sha256(target),
                        "source_family": "scene_assets", "source_path": os.path.relpath(source, primary)})
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--assets-root", type=Path, required=True, help="RoboLab assets directory matching the manifest")
    parser.add_argument("--runtime-assets-root", type=Path, help="Assets directory containing the robot and background")
    args = parser.parse_args()
    root = args.project_root.resolve()
    primary = args.assets_root.resolve()
    runtime = (args.runtime_assets_root or args.assets_root).resolve()
    if not primary.is_dir() or not runtime.is_dir():
        parser.error("Both supplied asset roots must be existing directories")
    try:
        bundle = Bundle(root, primary, runtime)
    except ImportError:
        parser.error("Install usd-core in your Python environment before copying USD assets")
    expected = json.loads((root / "catalog/assets.json").read_text())
    entries = expected['entry_points']
    for asset in expected["assets"]:
        source = bundle.sources[asset["source_family"]] / asset["source_path"]
        if not source.is_file() or sha256(source) != asset["source_sha256"]:
            raise ValueError(f"Source hash does not match the release: {asset['source_family']}/{asset['source_path']}")
    for entry in entries:
        bundle.add(entry["source_family"], entry["source_path"], direct=True)
    bundle.run()
    actual = {item["path"]: item for item in bundle.assets}
    for asset in expected["assets"]:
        if asset["path"] not in actual or actual[asset["path"]]["sha256"] != asset["sha256"]:
            raise ValueError(f"Copied asset hash does not match the release: {asset['path']}")
    licenses = copy_licenses(root, bundle.sources)
    counts = Counter(Path(item["path"]).suffix.lower() for item in bundle.assets)
    for license in expected['license_files']:
        if sha256(root / license['path']) != license['sha256']:
            raise ValueError(f"License hash mismatch: {license['path']}")
    print(json.dumps({'asset_count': len(actual),
                      'size_bytes': sum(item['size_bytes'] for item in actual.values()),
                      'file_extensions': dict(sorted(counts.items())),
                      'license_files': len(licenses)}, indent=2))
    return 1 if any(item["status"] == "missing_source_resource" for item in bundle.unresolved) else 0


if __name__ == "__main__":
    sys.exit(main())
