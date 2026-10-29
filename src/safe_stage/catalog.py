"""Catalog and prompt access without importing a simulator."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping

CONDITIONS = ("default", "generic", "specific")
CATEGORIES = ("ish", "clutter", "obstacle", "placement")


class SafeStageError(ValueError):
    """An absent or malformed Safe Stage data file."""


def find_data_root(root: str | Path | None = None) -> Path:
    """Find an explicit root, SAFE_STAGE_HOME, a checkout, or installed data.

    ``root`` and ``SAFE_STAGE_HOME`` always take precedence. A wrong explicit
    root is an error, rather than silently falling back to another dataset.
    """
    supplied = root if root is not None else os.environ.get("SAFE_STAGE_HOME")
    if supplied is not None:
        candidate = Path(supplied).expanduser().resolve()
        if not (candidate / "catalog/scenes.json").is_file():
            raise SafeStageError(f"No catalog/scenes.json under {candidate}")
        return candidate
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "catalog/scenes.json").is_file():
            return candidate
    resource = resources.files("safe_stage").joinpath("_data")
    try:
        candidate = Path(os.fspath(resource)).resolve()
    except TypeError as exc:
        raise SafeStageError(
            "Data in a zipped import must be extracted; set SAFE_STAGE_HOME "
            "to the extracted Safe Stage directory."
        ) from exc
    if (candidate / "catalog/scenes.json").is_file():
        return candidate
    raise SafeStageError(
        "Safe Stage data was not found. Install a wheel containing the core "
        "data, or use --root PATH / SAFE_STAGE_HOME."
    )


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SafeStageError(f"Cannot read JSON {path}: {exc}") from exc


@dataclass(frozen=True)
class Scene:
    """One catalog row, bound to the dataset that provides its files."""

    dataset: "Dataset"
    metadata: Mapping[str, Any]

    @property
    def scene_id(self) -> str:
        return self.metadata["scene_id"]

    @property
    def category(self) -> str:
        return self.metadata["category"]

    @property
    def scene_file(self) -> Path:
        return self.dataset.path(self.metadata["scene_file"])

    @property
    def task_file(self) -> Path:
        return self.dataset.path(self.metadata["task_file"])

    @property
    def preview_file(self) -> Path | None:
        value = self.metadata.get("preview_file")
        return self.dataset.path(value) if value else None

    def prompt(self, condition: str = "default") -> str:
        return self.dataset.prompt(self.scene_id, condition)

    def spec(self) -> Mapping[str, Any]:
        """Return this scene's physical specification from its category specification."""
        value = self.metadata.get("spec_file")
        if not value:
            raise SafeStageError(f"{self.scene_id} has no spec_file")
        payload = read_json(self.dataset.path(value))
        rows = payload.get("scenes", payload) if isinstance(payload, dict) else payload
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict) and row.get("scene_id") == self.scene_id:
                    return row
        if isinstance(rows, dict):
            if rows.get("scene_id") == self.scene_id:
                return rows
            if self.scene_id in rows and isinstance(rows[self.scene_id], dict):
                return rows[self.scene_id]
        raise SafeStageError(f"{self.scene_id} missing from {value}")

    def to_dict(self, condition: str = "default") -> dict[str, Any]:
        return {**self.metadata, "condition": condition, "prompt": self.prompt(condition)}


class Dataset:
    """Read a Safe Stage checkout or installed core dataset.

    Paths in the public catalog must be relative and remain inside the selected
    root. External assets are optional, but a selected root never falls back to
    another checkout when files are missing.
    """

    def __init__(self, root: str | Path | None = None):
        self.root = find_data_root(root)
        self.catalog = read_json(self.root / "catalog/scenes.json")
        if not isinstance(self.catalog, dict) or self.catalog.get("schema_version") != 1:
            raise SafeStageError("catalog/scenes.json must use schema_version 1")
        rows = self.catalog.get("scenes")
        if not isinstance(rows, list):
            raise SafeStageError("catalog/scenes.json must contain a scenes list")
        self._scenes: dict[str, Scene] = {}
        for row in rows:
            if not isinstance(row, dict):
                raise SafeStageError("Every scene entry must be an object")
            scene_id, category = row.get("scene_id"), row.get("category")
            if not isinstance(scene_id, str) or not scene_id:
                raise SafeStageError("Every scene needs a nonempty scene_id")
            if scene_id in self._scenes:
                raise SafeStageError(f"Duplicate scene_id: {scene_id}")
            if category not in CATEGORIES:
                raise SafeStageError(f"Unknown category for {scene_id}: {category}")
            self._scenes[scene_id] = Scene(self, dict(row))
        self._prompts: dict[str, dict[str, str]] = {}

    def __len__(self) -> int:
        return len(self._scenes)

    def __iter__(self) -> Iterator[Scene]:
        return iter(self._scenes.values())

    def path(self, relative: str) -> Path:
        """Resolve a portable data path, rejecting absolute paths and escapes."""
        if not isinstance(relative, str) or not relative or "\\" in relative:
            raise SafeStageError(f"Expected a nonempty POSIX relative path: {relative!r}")
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or ":" in pure.parts[0]:
            raise SafeStageError(f"Unsafe data path: {relative!r}")
        result = (self.root / Path(*pure.parts)).resolve()
        if not result.is_relative_to(self.root):
            raise SafeStageError(f"Data path leaves the selected root: {relative!r}")
        return result

    def get(self, scene_id: str) -> Scene:
        try:
            return self._scenes[scene_id.upper()]
        except KeyError as exc:
            raise SafeStageError(f"Unknown scene {scene_id!r}; use safe-stage list") from exc

    def scenes(self, category: str | None = None) -> list[Scene]:
        if category is None:
            return list(self)
        selected = ("ish", "clutter") if category == "grasp" else (category,)
        if any(item not in CATEGORIES for item in selected):
            raise SafeStageError(f"Unknown category: {category}")
        return [scene for scene in self if scene.category in selected]

    def prompts(self, condition: str = "default") -> dict[str, str]:
        if condition not in CONDITIONS:
            raise SafeStageError(f"Condition must be one of {', '.join(CONDITIONS)}")
        if condition not in self._prompts:
            payload = read_json(self.path(f"prompts/{condition}.json"))
            if not isinstance(payload, dict) or not all(
                isinstance(key, str) and isinstance(value, str) and value.strip()
                for key, value in payload.items()
            ):
                raise SafeStageError(f"prompts/{condition}.json must map IDs to nonempty strings")
            self._prompts[condition] = payload
        return dict(self._prompts[condition])

    def prompt(self, scene_id: str, condition: str = "default") -> str:
        scene = self.get(scene_id)
        try:
            return self.prompts(condition)[scene.scene_id]
        except KeyError as exc:
            raise SafeStageError(f"Missing {condition} prompt for {scene.scene_id}") from exc

    def violation_protocol(self) -> Mapping[str, Any]:
        """Read the versioned rule contract without loading the NumPy evaluator."""
        path = self.catalog.get('violation_protocol_file', 'catalog/violation_protocol.json')
        value = read_json(self.path(path))
        if not isinstance(value, dict) or value.get('schema_version') != 1:
            raise SafeStageError('Violation protocol must use schema_version 1')
        declared = self.catalog.get('violation_protocol')
        if declared is not None and value.get('protocol_version') != declared:
            raise SafeStageError('Catalog and violation protocol version differ')
        return value
