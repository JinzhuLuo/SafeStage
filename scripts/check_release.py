#!/usr/bin/env python3
"""Check the English public release without loading a simulator."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from safe_stage import Dataset
from safe_stage.validation import check_dataset

DATA_FOLDERS = ['catalog', 'prompts', 'data/scenes', 'data/tasks', 'data/specs', 'media', 'docs']
FROZEN_EXCLUDES = {'catalog/files.json'}
PRIVATE_PATH = re.compile(r'/(?:shared|home)/(?:user\d+|rdagli)(?:/|\b)')
FORBIDDEN_NAMES = {'runtime', 'models', 'checkpoints', '.cache', '.compat'}
FORBIDDEN_SUFFIXES = {'.pt', '.pth', '.ckpt', '.safetensors'}
CJK = re.compile(r'[\u3400-\u9fff\uf900-\ufaff]')


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def data_files(root):
    for folder in DATA_FOLDERS:
        for path in sorted((root / folder).rglob('*')):
            relative = path.relative_to(root).as_posix()
            if path.is_file() and not path.is_symlink() and '__pycache__' not in path.parts and relative not in FROZEN_EXCLUDES:
                yield path


def freeze(root):
    manifest = dict(schema_version=1, scope='portable core data included in source and wheel releases',
        files=[dict(path=p.relative_to(root).as_posix(), sha256=sha256(p), size_bytes=p.stat().st_size)
               for p in data_files(root)])
    (root / 'catalog/files.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return len(manifest['files'])


def public_files(root):
    if (root / '.git').exists():
        raw = subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=root)
        return sorted({root / name.decode() for name in raw.split(b'\0') if name
                       and (root / name.decode()).is_file()})
    folders = [*DATA_FOLDERS, 'src', 'scripts', 'tests', 'docs', 'examples', 'licenses', '.github']
    result = {p for folder in folders for p in (root / folder).rglob('*')
              if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    result.update(p for p in root.iterdir() if p.is_file() and p.suffix in {'.md', '.toml', '.cff', '.in', '.py'})
    return sorted(result)


def check(root):
    errors = []
    dataset = Dataset(root)
    report = check_dataset(dataset, hashes=True)
    if not report.ok:
        errors.extend(f'{i.code}: {i.path}: {i.message}' for i in report.issues if i.severity == 'error')
    if len(dataset) != 92 or len(dataset.scenes('grasp')) != 30:
        errors.append('Canonical 92-scene / 30-grasp counts differ')
    if dataset.catalog.get('violation_protocol') != 'SAFE-STAGE-SSR-v1':
        errors.append('Missing active Safe Stage violation protocol')
    files = public_files(root)
    scanned = 0
    for path in files:
        if not path.exists():
            errors.append(f'Missing public file: {path.relative_to(root)}'); continue
        relative = path.relative_to(root)
        if path.is_symlink():
            errors.append(f'Public symlink: {relative}')
        if FORBIDDEN_NAMES.intersection(relative.parts) or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f'Model/cache/runtime file in public tree: {relative}')
        if relative.parts[:2] in [('data', 'objects'), ('data', 'robots'), ('data', 'backgrounds')]:
            errors.append(f'Optional external asset entered public files: {relative}')
        if path.stat().st_size > 50 * 1024 * 1024:
            errors.append(f'Oversized public file: {relative}')
        if path.suffix.lower() in {'.json', '.csv', '.md', '.py', '.toml', '.cff', '.yml', '.yaml', '.usda', '.html', '.in'}:
            text = path.read_text(errors='replace'); scanned += 1
            if PRIVATE_PATH.search(text):
                errors.append(f'Original machine path in public file: {relative}')
            if CJK.search(text):
                errors.append(f'Non-English content in public file: {relative}')
    # No symlink is allowed even in the optional local bundle.
    symlinks = [p.relative_to(root).as_posix() for folder in ['data', 'media']
                for p in (root / folder).rglob('*') if p.is_symlink()]
    if symlinks:
        errors.append('Local data symlinks found: ' + ', '.join(symlinks[:10]))
    return dict(ok=not errors, scene_count=len(dataset), prompt_count=sum(len(dataset.prompts(c)) for c in ['default', 'generic', 'specific']),
        public_files_checked=len(files), public_text_files_scanned=scanned,
        symlinks_in_local_data=len(symlinks), models_or_runtime_bundled=False if not errors else None,
        physics_rerun=False, errors=errors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--freeze', action='store_true', help='Regenerate portable core data checksums before checking')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.freeze:
        freeze(args.root)
    result = check(args.root)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['ok'] else 1)
