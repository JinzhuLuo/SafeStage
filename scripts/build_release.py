#!/usr/bin/env python3
"""Build a core-only source ZIP; optional third-party data never enters it."""
from pathlib import Path
import argparse
import json
import zipfile

from check_release import ROOT, check, public_files
from safe_stage import __version__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / f'artifacts/safe-stage-{__version__}-source.zip')
    args = parser.parse_args()
    result = check(ROOT)
    if not result['ok']:
        raise SystemExit(json.dumps(result, indent=2))
    files = public_files(ROOT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, 'safe-stage/' + path.relative_to(ROOT).as_posix())
    print(json.dumps(dict(output=str(args.output), files=len(files),
                         size_bytes=args.output.stat().st_size, optional_asset_bundle_included=False,
                         model_weights_included=False), indent=2))


if __name__ == '__main__':
    main()
