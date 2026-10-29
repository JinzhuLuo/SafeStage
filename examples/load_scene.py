"""Read a scene and exact instruction without a simulator or policy model.

    python examples/load_scene.py P25 --condition specific
    python examples/load_scene.py O31 --root /path/to/safe-stage
"""

import argparse
import json

from safe_stage import CONDITIONS, Dataset

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("scene_id", nargs="?", default="ISH-N16")
parser.add_argument("--root")
parser.add_argument("--condition", choices=CONDITIONS, default="default")
args = parser.parse_args()

dataset = Dataset(args.root)
scene = dataset.get(args.scene_id)
print(json.dumps(scene.to_dict(args.condition), ensure_ascii=False, indent=2))
print("USD:", scene.scene_file)
print("Instruction:", scene.prompt(args.condition))
