"""Record a scene under gravity using an Isaac Sim Python environment.

Select an available GPU with CUDA_VISIBLE_DEVICES before running:

    python examples/simulate_scene.py P25 --headless --steps 240
"""

from __future__ import annotations

import argparse

from safe_stage import Dataset
from safe_stage.simulation import run_gravity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_id")
    parser.add_argument("--root")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dt", type=float, default=1.0 / 120.0)
    parser.add_argument("--output", help="Write every measured PhysX body pose, dt, gravity, and raw displacement observations to JSON")
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("--steps must be nonnegative")
    result = run_gravity(Dataset(args.root), args.scene_id, steps=args.steps,
                         dt=args.dt, headless=args.headless,
                         device=args.device, output=args.output)
    print(f"Loaded {result['scene_id']}; recorded {result['body_count']} bodies over "
          f"{result['recorded_steps']} physics steps. Numerical validity: {result['numerically_valid']}")
    if args.output:
        print("Recording:", args.output)


if __name__ == "__main__":
    main()
