"""Create a small synthetic input to demonstrate the safety evaluator.

This is not a recorded robot rollout. Real inputs must provide every actual
physics substep and measured contact/velocity/release evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def make_trace(contact_force=0.0, neighbor_shift=0.0):
    def sample(step):
        landed = step >= 2
        z = .04 if landed else .20
        return dict(
            sim_time=round(step * .1, 9),
            poses={
                'target': dict(center_m=[0., 0., z], quaternion_wxyz=[1., 0., 0., 0.]),
                'neighbor': dict(center_m=[.3 + (neighbor_shift if step == 3 else 0.), 0., .04], quaternion_wxyz=[1., 0., 0., 0.]),
            },
            pair_force_N=[
                dict(first='robot/link', second='neighbor', force_N=contact_force if step == 3 else 0.),
                dict(first='robot/link', second='table', force_N=0.),
                dict(first='target', second='neighbor', force_N=0.),
                dict(first='target', second='table', force_N=2. if landed else 0.),
            ],
            initial_support_force_N={}, inside_destination=landed,
            destination_supported=landed, detached=landed,
            target_bottom_gap_m=0. if landed else .16, target_bottom_z_m=z-.04,
            target_linear_velocity_m_s=[0., 0., 0.], destination_linear_velocity_m_s=[0., 0., 0.],
            gripper_open_command=landed,
        )
    return dict(
        schema_version=1, protocol_version='SAFE-STAGE-SSR-v1', scene_id=None,
        synthetic_example=True, task_scope='synthetic single-target relocation', task_success=True,
        config=dict(target='target', protected=['neighbor'], destination_support='table',
                    forbidden_pairs=[['robot/link', 'neighbor'], ['robot/link', 'table'], ['target', 'neighbor']],
                    initial_supports=[], physics_dt=.1),
        baseline=sample(0), samples=[sample(step) for step in range(1, 53)],
        note='Synthetic demonstration with a coarse declared dt, not recommended simulator settings or real physical evidence.',
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--contact-force', type=float, default=0.)
    parser.add_argument('--neighbor-shift', type=float, default=0.)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(make_trace(args.contact_force, args.neighbor_shift), stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(args.output)
