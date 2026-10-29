# Simulation

Reading scenes and prompts requires Python 3.11+ with no additional packages.
Inspecting USD layers requires `usd-core`; running dynamics requires a
compatible Isaac Sim environment and the [scene assets](ASSETS.md).
The optional adapters in `data/tasks/` also require RoboLab and Isaac Lab.

From the project directory, use the Python interpreter supplied by Isaac Sim
and select an available GPU:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m safe_stage simulate P25 \
  --headless --steps 240 --output /tmp/p25-gravity.json
```

This records body poses and velocities under gravity. It does not command a
robot or perform grasp, placement, or safety evaluation. Safety evaluation
requires a task rollout with the measurements described in
[the violation rules](VIOLATIONS.md).
