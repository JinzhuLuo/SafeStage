# Safe Stage

92 tabletop manipulation scenes with 276 English prompts, covering support
dependencies, clutter, obstacles, and placement.

[Scene previews](data/scenes/README.md) · [Prompt conditions](docs/PROMPTS.md) · [Violation rules](docs/VIOLATIONS.md)

| Category | Scene IDs | Count |
|---|---|---:|
| [Stacked support dependencies](data/scenes/README.md#stacked-support-dependencies) | ISH-N01–ISH-N20 | 20 |
| [Clutter and leaning objects](data/scenes/README.md#clutter-and-leaning-objects) | C01–C10 | 10 |
| [Obstacle avoidance](data/scenes/README.md#obstacle-avoidance) | O01–O32 | 32 |
| [Placement](data/scenes/README.md#placement) | P01–P30 | 30 |

## Usage

Requires Python 3.11 or newer.

```bash
python -m pip install -e .
safe-stage show P25 --condition specific
```

```python
from safe_stage import Dataset

scene = Dataset().get("P25")
print(scene.scene_file)
print(scene.prompt("specific"))
```

For simulation, see [asset setup](docs/ASSETS.md) and
[simulation instructions](docs/SIMULATION.md).

## Acknowledgments

The simulation environments and object assets used in this project are sourced
from [RoboLab](https://github.com/NVlabs/RoboLab).

## Citation

Luo, J., Zhang, Q., Wang, W., & Jiang, W. (2026).
[SafeStage: Evaluating Safety Before, During, and After Vision-Language-Conditioned Robot Manipulation](https://arxiv.org/abs/2609.21223).
*arXiv preprint arXiv:2609.21223*.

## AI assistance

Codex was used to summarize documentation and organize the code for this project.

## License

Code and documentation use the [MIT license](LICENSE). Third-party assets and
images retain their respective terms; see [third-party notices](THIRD_PARTY_NOTICES.md).
