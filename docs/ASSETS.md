# Asset setup

Scene geometry, textures, robot assets, and backgrounds are separate from
this repository. Use authorized local asset directories matching
[`catalog/assets.json`](../catalog/assets.json).

From the project directory, install USD support and copy the assets:

```bash
python -m pip install -e '.[usd]'
python scripts/copy_assets.py \
  --assets-root /path/to/RoboLab/assets \
  --runtime-assets-root /path/to/robot-and-background/assets
```

The runtime assets directory must contain
`robots/franka_robotiq_2f_85_flattened.usd` and
`backgrounds/default/home_office.exr`. The command verifies matching files,
copies them into `data/`, and leaves the supplied directories unchanged.

Some materials need renderer-provided MDL modules or remote NVIDIA resources.
They may appear with default materials when those resources are unavailable.

Asset licenses vary. Review the [third-party notices](../THIRD_PARTY_NOTICES.md)
before using or redistributing the assets or derived images; the project's
software license does not cover them.
