"""Dynamic clutter/leaning candidate; no object constraints or policy intervention."""
from dataclasses import dataclass
from pathlib import Path
import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils import configclass
from robolab.core.scenes.utils import import_scene
from robolab.core.task.task import Task
from robolab.core.task.conditionals import object_on_top

@configclass
class Terminations:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    success = TerminationTermCfg(func=object_on_top, params={"object": 'mug',
        "reference_object": "plate_large", "require_gripper_detached": True})

@dataclass
class ClutterC03Task(Task):
    contact_object_list = ('mug', 'parmesan_cheese_canister', 'coffee_can', 'spam_can', 'remote_control', 'plate_large', 'bin_a05', 'table')
    scene = import_scene(str(Path(__file__).resolve().parents[2] / 'scenes' / 'clutter' / 'C03.usda'))
    terminations = Terminations
    instruction: str = 'Move the red YCB mug to the large plate.'
    episode_length_s: int = 90
    attributes = ["safe_stage", "clutter"]
    subtasks = None
