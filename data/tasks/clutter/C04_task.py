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
    success = TerminationTermCfg(func=object_on_top, params={"object": 'mustard_bottle',
        "reference_object": "plate_large", "require_gripper_detached": True})

@dataclass
class ClutterC04Task(Task):
    contact_object_list = ('mustard_bottle', 'tomato_sauce_can', 'canned_peaches', 'cream_cheese', 'jello', 'plate_large', 'bin_b03', 'table')
    scene = import_scene(str(Path(__file__).resolve().parents[2] / 'scenes' / 'clutter' / 'C04.usda'))
    terminations = Terminations
    instruction: str = 'Move the mustard bottle to the large plate.'
    episode_length_s: int = 90
    attributes = ["safe_stage", "clutter"]
    subtasks = None
