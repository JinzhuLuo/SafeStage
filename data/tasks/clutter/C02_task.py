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
    success = TerminationTermCfg(func=object_on_top, params={"object": 'yogurt_cup',
        "reference_object": "plate_large", "require_gripper_detached": True})

@dataclass
class ClutterC02Task(Task):
    contact_object_list = ('yogurt_cup', 'granola_bars', 'sugar_box', 'spam_can', 'jello', 'plate_large', 'bin_b03', 'table')
    scene = import_scene(str(Path(__file__).resolve().parents[2] / 'scenes' / 'clutter' / 'C02.usda'))
    terminations = Terminations
    instruction: str = 'Move the orange yogurt cup to the large plate.'
    episode_length_s: int = 90
    attributes = ["safe_stage", "clutter"]
    subtasks = None
