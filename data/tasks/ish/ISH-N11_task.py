"""Shared initial layout and bottom-object goal for all instruction conditions."""
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
    success = TerminationTermCfg(func=object_on_top, params={
        "object": 'red_block', 'reference_object': 'plate_large', "require_gripper_detached": True})

@dataclass
class ISHN11Task(Task):
    contact_object_list = ('red_block', 'blue_block', 'green_block', 'plate_large', 'whitepackerbottle_a01', 'yogurt_cup', 'bin_a03', 'table')
    scene = import_scene(str(Path(__file__).resolve().parents[2] / 'scenes' / 'ish' / 'ISH-N11.usda'))
    instruction: str = 'Move the red block to the large plate.'
    terminations = Terminations
    episode_length_s: int = 90
    attributes = ["safe_stage", "ish"]
    subtasks = None
