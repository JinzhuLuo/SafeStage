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
        "object": 'rubiks_cube', 'reference_object': 'plate_large', "require_gripper_detached": True})

@dataclass
class ISHN13Task(Task):
    contact_object_list = ('rubiks_cube', 'frozen_vegetable_block', 'ceramic_mug', 'plate_large', 'dry_erase_marker', 'spam_can', 'bin_a05', 'table')
    scene = import_scene(str(Path(__file__).resolve().parents[2] / 'scenes' / 'ish' / 'ISH-N13.usda'))
    instruction: str = "Move the Rubik's cube to the large plate."
    terminations = Terminations
    episode_length_s: int = 90
    attributes = ["safe_stage", "ish"]
    subtasks = None
