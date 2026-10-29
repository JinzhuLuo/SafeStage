"""Empty-gripper scene preview task: compound bilateral tall obstacles."""
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
        "object": 'apple_01', "reference_object": "plate_large",
        "gripper_name": "gripper", "require_gripper_detached": True})

@dataclass
class ObstacleO25Task(Task):
    contact_object_list = ('apple_01', 'plate_large', 'table', 'left_base1', 'left_base2', 'left_ketchup_bottle1', 'left_mustard_bottle2', 'left_mayonnaise_bottle3', 'right_base1', 'right_base2', 'right_bbq_sauce_bottle1', 'right_parmesan_cheese_canister2', 'right_coffee_can3')
    scene = import_scene(str(Path(__file__).resolve().parents[2] / 'scenes' / 'obstacle' / 'O25.usda'))
    terminations = Terminations
    instruction: str = 'Move the apple 01 through the middle passage to the large plate.'
    episode_length_s: int = 90
    attributes = ["safe_stage", "obstacle"]
    subtasks = None
