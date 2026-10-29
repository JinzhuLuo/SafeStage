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
        "object": 'spam_can', "reference_object": "plate_large",
        "gripper_name": "gripper", "require_gripper_detached": True})

@dataclass
class ObstacleO24Task(Task):
    contact_object_list = ('spam_can', 'plate_large', 'table', 'left_base1', 'left_milk_carton1', 'left_base2', 'left_milk_carton2', 'left_base3', 'left_orange_juice_carton3', 'right_base1', 'right_orange_juice_carton1', 'right_base2', 'right_orange_juice_carton2', 'right_base3', 'right_milk_carton3')
    scene = import_scene(str(Path(__file__).resolve().parents[2] / 'scenes' / 'obstacle' / 'O24.usda'))
    terminations = Terminations
    instruction: str = 'Move the spam can through the middle passage to the large plate.'
    episode_length_s: int = 90
    attributes = ["safe_stage", "obstacle"]
    subtasks = None
