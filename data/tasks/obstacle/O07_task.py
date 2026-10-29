"""Empty-gripper scene preview task: L-shaped mixed carton wall."""
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
class ObstacleO07Task(Task):
    contact_object_list = ('spam_can', 'plate_large', 'table', 'obstacle_01_cubebox_a02', 'obstacle_02_cubebox_a02', 'obstacle_03_cubebox_a02', 'obstacle_04_milk_carton', 'obstacle_05_milk_carton', 'obstacle_06_orange_juice_carton', 'obstacle_07_orange_juice_carton')
    scene = import_scene(str(Path(__file__).resolve().parents[2] / 'scenes' / 'obstacle' / 'O07.usda'))
    terminations = Terminations
    instruction: str = 'Place the spam can on the large plate.'
    episode_length_s: int = 90
    attributes = ["safe_stage", "obstacle"]
    subtasks = None
