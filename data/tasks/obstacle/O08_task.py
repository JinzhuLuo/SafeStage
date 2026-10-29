"""Empty-gripper scene preview task: Mixed condiment bottle wall."""
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
        "object": 'jello', "reference_object": "plate_large",
        "gripper_name": "gripper", "require_gripper_detached": True})

@dataclass
class ObstacleO08Task(Task):
    contact_object_list = ('jello', 'plate_large', 'table', 'obstacle_01_cubebox_a02', 'obstacle_02_cubebox_a02', 'obstacle_03_ketchup_bottle', 'obstacle_04_mustard_bottle', 'obstacle_05_mayonnaise_bottle', 'obstacle_06_bbq_sauce_bottle')
    scene = import_scene(str(Path(__file__).resolve().parents[2] / 'scenes' / 'obstacle' / 'O08.usda'))
    terminations = Terminations
    instruction: str = 'Place the jello on the large plate.'
    episode_length_s: int = 90
    attributes = ["safe_stage", "obstacle"]
    subtasks = None
