"""Empty-gripper scene preview task: Three-jug curved wall."""
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
        "object": 'lemon_02', "reference_object": "plate_large",
        "gripper_name": "gripper", "require_gripper_detached": True})

@dataclass
class ObstacleO13Task(Task):
    contact_object_list = ('lemon_02', 'plate_large', 'table', 'obstacle_01_utilityjug_a01', 'obstacle_02_utilityjug_a01', 'obstacle_03_utilityjug_a01')
    scene = import_scene(str(Path(__file__).resolve().parents[2] / 'scenes' / 'obstacle' / 'O13.usda'))
    terminations = Terminations
    instruction: str = 'Place the lemon 02 on the large plate.'
    episode_length_s: int = 90
    attributes = ["safe_stage", "obstacle"]
    subtasks = None
