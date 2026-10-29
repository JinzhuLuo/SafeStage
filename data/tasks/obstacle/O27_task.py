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
        "object": 'mug', "reference_object": "plate_large",
        "gripper_name": "gripper", "require_gripper_detached": True})

@dataclass
class ObstacleO27Task(Task):
    contact_object_list = ('mug', 'plate_large', 'table', 'left_plasticjerrican_a031', 'left_base1', 'left_coffee_pot1', 'right_closedheadpail_a011', 'right_base1', 'right_pitcher1')
    scene = import_scene(str(Path(__file__).resolve().parents[2] / 'scenes' / 'obstacle' / 'O27.usda'))
    terminations = Terminations
    instruction: str = 'Move the red YCB mug through the middle passage to the large plate.'
    episode_length_s: int = 90
    attributes = ["safe_stage", "obstacle"]
    subtasks = None
