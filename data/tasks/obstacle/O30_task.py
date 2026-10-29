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
        "object": 'canned_peaches', "reference_object": "plate_large",
        "gripper_name": "gripper", "require_gripper_detached": True})

@dataclass
class ObstacleO30Task(Task):
    contact_object_list = ('canned_peaches', 'plate_large', 'table', 'left_base1', 'left_base2', 'left_coffee_pot1', 'left_utilityjug_a011', 'right_base1', 'right_base2', 'right_pitcher1', 'right_milkjug_a031')
    scene = import_scene(str(Path(__file__).resolve().parents[2] / 'scenes' / 'obstacle' / 'O30.usda'))
    terminations = Terminations
    instruction: str = 'Move the peach can through the middle passage to the large plate.'
    episode_length_s: int = 90
    attributes = ["safe_stage", "obstacle"]
    subtasks = None
