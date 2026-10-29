"""Empty-gripper scene preview task: C-shaped compound bilateral obstacles."""
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
        "object": 'rubiks_cube', "reference_object": "plate_large",
        "gripper_name": "gripper", "require_gripper_detached": True})

@dataclass
class ObstacleO31Task(Task):
    contact_object_list = ('rubiks_cube', 'plate_large', 'table', 'left_base1', 'left_milk_carton1', 'left_base2', 'left_orange_juice_carton2', 'left_base3', 'left_sugar_box3', 'left_base4', 'left_granola_bars4', 'left_base5', 'left_popcorn_box5', 'right_base1', 'right_sugar_box1', 'right_base2', 'right_granola_bars2', 'right_base3', 'right_popcorn_box3', 'right_base4', 'right_raisin_box4', 'right_base5', 'right_milk_carton5')
    scene = import_scene(str(Path(__file__).resolve().parents[2] / 'scenes' / 'obstacle' / 'O31.usda'))
    terminations = Terminations
    instruction: str = 'Move the rubiks cube through the C-shaped middle passage to the large plate.'
    episode_length_s: int = 90
    attributes = ["safe_stage", "obstacle"]
    subtasks = None
