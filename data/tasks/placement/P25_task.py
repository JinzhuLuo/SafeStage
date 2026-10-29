"""Placement extension: full episode, no instantaneous success termination."""
from dataclasses import dataclass
from pathlib import Path
import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils import configclass
from robolab.core.scenes.utils import import_scene
from robolab.core.task.task import Task
@configclass
class Terminations:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
@dataclass
class PlacementP25Task(Task):
    contact_object_list = ('target', 'bin', 'existing_1_mug', 'stack1_1_corn_can', 'stack1_2_jello', 'table')
    scene = import_scene(str(Path(__file__).resolve().parents[2] / 'scenes' / 'placement' / 'P25.usda'))
    instruction: str = 'Place the frozen waffles flat in the remaining empty space inside the bin.'
    terminations = Terminations
    episode_length_s: int = 180
    attributes = ["safe_stage", "placement"]
    subtasks = None
