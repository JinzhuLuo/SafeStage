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
class PlacementP28Task(Task):
    contact_object_list = ('target', 'support_left', 'support_right', 'table')
    scene = import_scene(str(Path(__file__).resolve().parents[2] / 'scenes' / 'placement' / 'P28.usda'))
    instruction: str = 'Place the raisin box flat across both low blocks, with each end supported and the middle over the gap.'
    terminations = Terminations
    episode_length_s: int = 180
    attributes = ["safe_stage", "placement"]
    subtasks = None
