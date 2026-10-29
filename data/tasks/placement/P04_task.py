"""No instantaneous-success termination: evaluate a full 5 s release window."""
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
class PlacementP04Task(Task):
    contact_object_list = ('orange_01', 'storage_box', 'table')
    scene = import_scene(str(Path(__file__).resolve().parents[2] / 'scenes' / 'placement' / 'P04.usda'))
    instruction: str = 'Place the orange on the top of the upside-down storage box.'
    terminations = Terminations
    episode_length_s: int = 90
    attributes = ["safe_stage", "placement"]
    subtasks = None
