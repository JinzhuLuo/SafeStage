"""Include core scenes, documentation, and previews in wheels."""
from pathlib import Path
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py

ROOT = Path(__file__).parent


class BuildScenePackage(build_py):
    def run(self):
        super().run()
        target = Path(self.build_lib) / 'safe_stage' / '_data'
        for relative in ['catalog', 'prompts', 'data/scenes', 'data/tasks', 'data/specs', 'media', 'docs']:
            source = ROOT / relative
            if source.is_dir():
                shutil.copytree(source, target / relative, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for relative in ['README.md', 'LICENSE', 'THIRD_PARTY_NOTICES.md']:
            shutil.copy2(ROOT / relative, target / relative)
        shutil.copytree(ROOT / 'licenses', target / 'licenses', dirs_exist_ok=True)

    def get_outputs(self, include_bytecode=1):
        outputs = super().get_outputs(include_bytecode)
        folder = Path(self.build_lib) / 'safe_stage' / '_data'
        if folder.exists():
            outputs.extend(str(p) for p in folder.rglob('*') if p.is_file())
        return outputs


setup(cmdclass={'build_py': BuildScenePackage})

