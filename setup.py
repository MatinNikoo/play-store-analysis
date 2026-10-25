"""Copies the model files into the package when it is built, so a normal `pip install .` works too."""

import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

ROOT = Path(__file__).resolve().parent
SHIPPED = ["models/milestones.json", "models/comparables.csv.gz", "results/study/market.csv",
           "results/study/checks.json"]


class BuildPy(build_py):
    def run(self):
        super().run()
        dest = Path(self.build_lib) / "playodds" / "_data"
        dest.mkdir(parents=True, exist_ok=True)
        for f in SHIPPED:
            shutil.copy(ROOT / f, dest / Path(f).name)


setup(cmdclass={"build_py": BuildPy})
