from pathlib import Path

from setuptools import find_packages, setup

setup(
    name='sonify',
    packages=find_packages(),
    scripts=[str(Path('bin') / 'sonify')],
)
