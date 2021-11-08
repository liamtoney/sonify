from setuptools import find_packages, setup

from sonify import __version__

setup(
    name='sonify',
    version=__version__,
    packages=find_packages(),
    entry_points=dict(console_scripts='sonify = sonify.sonify:main'),
)
