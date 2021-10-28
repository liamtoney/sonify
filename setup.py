from setuptools import find_packages, setup

setup(
    name='sonify',
    packages=find_packages(),
    entry_points=dict(console_scripts='sonify = sonify.sonify:main'),
)
