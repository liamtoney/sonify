"""
This file only exists to facilitate custom version formatting...
"""

from setuptools import setup

setup(
    use_scm_version=dict(
        local_scheme=lambda version: '+' + version.node,
        version_scheme=lambda version: '0',
        version_file='sonify/_version.py',
    )
)
