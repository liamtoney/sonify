"""
This file only exists to facilitate custom version formatting...
"""

from setuptools import setup

setup(
    use_scm_version=dict(
        local_scheme=lambda version: '+' + version.node[:8],  # "g<7_char_commit_hash>"
        version_scheme=lambda version: '0',
    )
)
