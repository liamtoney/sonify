import subprocess
from pathlib import Path

__version__ = (
    '0+g'  # Makes the version number PEP 440 compliant
    + subprocess.run(
        [
            'git',
            '-C',
            Path(__file__).resolve().parent,
            'rev-parse',
            '--short=7',  # First 7 characters of the commit hash
            'HEAD',
        ],
        capture_output=True,
        text=True,
    ).stdout.strip()
)

del subprocess
del Path

from .sonify import sonify
