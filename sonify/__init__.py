import subprocess
from pathlib import Path

__version__ = subprocess.run(
    [
        'git',
        '-C',
        Path(__file__).resolve().parent,
        'rev-parse',
        '--short=7',
        'HEAD',
    ],
    capture_output=True,
    text=True,
).stdout.strip()

del subprocess
del Path

from .sonify import sonify
