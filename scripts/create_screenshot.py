#!/usr/bin/env python

"""
Parse the example command from README.rst, run it, and extract a screenshot.

Run with: poe screenshot
"""

import shlex
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg

ROOT_DIR = Path(__file__).resolve().parent.parent
SCREENSHOT = ROOT_DIR / 'docs' / '_static' / 'screenshot.png'

FRAME = 14  # Which frame number to extract (1-indexed)

# Extract frame, then resize
FILTER = rf'select=eq(n\,{FRAME - 1}),scale=iw/2:ih/2,format=yuv444p'

# Grab the text between the markers, dropping RST directive lines
block = (
    (ROOT_DIR / 'README.rst').read_text().split('.. ~BEGIN~')[1].split('.. ~END~')[0]
)
command = shlex.split(
    ' '.join(
        line.strip().removesuffix('\\')
        for line in block.splitlines()
        if line.strip() and not line.startswith('.. ')
    )
)
print(f'Running: {shlex.join(command)}')

with tempfile.TemporaryDirectory() as temp_dir:
    subprocess.run(command, cwd=temp_dir, check=True)
    video_file = next(Path(temp_dir).glob('*.mp4'))
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-v', 'warning', '-i', video_file]
        + ['-vf', FILTER, '-frames:v', '1', '-update', '1', SCREENSHOT],
        check=True,
    )
print(f'Saved {SCREENSHOT}')
