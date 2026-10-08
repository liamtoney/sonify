#!/usr/bin/env python

"""
Parse the example command from README.rst, run it, and extract a screenshot.

Run with: poe screenshot
"""

import re
import shlex
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg

REPO_DIR = Path(__file__).resolve().parent.parent
README = REPO_DIR / 'README.rst'
SCREENSHOT = REPO_DIR / 'screenshot.png'

FRAME = 14  # Which frame number to extract (1-indexed)

# Extract frame, then resize. Commas are escaped for FFmpeg's filter syntax.
FILTER = rf'select=eq(n\,{FRAME - 1}),scale=iw/2:ih/2,format=yuv444p'


def _get_command():
    """Return the sonify call between the ~BEGIN~ and ~END~ markers as a list."""
    text = README.read_text()
    block = re.search(
        r'^\.\. ~BEGIN~$(.*?)^\.\. ~END~$', text, re.MULTILINE | re.DOTALL
    )
    lines = [
        line.strip().removesuffix('\\').strip()
        for line in block.group(1).splitlines()
        # Drop blank lines and RST directives/comments
        if line.strip() and not line.startswith('.. ')
    ]
    return shlex.split(' '.join(lines))


if __name__ == '__main__':
    command = _get_command()
    print(f'Running: {shlex.join(command)}')

    # The temporary directory takes the place of the manual `rm *.mp4`
    with tempfile.TemporaryDirectory() as temp_dir:
        subprocess.run(command, cwd=temp_dir, check=True)
        video_file = next(Path(temp_dir).glob('*.mp4'))
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                '-y',
                '-v',
                'warning',
                '-i',
                str(video_file),
                '-filter:v',
                FILTER,
                '-frames:v',
                '1',
                '-update',
                '1',
                str(SCREENSHOT),
            ],
            check=True,
        )
    print(f'Saved {SCREENSHOT}')
