import subprocess
import tempfile
from pathlib import Path

from sonify.sonify import _ffmpeg_combine

TEST_DATA_DIR = Path(__file__).resolve().parent / 'data'
BASELINE_DIR = Path(__file__).resolve().parent / 'baseline'

# Created with: sonify AV ILSW BHZ 2019-06-21T00:00 2019-06-21T00:10
# Temporary files 47.wav and 47.mp4 captured and renamed to test.wav and test.mp4
# Output video AV_ILSW_BHZ_200x.mp4 renamed to test_ffmpeg_combine.mp4
AUDIO_FILE = TEST_DATA_DIR / 'test.wav'
VIDEO_FILE = TEST_DATA_DIR / 'test.mp4'
BASELINE_FILE = BASELINE_DIR / 'test_ffmpeg_combine.mp4'


def _get_md5(video_file):
    md5 = (
        subprocess.run(
            ['ffmpeg', '-i', video_file, '-f', 'md5', '-'],
            capture_output=True,
            text=True,
        )
        .stdout.strip()
        .split('=')[-1]
    )
    return md5


original_hash = _get_md5(BASELINE_FILE)


def test_ffmpeg_combine():
    with tempfile.NamedTemporaryFile(suffix='.mp4') as f:
        _ffmpeg_combine(AUDIO_FILE, VIDEO_FILE, Path(f.name), call_str='')
        test_hash = _get_md5(f.name)
    assert test_hash == original_hash
