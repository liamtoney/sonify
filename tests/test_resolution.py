import tempfile
from pathlib import Path

import imageio_ffmpeg
from obspy import UTCDateTime

from sonify import sonify
from sonify.sonify import RESOLUTIONS


def test_resolution():
    with tempfile.TemporaryDirectory() as temp_dir_name:
        # Iterate over all resolution options
        for resolution, target_dims in RESOLUTIONS.items():
            # Run sonify for this resolution
            sonify(
                network='AV',
                station='ILSW',
                channel='BHZ',
                starttime=UTCDateTime(2019, 6, 20, 23, 55),
                endtime=UTCDateTime(2019, 6, 21, 0, 10),
                freqmax=20,  # So we avoid the Nyquist warning
                output_dir=temp_dir_name,
                resolution=resolution,
            )
            # Read resolution of output file
            reader = imageio_ffmpeg.read_frames(
                Path(temp_dir_name) / 'AV_ILSW_BHZ_200x.mp4'
            )
            try:
                output_dims = tuple(next(reader)['size'])  # (width, height)
            finally:
                reader.close()  # Terminates the FFmpeg process

            # Test dimensions
            assert output_dims == target_dims, f'Issue with {resolution}!'
