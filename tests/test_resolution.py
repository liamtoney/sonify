import subprocess
import tempfile
from pathlib import Path

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
            output_dims = tuple(
                int(d)
                for d in subprocess.run(
                    [
                        'ffprobe',
                        '-loglevel',
                        'error',
                        '-show_entries',
                        'stream=width,height',
                        '-of',
                        'default=nokey=1:noprint_wrappers=1',
                        str(Path(temp_dir_name) / 'AV_ILSW_BHZ_200x.mp4'),
                    ],
                    capture_output=True,
                    text=True,
                )
                .stdout.strip()
                .split('\n')
            )

            # Test dimensions
            assert output_dims == target_dims, f'Issue with {resolution}!'
