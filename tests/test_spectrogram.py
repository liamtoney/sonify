import pytest
from obspy import read

from sonify.sonify import _spectrogram

# Set kwargs for all pytest-mpl tests
PYTEST_MPL_KWARGS = dict(style='default', savefig_kwargs=dict(bbox_inches='tight'))

tr = read()[0]  # Grab first Trace in ObsPy's default Stream
tr.remove_response()


@pytest.mark.mpl_image_compare(**PYTEST_MPL_KWARGS)
def test_spectrogram():
    fig = _spectrogram(
        tr=tr,
        starttime=tr.stats.starttime + 2,
        endtime=tr.stats.endtime - 3,
        is_infrasound=False,
        rescale=1e6,  # Convert m to µm
        spec_win_dur=1,
        db_lim='smart',
        freq_lim=(4, 40),
        log=False,
        is_local_time=False,
    )[0]
    return fig
