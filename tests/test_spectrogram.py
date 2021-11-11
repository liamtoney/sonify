import pytest
from obspy import read

from sonify.sonify import _spectrogram

tr = read()[0]  # Grab first Trace in ObsPy's default Stream
tr.remove_response()


@pytest.mark.mpl_image_compare(
    style='default', savefig_kwargs=dict(bbox_inches='tight')
)
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
