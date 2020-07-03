import os
import subprocess
import warnings

import colorcet as cc
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import AnchoredText
from obspy.clients.fdsn import Client
from scipy import signal

plt.ioff()


LOWEST_AUDIBLE_FREQUENCY = 20  # [Hz]
HIGHEST_AUDIBLE_FREQUENCY = 20000  # [Hz]

AUDIO_SAMPLE_RATE = 44100  # [Hz]

TAPER = 0.01

RESOLUTION = (3840, 2160)  # [px] Output video resolution (width, height)
DPI = 500

# For spectrograms
REFERENCE_PRESSURE = 20e-6  # [Pa]
REFERENCE_VELOCITY = 1  # [m/s]

MS_PER_S = 1000  # [ms/s]

# Colorbar extension triangle height as proportion of colorbar length
EXTENDFRAC = 0.05


def sonify(
    network,
    station,
    channel,
    starttime,
    endtime,
    location='*',
    freqmin=None,
    freqmax=None,
    speed_up_factor=200,
    fps=1,
    output_dir=None,
    spec_win_dur=5,
    db_lim=None,
):
    """
    Produce an animated spectrogram with a soundtrack derived from sped-up
    seismic or infrasound data.

    Args:
        network (str): SEED network code
        station (str): SEED station code
        channel (str): SEED channel code
        starttime (:class:`~obspy.core.utcdatetime.UTCDateTime`): Start time of
            animation
        endtime (:class:`~obspy.core.utcdatetime.UTCDateTime`): End time of
            animation
        location (str): SEED location code
        freqmin (int or float): Lower bandpass corner [Hz] (defaults to
            ``LOWEST_AUDIBLE_FREQUENCY`` / `speed_up_factor`)
        freqmax (int or float): Upper bandpass corner [Hz] (defaults to
            ``HIGHEST_AUDIBLE_FREQUENCY`` / `speed_up_factor` or the
            `Nyquist frequency <https://en.wikipedia.org/wiki/Nyquist_frequency>`__,
            whichever is smaller)
        speed_up_factor (int or float): Factor by which to speed up the
            waveform data (higher values = higher pitches)
        fps (int or float): Frames per second for output video
        output_dir (str): Directory where output video should be saved
            (defaults to :func:`os.getcwd`)
        spec_win_dur (int or float): Duration of spectrogram window [s]
        db_lim (tuple): Tuple specifying colorbar / colormap limits for
            spectrogram [dB]
    """

    # Use current working directory if none provided
    if not output_dir:
        output_dir = os.getcwd()

    pad = (endtime - starttime) * TAPER

    client = Client('IRIS')

    print('Retrieving data...')
    st = client.get_waveforms(
        network,
        station,
        location,
        channel,
        starttime - pad,
        endtime + pad,
        attach_response=True,
    )
    print('Done')

    if st.count() != 1:
        warnings.warn('Stream contains more than one Trace. Using first entry!')
        [print(tr.id) for tr in st]
    tr = st[0]

    # All infrasound sensors have a "?DF" channel pattern
    if tr.stats.channel[1:3] == 'DF':
        is_infrasound = True
    # All high-gain seismometers have a "?H?" channel pattern
    elif tr.stats.channel[1] == 'H':
        is_infrasound = False
    # We can't figure out what type of sensor this is...
    else:
        raise ValueError(
            f'Channel {tr.stats.channel} is not an infrasound or seismic channel!'
        )

    if not freqmax:
        freqmax = np.min(
            [tr.stats.sampling_rate / 2, HIGHEST_AUDIBLE_FREQUENCY / speed_up_factor]
        )
    if not freqmin:
        freqmin = LOWEST_AUDIBLE_FREQUENCY / speed_up_factor

    tr.remove_response()  # Units are m/s OR Pa after response removal
    tr.detrend('demean')
    print(f'Applying {freqmin}-{freqmax} Hz bandpass')
    tr.filter('bandpass', freqmin=freqmin, freqmax=freqmax, zerophase=True)

    # Make trimmed version
    tr_trim = tr.copy()
    tr_trim.trim(starttime, endtime)

    # MAKE AUDIO FILE

    tr_audio = tr_trim.copy()
    tr_audio.interpolate(sampling_rate=AUDIO_SAMPLE_RATE / speed_up_factor)
    tr_audio.taper(TAPER)
    audio_filename = os.path.join(output_dir, 'sonify-tmp.wav')
    print('Saving audio file...')
    tr_audio.write(
        audio_filename, format='WAV', width=4, rescale=True, framerate=AUDIO_SAMPLE_RATE
    )
    print('Done')

    # MAKE VIDEO FILE

    timing_tr = tr_trim.copy().interpolate(sampling_rate=fps / speed_up_factor)
    times = timing_tr.times('UTCDateTime')

    # Define update function
    def _march_forward(frame, spec_line, wf_line, time_box):

        spec_line.set_xdata(times[frame].matplotlib_date)
        wf_line.set_xdata(times[frame].matplotlib_date)
        time_box.txt.set_text(times[frame].strftime('%H:%M:%S'))

        return spec_line, wf_line, time_box

    fig, *fargs = _spectrogram(
        tr,
        starttime,
        endtime,
        is_infrasound,
        win_dur=spec_win_dur,
        db_lim=db_lim,
        freq_lim=(freqmin, freqmax),
    )

    # Create animation
    interval = ((1 / timing_tr.stats.sampling_rate) * MS_PER_S) / speed_up_factor
    animation = FuncAnimation(
        fig,
        func=_march_forward,
        frames=times.size,
        fargs=fargs,
        interval=interval,
        blit=True,
    )

    video_filename = os.path.join(output_dir, 'sonify-tmp.mp4')
    print('Saving animation. This may take a while...')
    animation.save(
        video_filename,
        dpi=DPI,
        progress_callback=lambda i, n: print(
            '{:.1f}%'.format(((i + 1) / n) * 100), end='\r'
        ),
    )
    print('\nDone')

    # MAKE COMBINED FILE

    basename = '_'.join(
        [
            tr.stats.network,
            tr.stats.station,
            tr.stats.channel,
            str(speed_up_factor) + 'x',
        ]
    )
    output_filename = os.path.join(output_dir, f'{basename}.mp4')
    _ffmpeg_combine(audio_filename, video_filename, output_filename)


def _spectrogram(
    tr, starttime, endtime, is_infrasound, win_dur=5, db_lim=None, freq_lim=None
):
    """
    Make a combination waveform and spectrogram plot for an infrasound or
    seismic signal.

    Args:
        tr (:class:`~obspy.core.trace.Trace`): Input data, usually starts
            before `starttime` and ends after `endtime` (this function expects
            the response to be removed!)
        starttime (:class:`~obspy.core.utcdatetime.UTCDateTime`): Start time
        endtime (:class:`~obspy.core.utcdatetime.UTCDateTime`): End time
        is_infrasound (bool): `True` if infrasound, `False` if seismic
        win_dur (int or float): Duration of window [s] (this usually must be
            adjusted depending upon the total duration of the signal)
        db_lim (tuple): Tuple specifying colorbar / colormap limits [dB]
        freq_lim (tuple): Tuple defining frequency limits for spectrogram plot

    Returns:
        Tuple of (`fig`, `spec_line`, `wf_line`, `time_box`)
    """

    if is_infrasound:
        ref_val = REFERENCE_PRESSURE
        ylab = 'Pressure (Pa)'
        clab = 'Power (dB$_{%g\ \mathrm{μPa}}$/Hz)' % (REFERENCE_PRESSURE * 1e6)
        rescale = 1
    else:
        ref_val = REFERENCE_VELOCITY
        ylab = 'Velocity (μm/s)'
        clab = 'Power (dB$_{%g\ \mathrm{m/s}}$/Hz)' % REFERENCE_VELOCITY
        rescale = 1e6  # Converting to μm/s

    fs = tr.stats.sampling_rate
    nperseg = int(win_dur * fs)  # Samples
    nfft = np.power(2, int(np.ceil(np.log2(nperseg))) + 1)  # Pad fft with zeroes

    f, t, sxx = signal.spectrogram(
        tr.data, fs, window='hann', nperseg=nperseg, nfft=nfft
    )

    sxx_db = 20 * np.log10(np.sqrt(sxx) / ref_val)  # [dB / Hz]

    t_mpl = tr.stats.starttime.matplotlib_date + (t / mdates.SEC_PER_DAY)

    fig = plt.figure(figsize=np.array(RESOLUTION) / DPI)

    # width_ratios effectively controls the colorbar width
    gs = GridSpec(2, 2, figure=fig, height_ratios=[2, 1], width_ratios=[40, 1])

    spec_ax = fig.add_subplot(gs[0, 0])
    wf_ax = fig.add_subplot(gs[1, 0], sharex=spec_ax)  # Share x-axis with spec
    cax = fig.add_subplot(gs[0, 1])

    wf_ax.plot(tr.times('matplotlib'), tr.data * rescale, 'k', linewidth=0.5)
    wf_ax.set_ylabel(ylab)
    wf_ax.grid(linestyle=':')
    max_value = np.abs(tr.copy().trim(starttime, endtime).data).max() * rescale
    wf_ax.set_ylim(-max_value, max_value)

    im = spec_ax.pcolormesh(t_mpl, f, sxx_db, cmap=cc.m_rainbow, rasterized=True)

    spec_ax.set_ylabel('Frequency (Hz)')
    spec_ax.grid(linestyle=':')
    spec_ax.set_ylim(freq_lim)

    # Tick locating and formatting
    locator = mdates.AutoDateLocator()
    wf_ax.xaxis.set_major_locator(locator)
    wf_ax.xaxis.set_major_formatter(_UTCDateFormatter(locator))
    fig.autofmt_xdate()

    # "Crop" x-axis!
    wf_ax.set_xlim(starttime.matplotlib_date, endtime.matplotlib_date)

    # Initialize animated stuff
    line_kwargs = dict(x=starttime.matplotlib_date, color='red', linewidth=1)
    spec_line = spec_ax.axvline(**line_kwargs)
    wf_line = wf_ax.axvline(**line_kwargs)
    time_box = AnchoredText(
        s=starttime.strftime('%H:%M:%S'),
        pad=0.2,
        loc='lower right',
        borderpad=0,
        prop=dict(color='red'),
    )
    spec_ax.add_artist(time_box)

    # Clip image to db_lim if provided (doesn't clip if db_lim=None)
    db_min, db_max = im.get_clim()
    im.set_clim(db_lim)

    # Automatically determine whether to show triangle extensions on colorbar
    # (kind of adopted from xarray)
    if db_lim:
        min_extend = db_min < db_lim[0]
        max_extend = db_max > db_lim[1]
    else:
        min_extend = False
        max_extend = False
    if min_extend and max_extend:
        extend = 'both'
    elif min_extend:
        extend = 'min'
    elif max_extend:
        extend = 'max'
    else:
        extend = 'neither'

    fig.colorbar(im, cax, extend=extend, extendfrac=EXTENDFRAC, label=clab)

    spec_ax.set_title(
        '.'.join(
            [tr.stats.network, tr.stats.station, tr.stats.location, tr.stats.channel]
        )
    )

    # Repeat tight_layout and update, janky but works...
    for _ in range(2):
        gs.tight_layout(fig)
        gs.update(hspace=0, wspace=0.05)

    # Finnicky formatting to get extension triangles (if they exist) to extend
    # above and below the vertical extent of the spectrogram axes
    pos = cax.get_position()
    triangle_height = EXTENDFRAC * pos.height
    ymin = pos.ymin
    height = pos.height
    if min_extend and max_extend:
        ymin -= triangle_height
        height += 2 * triangle_height
    elif min_extend and not max_extend:
        ymin -= triangle_height
        height += triangle_height
    elif max_extend and not min_extend:
        height += triangle_height
    else:
        pass
    cax.set_position([pos.xmin, ymin, pos.width, height])

    return fig, spec_line, wf_line, time_box


def _ffmpeg_combine(audio_filename, video_filename, output_filename):
    """
    Combine audio and video files into a single movie. Uses a system call to
    `ffmpeg <https://www.ffmpeg.org/>`__.

    Args:
        audio_filename (str): Audio file to use (full path)
        video_filename (str): Video file to use (full path)
        output_filename (str): Output filename (full path)
    """

    args = [
        'ffmpeg',
        '-y',
        '-v',
        'warning',
        '-i',
        video_filename,
        '-guess_layout_max',
        '0',
        '-i',
        audio_filename,
        '-c:v',
        'copy',
        '-c:a',
        'aac',
        '-b:a',
        '320k',
        '-ac',
        '2',
        output_filename,
    ]
    print('Combining video and audio using ffmpeg...')
    code = subprocess.call(args)
    if code == 0:
        print(f'Video saved as {output_filename}')
        os.remove(audio_filename)
        os.remove(video_filename)
    else:
        raise OSError(
            'Issue with ffmpeg conversion. Check error messages and try again.'
        )


# Subclass ConciseDateFormatter (modifies __init__() and set_axis() methods)
class _UTCDateFormatter(mdates.ConciseDateFormatter):
    def __init__(self, locator, tz=None):
        super().__init__(locator, tz=tz, show_offset=True)

        # Re-format datetimes
        self.formats[1] = '%B'
        self.zero_formats[2:4] = ['%B', '%B %d']
        self.offset_formats = [
            'UTC time',
            'UTC time in %Y',
            'UTC time in %B %Y',
            'UTC time on %B %d, %Y',
            'UTC time on %B %d, %Y',
            'UTC time on %B %d, %Y at %H:%M',
        ]

    def set_axis(self, axis):
        self.axis = axis

        # If this is an x-axis (usually is!) then center the offset text
        if self.axis.axis_name == 'x':
            offset = self.axis.get_offset_text()
            offset.set_horizontalalignment('center')
            offset.set_x(0.5)
