#!/usr/bin/env python

import argparse
import subprocess
import tempfile
import warnings
from pathlib import Path
from types import MethodType

import matplotlib.dates as mdates
import numpy as np
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import AnchoredText
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from scipy import signal

LOWEST_AUDIBLE_FREQUENCY = 20  # [Hz]
HIGHEST_AUDIBLE_FREQUENCY = 20000  # [Hz]

AUDIO_SAMPLE_RATE = 44100  # [Hz]

PAD = 60  # [s] Extra data to download on either side of requested time slice

RESOLUTION = (3840, 2160)  # [px] Output video resolution (width, height)
DPI = 500

# For spectrograms
REFERENCE_PRESSURE = 20e-6  # [Pa]
REFERENCE_VELOCITY = 1  # [m/s]

MS_PER_S = 1000  # [ms/s]

# Colorbar extension triangle height as proportion of colorbar length
EXTENDFRAC = 0.04


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
    db_lim='smart',
    log=False,
    utc_offset=None,
):
    """
    Produce an animated spectrogram with a soundtrack derived from sped-up
    seismic or infrasound data.

    Args:
        network (str): SEED network code
        station (str): SEED station code
        channel (str): SEED channel code
        starttime (:class:`~obspy.core.utcdatetime.UTCDateTime`): Start time of
            animation (UTC)
        endtime (:class:`~obspy.core.utcdatetime.UTCDateTime`): End time of
            animation (UTC)
        location (str): SEED location code
        freqmin (int or float): Lower bandpass corner [Hz] (defaults to
            ``LOWEST_AUDIBLE_FREQUENCY`` / `speed_up_factor`)
        freqmax (int or float): Upper bandpass corner [Hz] (defaults to
            ``HIGHEST_AUDIBLE_FREQUENCY`` / `speed_up_factor` or the
            `Nyquist frequency <https://en.wikipedia.org/wiki/Nyquist_frequency>`__,
            whichever is smaller)
        speed_up_factor (int): Factor by which to speed up the waveform data
            (higher values = higher pitches)
        fps (int): Frames per second for output video
        output_dir (str or :class:`~pathlib.Path`): Directory where output video
            should be saved (defaults to :meth:`~pathlib.Path.cwd`)
        spec_win_dur (int or float): Duration of spectrogram window [s]
        db_lim (tuple or str): Tuple defining min and max colormap cutoffs [dB],
            `'smart'` for a sensible automatic choice, or `None` for no clipping
        log (bool): If `True`, use log scaling for :math:`y`-axis of spectrogram
        utc_offset (int or float): If not `None`, convert UTC time to local time
            using this offset [hours] before plotting
    """

    # Use current working directory if none provided
    if not output_dir:
        output_dir = Path().cwd()
    output_dir = Path(str(output_dir)).expanduser().resolve()
    if not output_dir.exists():
        raise FileNotFoundError(f'Directory {output_dir} does not exist!')

    client = Client('IRIS')

    print('Retrieving data...')
    st = client.get_waveforms(
        network,
        station,
        location,
        channel,
        starttime - PAD,
        endtime + PAD,
        attach_response=True,
    )
    print('Done')

    # Merge Traces with the same IDs
    st.merge(fill_value='interpolate')

    if st.count() != 1:
        warnings.warn('Stream contains more than one Trace. Using first entry!')
        for tr in st:
            print(tr.id)
    tr = st[0]

    # Adjust starttime so we have nice numbers in time box (carefully!)
    offset = np.abs(tr.stats.starttime - (starttime - PAD))  # [s]
    if offset > tr.stats.delta:
        warnings.warn(
            f'Difference between requested and actual starttime is {offset} s, '
            f'which is larger than the data sample interval ({tr.stats.delta} s). '
            'Not adjusting starttime of downloaded data; beware of inaccurate timing!'
        )
    else:
        tr.stats.starttime = starttime - PAD

    # Apply UTC offset if provided
    if utc_offset is not None:
        signed_offset = f'{utc_offset:{"+" if utc_offset else ""}}'
        print(f'Converting to local time using UTC offset of {signed_offset} hours')
        utc_offset_sec = utc_offset * mdates.SEC_PER_HOUR
        starttime += utc_offset_sec
        endtime += utc_offset_sec
        tr.stats.starttime += utc_offset_sec

    # All infrasound sensors have a "?DF" channel pattern
    if tr.stats.channel[1:3] == 'DF':
        is_infrasound = True
        rescale = 1  # No conversion
    # All high-gain seismometers have a "?H?" channel pattern
    elif tr.stats.channel[1] == 'H':
        is_infrasound = False
        rescale = 1e6  # Convert m to μm
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
    tr.taper(max_percentage=None, max_length=PAD / 2)  # Taper away some of PAD
    print(f'Applying {freqmin}-{freqmax} Hz bandpass')
    tr.filter('bandpass', freqmin=freqmin, freqmax=freqmax, zerophase=True)

    # Make trimmed version
    tr_trim = tr.copy()
    tr_trim.trim(starttime, endtime)

    # Create temporary directory for audio and video files
    temp_dir = tempfile.TemporaryDirectory()

    # MAKE AUDIO FILE

    tr_audio = tr_trim.copy()
    tr_audio.interpolate(sampling_rate=AUDIO_SAMPLE_RATE / speed_up_factor)
    tr_audio.taper(0.01)  # For smooth start and end
    audio_file = Path(temp_dir.name) / '47.wav'
    print('Saving audio file...')
    tr_audio.write(
        str(audio_file),
        format='WAV',
        width=4,
        rescale=True,
        framerate=AUDIO_SAMPLE_RATE,
    )
    print('Done')

    # MAKE VIDEO FILE

    timing_tr = tr_trim.copy().interpolate(sampling_rate=fps / speed_up_factor)
    times = timing_tr.times('UTCDateTime')[:-1]  # Remove extra frame

    # Define update function
    def _march_forward(frame, spec_line, wf_line, time_box, wf_progress):

        spec_line.set_xdata(times[frame].matplotlib_date)
        wf_line.set_xdata(times[frame].matplotlib_date)
        time_box.txt.set_text(times[frame].strftime('%H:%M:%S'))
        tr_progress = tr.copy().trim(endtime=times[frame])
        wf_progress.set_xdata(tr_progress.times('matplotlib'))
        wf_progress.set_ydata(tr_progress.data * rescale)

    fig, *fargs = _spectrogram(
        tr,
        starttime,
        endtime,
        is_infrasound,
        rescale,
        spec_win_dur,
        db_lim,
        (freqmin, freqmax),
        log,
        utc_offset is not None,
    )

    # Create animation
    interval = ((1 / timing_tr.stats.sampling_rate) * MS_PER_S) / speed_up_factor
    animation = FuncAnimation(
        fig,
        func=_march_forward,
        frames=times.size,
        fargs=fargs,
        interval=interval,
    )

    video_file = Path(temp_dir.name) / '47.mp4'
    print('Saving animation. This may take a while...')
    animation.save(
        video_file,
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
    output_file = output_dir / f'{basename}.mp4'
    _ffmpeg_combine(audio_file, video_file, output_file)

    # Clean up temporary directory, just to be safe
    temp_dir.cleanup()


def _spectrogram(
    tr,
    starttime,
    endtime,
    is_infrasound,
    rescale,
    spec_win_dur,
    db_lim,
    freq_lim,
    log,
    is_local_time,
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
        rescale (int or float): Scale waveforms by this factor for plotting
        spec_win_dur (int or float): See docstring for :func:`~sonify.sonify`
        db_lim (tuple or str): See docstring for :func:`~sonify.sonify`
        freq_lim (tuple): Tuple defining frequency limits for spectrogram plot
        log (bool): See docstring for :func:`~sonify.sonify`
        is_local_time (bool): Passed to :class:`_UTCDateFormatter`

    Returns:
        Tuple of (`fig`, `spec_line`, `wf_line`, `time_box`, `wf_progress`)
    """

    if is_infrasound:
        ylab = 'Pressure (Pa)'
        clab = f'Power (dB rel. [{REFERENCE_PRESSURE * 1e6:g} μPa]$^2$ Hz$^{{-1}}$)'
        ref_val = REFERENCE_PRESSURE
    else:
        ylab = 'Velocity (μm s$^{-1}$)'
        if REFERENCE_VELOCITY == 1:
            clab = (
                f'Power (dB rel. {REFERENCE_VELOCITY:g} [m s$^{{-1}}$]$^2$ Hz$^{{-1}}$)'
            )
        else:
            clab = (
                f'Power (dB rel. [{REFERENCE_VELOCITY:g} m s$^{{-1}}$]$^2$ Hz$^{{-1}}$)'
            )
        ref_val = REFERENCE_VELOCITY

    fs = tr.stats.sampling_rate
    nperseg = int(spec_win_dur * fs)  # Samples
    nfft = np.power(2, int(np.ceil(np.log2(nperseg))) + 1)  # Pad fft with zeroes

    f, t, sxx = signal.spectrogram(
        tr.data, fs, window='hann', nperseg=nperseg, noverlap=nperseg // 2, nfft=nfft
    )

    # [dB rel. (ref_val <ref_val_unit>)^2 Hz^-1]
    sxx_db = 10 * np.log10(sxx / (ref_val ** 2))

    t_mpl = tr.stats.starttime.matplotlib_date + (t / mdates.SEC_PER_DAY)

    fig = Figure(figsize=np.array(RESOLUTION) / DPI)

    # width_ratios effectively controls the colorbar width
    gs = GridSpec(2, 2, figure=fig, height_ratios=[2, 1], width_ratios=[40, 1])

    spec_ax = fig.add_subplot(gs[0, 0])
    wf_ax = fig.add_subplot(gs[1, 0], sharex=spec_ax)  # Share x-axis with spec
    cax = fig.add_subplot(gs[0, 1])

    wf_lw = 0.5
    wf_ax.plot(tr.times('matplotlib'), tr.data * rescale, '#b0b0b0', linewidth=wf_lw)
    wf_progress = wf_ax.plot(np.nan, np.nan, 'black', linewidth=wf_lw)[0]
    wf_ax.set_ylabel(ylab)
    wf_ax.grid(linestyle=':')
    max_value = np.abs(tr.copy().trim(starttime, endtime).data).max() * rescale
    wf_ax.set_ylim(-max_value, max_value)

    im = spec_ax.pcolormesh(
        t_mpl, f, sxx_db, cmap='inferno', shading='nearest', rasterized=True
    )

    spec_ax.set_ylabel('Frequency (Hz)')
    spec_ax.grid(linestyle=':')
    spec_ax.set_ylim(freq_lim)
    if log:
        spec_ax.set_yscale('log')

    # Tick locating and formatting
    locator = mdates.AutoDateLocator()
    wf_ax.xaxis.set_major_locator(locator)
    wf_ax.xaxis.set_major_formatter(_UTCDateFormatter(locator, is_local_time))
    fig.autofmt_xdate()

    # "Crop" x-axis!
    wf_ax.set_xlim(starttime.matplotlib_date, endtime.matplotlib_date)

    # Initialize animated stuff
    line_kwargs = dict(x=starttime.matplotlib_date, color='forestgreen', linewidth=1)
    spec_line = spec_ax.axvline(**line_kwargs)
    wf_line = wf_ax.axvline(ymin=0.01, clip_on=False, zorder=10, **line_kwargs)
    time_box = AnchoredText(
        s=starttime.strftime('%H:%M:%S'),
        pad=0.2,
        loc='lower right',
        bbox_to_anchor=[1, 1],
        bbox_transform=wf_ax.transAxes,
        borderpad=0,
        prop=dict(color='forestgreen'),
    )
    time_box.txt._text.set_y(-5)  # [pixels] Shift text to vertically center it
    time_box.zorder = 12  # This should place it on the very top; see below
    time_box.patch.set_linewidth(rcParams['axes.linewidth'])
    wf_ax.add_artist(time_box)

    # Adjustments to ensure time marker line is zordered properly
    # 9 is below marker; 11 is above marker
    spec_ax.spines['bottom'].set_zorder(9)
    wf_ax.spines['top'].set_zorder(9)
    for side in 'bottom', 'left', 'right':
        wf_ax.spines[side].set_zorder(11)

    # Pick smart limits rounded to nearest 10
    if db_lim == 'smart':
        db_min = np.percentile(sxx_db, 20)
        db_max = sxx_db.max()
        db_lim = (np.ceil(db_min / 10) * 10, np.floor(db_max / 10) * 10)

    # Clip image to db_lim if provided (doesn't clip if db_lim=None)
    im.set_clim(db_lim)

    # Automatically determine whether to show triangle extensions on colorbar
    # (kind of adopted from xarray)
    if db_lim:
        min_extend = sxx_db.min() < db_lim[0]
        max_extend = sxx_db.max() > db_lim[1]
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

    spec_ax.set_title(tr.id)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0, wspace=0.05)

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

    return fig, spec_line, wf_line, time_box, wf_progress


def _ffmpeg_combine(audio_file, video_file, output_file):
    """
    Combine audio and video files into a single movie. Uses a system call to
    `ffmpeg <https://www.ffmpeg.org/>`__.

    Args:
        audio_file (:class:`~pathlib.Path`): Audio file to use
        video_file (:class:`~pathlib.Path`): Video file to use
        output_file (:class:`~pathlib.Path`): Output file (full path)
    """

    args = [
        'ffmpeg',
        '-y',
        '-v',
        'warning',
        '-i',
        video_file,
        '-guess_layout_max',
        '0',
        '-i',
        audio_file,
        '-c:v',
        'copy',
        '-c:a',
        'aac',
        '-b:a',
        '320k',
        '-ac',
        '2',
        output_file,
    ]
    print('Combining video and audio using ffmpeg...')
    code = subprocess.call(args)

    if code == 0:
        print(f'Video saved as {output_file}')
    else:
        output_file.unlink(missing_ok=True)  # Remove file if it was made
        raise OSError(
            'Issue with ffmpeg conversion. Check error messages and try again.'
        )


# Subclass ConciseDateFormatter (modifies __init__() and set_axis() methods)
class _UTCDateFormatter(mdates.ConciseDateFormatter):
    def __init__(self, locator, is_local_time):
        super().__init__(locator)

        # Determine proper time label (local time or UTC)
        if is_local_time:
            time_type = 'Local'
        else:
            time_type = 'UTC'

        # Re-format datetimes
        self.formats[1] = '%B'
        self.zero_formats[2:4] = ['%B', '%B %d']
        self.offset_formats = [
            f'{time_type} time',
            f'{time_type} time in %Y',
            f'{time_type} time in %B %Y',
            f'{time_type} time on %B %d, %Y',
            f'{time_type} time on %B %d, %Y',
            f'{time_type} time on %B %d, %Y at %H:%M',
        ]

    def set_axis(self, axis):
        self.axis = axis

        # If this is an x-axis (usually is!) then center the offset text
        if self.axis.axis_name == 'x':
            offset = self.axis.get_offset_text()
            offset.set_horizontalalignment('center')
            offset.set_x(0.5)


def main():
    """
    This function is run when sonify.py is called as a script.
    """

    parser = argparse.ArgumentParser(
        description='Produce an animated spectrogram with a soundtrack derived from sped-up seismic or infrasound data.',
        allow_abbrev=False,
    )

    # Hack the printing function of the parser to fix --db_lim option formatting
    def _print_message_replace(self, message, file=None):
        if message:
            if file is None:
                file = _sys.stderr
            file.write(message.replace('[DB_LIM ...]', '[DB_LIM]'))

    parser._print_message = MethodType(_print_message_replace, parser)

    parser.add_argument('network', help='SEED network code')
    parser.add_argument('station', help='SEED station code')
    parser.add_argument('channel', help='SEED channel code')
    parser.add_argument(
        'starttime',
        type=UTCDateTime,
        help='start time of animation (UTC), format yyyy-mm-ddThh:mm:ss',
    )
    parser.add_argument(
        'endtime',
        type=UTCDateTime,
        help='end time of animation (UTC), format yyyy-mm-ddThh:mm:ss',
    )
    parser.add_argument('--location', default='*', help='SEED location code')
    parser.add_argument(
        '--freqmin',
        default=None,
        type=float,
        help=f'lower bandpass corner [Hz] (defaults to {LOWEST_AUDIBLE_FREQUENCY} Hz / "SPEED_UP_FACTOR")',
    )
    parser.add_argument(
        '--freqmax',
        default=None,
        type=float,
        help=f'upper bandpass corner [Hz] (defaults to {HIGHEST_AUDIBLE_FREQUENCY:,} Hz / "SPEED_UP_FACTOR" or the Nyquist frequency, whichever is smaller)',
    )
    parser.add_argument(
        '--speed_up_factor',
        default=200,
        type=int,
        help='factor by which to speed up the waveform data (higher values = higher pitches)',
    )
    parser.add_argument(
        '--fps', default=1, type=int, help='frames per second for output video'
    )
    parser.add_argument(
        '--output_dir',
        default=None,
        help='directory where output video should be saved (defaults to current working directory)',
    )
    parser.add_argument(
        '--spec_win_dur',
        default=5,
        type=float,
        help='duration of spectrogram window [s]',
    )
    parser.add_argument(
        '--db_lim',
        default='smart',
        nargs='+',
        help='numbers "<min>" "<max>" defining min and max colormap cutoffs [dB], "smart" for a sensible automatic choice, or "None" for no clipping',
    )
    parser.add_argument(
        '--log',
        action='store_true',
        help='use log scaling for y-axis of spectrogram',
    )
    parser.add_argument(
        '--utc_offset',
        default=None,
        type=float,
        help='if provided, convert UTC time to local time using this offset [hours] before plotting',
    )

    input_args = parser.parse_args()

    # Extra type check for db_lim kwarg
    db_lim_error = False
    db_lim = np.atleast_1d(input_args.db_lim)
    if db_lim.size == 1:
        db_lim = db_lim[0]
        if db_lim == 'smart':
            pass
        elif db_lim == 'None':
            db_lim = None
        else:
            db_lim_error = True
    elif db_lim.size == 2:
        try:
            db_lim = tuple(float(s) for s in db_lim)
        except ValueError:
            db_lim_error = True
    else:  # User provided more than 2 args
        db_lim_error = True
    if db_lim_error:
        parser.error(
            'argument --db_lim: must be one of "smart", "None", or two numeric values "<min>" "<max>"'
        )

    sonify(
        input_args.network,
        input_args.station,
        input_args.channel,
        input_args.starttime,
        input_args.endtime,
        input_args.location,
        input_args.freqmin,
        input_args.freqmax,
        input_args.speed_up_factor,
        input_args.fps,
        input_args.output_dir,
        input_args.spec_win_dur,
        db_lim,
        input_args.log,
        input_args.utc_offset,
    )


if __name__ == '__main__':
    main()
