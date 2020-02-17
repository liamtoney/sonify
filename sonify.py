import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import spectrogram
import colorcet as cc
import warnings
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import os
from matplotlib.animation import FuncAnimation
from matplotlib.offsetbox import AnchoredText
import subprocess

plt.ioff()


LOWEST_AUDIBLE_FREQUENCY = 20  # [Hz]
HIGHEST_AUDIBLE_FREQUENCY = 20000  # [Hz]

AUDIO_SAMPLE_RATE = 44100  # [Hz]

TAPER = 0.01

RESOLUTION = (3840, 2160)  # [px] Output video resolution (width, height)
DPI = 400

# For spectrograms
REFERENCE_PRESSURE = 20e-6  # [Pa]
REFERENCE_VELOCITY = 1  # [m/s]

MS_PER_S = 1000  # [ms/s]


def sonify(network, station, channel, starttime, endtime, location='*',
           freqmin=None, freqmax=None, speed_up_factor=200, fps=1,
           output_dir=None, spec_win_dur=5, db_lim=None):

    # Use current working directory if none provided
    if not output_dir:
        output_dir = os.getcwd()

    pad = (endtime - starttime) * TAPER

    client = Client('IRIS')

    print('Retrieving data...')
    st = client.get_waveforms(network, station, location, channel,
                              starttime - pad, endtime + pad,
                              attach_response=True)
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
        raise ValueError(f'Channel {tr.stats.channel} is not an infrasound or '
                         'seismic channel!')

    if not freqmax:
        freqmax = np.min([tr.stats.sampling_rate / 2,
                          HIGHEST_AUDIBLE_FREQUENCY / speed_up_factor])
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
    tr_audio.write(audio_filename, format='WAV', width=4, rescale=True,
                   framerate=AUDIO_SAMPLE_RATE)
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

    fig, *fargs = _spectrogram(tr, starttime, endtime, is_infrasound,
                               win_dur=spec_win_dur, db_lim=db_lim,
                               freq_lim=(freqmin, freqmax))

    # Create animation
    interval = ((1 / timing_tr.stats.sampling_rate) * MS_PER_S) / speed_up_factor
    animation = FuncAnimation(fig, func=_march_forward, frames=times.size,
                              fargs=fargs, interval=interval, blit=True)

    video_filename = os.path.join(output_dir, 'sonify-tmp.mp4')
    print('Saving animation. This may take a while...')
    animation.save(video_filename, dpi=DPI,
                   progress_callback=lambda i, n: print('{:.1f}%'.format(((i + 1) / n) * 100), end='\r'))
    print('\nDone')

    # MAKE COMBINED FILE

    basename = '_'.join([tr.stats.network, tr.stats.station, tr.stats.channel,
                         str(speed_up_factor) + 'x'])
    output_filename = os.path.join(output_dir, f'{basename}.mp4')
    _ffmpeg_combine(audio_filename, video_filename, output_filename)


def _spectrogram(tr, starttime, endtime, is_infrasound, win_dur=5, db_lim=None,
                 freq_lim=None):
    """
    Make a combination trace and spectrogram plot for an infrasound or seismic
    signal.

    Args:
        tr: ObsPy Trace object (this code expects the response to be removed!)
        starttime: UTCDateTime
        endtime: UTCDateTime
        is_infrasound (bool): True if infrasound, False if seismic
        win_dur: Segment length in seconds. This usually must be adjusted
            depending upon the total duration of the signal (default: 5)
        db_lim: Tuple defining min and max dB colormap cutoffs (default: None,
            i.e. don't clip at all)
        freq_lim: Tuple defining frequency limits for spectrogram plot (default:
            None, i.e. use automatically scaled limits)

    Returns:
        Figure with combination plot
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
    nperseg = int(win_dur*fs)  # Samples
    nfft = np.power(2, int(np.ceil(np.log2(nperseg)))+1)  # Pad fft with zeroes

    f, t, sxx = spectrogram(tr.data, fs, window='hann', nperseg=nperseg,
                            nfft=nfft)

    sxx_db = 20*np.log10(np.sqrt(sxx)/ref_val)  # [dB / Hz]

    t_mpl = tr.stats.starttime.matplotlib_date + (t / mdates.SEC_PER_DAY)

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex='all',
                             figsize=np.array(RESOLUTION) / DPI,
                             gridspec_kw=dict(height_ratios=[2, 1],
                                              hspace=0))

    axes[1].plot(tr.times('matplotlib'), tr.data*rescale, 'k', linewidth=0.5)
    axes[1].set_ylabel(ylab)
    axes[1].grid(linestyle=':')
    max_value = np.abs(tr.copy().trim(starttime, endtime).data).max() * rescale
    axes[1].set_ylim(-max_value, max_value)

    im = axes[0].pcolormesh(t_mpl, f, sxx_db, cmap=cc.m_rainbow,
                            rasterized=True)

    axes[0].set_ylabel('Frequency (Hz)')
    axes[0].grid(linestyle=':')
    axes[0].set_ylim(freq_lim)

    date = tr.stats.starttime.strftime('%B %d, %Y')
    axes[1].set_xlabel('UTC time (HH:MM) starting on {}'.format(date))
    axes[1].set_xlim(starttime.matplotlib_date, endtime.matplotlib_date)
    axes[1].xaxis_date()
    formatter = axes[1].xaxis.get_major_formatter()
    formatter.scaled[1/mdates.MINUTES_PER_DAY] = '%H:%M'
    formatter.scaled[1/mdates.SEC_PER_DAY] = '%H:%M:%S'
    fig.autofmt_xdate()

    # Initialize animated stuff
    line_kwargs = dict(x=starttime.matplotlib_date, color='red', linewidth=1)
    spec_line = axes[0].axvline(**line_kwargs)
    wf_line = axes[1].axvline(**line_kwargs)
    time_box = AnchoredText(s=starttime.strftime('%H:%M:%S'), pad=0.2,
                            loc='lower right', borderpad=0, prop=dict(color='red'))
    axes[0].add_artist(time_box)

    im.set_clim(db_lim)

    # Make room for colorbar in figure window
    fig.subplots_adjust(right=0.85)

    box = axes[0].get_position()
    cax = fig.add_axes([box.xmax+box.height/20, box.y0, box.height/20,
                        box.height])
    cbar = fig.colorbar(im, cax)
    cbar.set_label(clab)

    axes[0].set_title('.'.join([tr.stats.network, tr.stats.station,
                                tr.stats.location, tr.stats.channel]))

    return fig, spec_line, wf_line, time_box


def _ffmpeg_combine(audio_filename, video_filename, output_filename):
    """
    Combine video and audio files into a single movie. Uses a system call to
    ffmpeg.

    Args:
        audio_filename (str): Audio file to use (full path)
        video_filename (str): Video file to use (full path)
        output_filename (str): Output filename (full path)
    """

    args = ['ffmpeg', '-y', '-v', 'warning', '-i', video_filename, '-i',
            audio_filename, '-c:v', 'copy', '-c:a', 'aac', '-ac', '2',
            output_filename]
    print('Combining video and audio using ffmpeg...')
    code = subprocess.call(args)
    if code == 0:
        print(f'Video saved as {output_filename}')
        os.remove(audio_filename)
        os.remove(video_filename)
    else:
        raise OSError('Issue with ffmpeg conversion. Check error messages and '
                      'try again.')
