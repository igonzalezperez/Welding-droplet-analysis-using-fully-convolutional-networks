'''
DOC
'''
# %% IMPORTS
import numpy as np
# %% FUNCTIONS


def parse_centroid_coords(centroids):
    '''
    Receives list of tuples [(x1, y1), (x2, y2), ...] and returns x and y in separate lists > ([x1, x2, ...],[y1, y2, ...])
    '''
    x_coord = []
    y_coord = []
    for cent in centroids:
        x_coord.append(cent[0])
        y_coord.append(cent[1])
    return x_coord, y_coord


def smooth_signal(signal, window_len=11, window='hanning'):
    """
    smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    np.hanning, np.hamming, np.bartlett, np.blackman, np.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if signal.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if signal.size < window_len:
        raise ValueError(
            "Input vector needs to be bigger than window size.")

    if window_len < 3:
        return signal

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError(
            "Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    signal = np.r_[signal[window_len-1:0:-1],
                   signal, signal[-2:-window_len-1:-1]]

    if window == 'flat':  # moving average
        win = np.ones(window_len, 'd')
    else:
        win = eval('np.'+window+'(window_len)')

    smoothed_signal = np.convolve(win/win.sum(), signal, mode='valid')
    return smoothed_signal
