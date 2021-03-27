'''
DOC
'''
# %% IMPORTS
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt

# %% FUNCTIONS


def set_size(width_pt=472.03123, fraction=1, aspect_ratio=0.6180339887498949, subplots=(1, 1)):
    """Set figure dimensions to sit nicely in our document.

    Parameters
    ----------
    width_pt: float
            Document width in points
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * aspect_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def latex_plot_config():
    '''
    Config. to output images for LaTex.
    '''
    import matplotlib
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })


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
    return smoothed_signal[int(window_len/2-1):-int(window_len/2)-1]


def axvlines(xs, ax=None, lims=None, **plot_kwargs):
    '''
    Draw vertical lines on plot
    :param xs: A scalar, list, or 1D array of horizontal offsets
    :param ax: The axis (or none to use gca)
    :param lims: Optionally the (ymin, ymax) of the lines
    :param plot_kwargs: Keyword arguments to be passed to plot
    :return: The plot object corresponding to the lines.
    '''
    if ax is None:
        ax = plt.gca()
    xs = np.array((xs, ) if np.isscalar(xs) else xs, copy=False)
    if lims is None:
        lims = ax.get_ylim()
    x_points = np.repeat(xs[:, None], repeats=3, axis=1).flatten()
    y_points = np.repeat(np.array(lims + (np.nan, ))
                         [None, :], repeats=len(xs), axis=0).flatten()
    plot = ax.plot(x_points, y_points, scaley=False, **plot_kwargs)
    return plot


def time_seq():
    '''
    DOC
    '''
    time_array = [333, 333, 333, 334]
    for i in cycle(time_array):
        yield i


def polynomial_function(coefs, arr):
    '''
    Computes polynomial of an array given the coefficients -> polynomial(coefs, arr) = coef[0]*arr**2 + coef[1]*arr + coef[0]

    Args:
    coefs {1darray} -- Coefficients of polynomial from higest to lowest power
    arr {ndarray} -- Independent variable of polynomial

    Returns:
    {ndarray} -- Polynomial of arr with coefs
    '''
    return sum([coefs[len(coefs)-i-1]*arr**i for i in range(len(coefs))])


def derivative(func_arr, x_arr):
    '''
    Derivative of array using central differences.

    Args:
    func_arr {1darray} -- Function to differentiate
    x_arr {1darray} -- Independent variable

    Returns:
    out {1darray} -- Derivative df/dx
    '''
    out = []
    for i, _ in enumerate(func_arr):
        if i == 0:  # forward difference
            dif = (func_arr[i+1]-func_arr[i])/(x_arr[i+1]-x_arr[i])
        elif i == len(func_arr)-1:  # backward difference
            dif = (func_arr[i]-func_arr[i-1])/(x_arr[i]-x_arr[i-1])
        else:  # central difference
            dif = (func_arr[i+1]-func_arr[i-1])/(x_arr[i+1]-x_arr[i-1])
        out.append(dif)
    return np.array(out)
