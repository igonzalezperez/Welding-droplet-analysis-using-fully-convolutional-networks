import pickle
import numpy as np
import scipy
from scipy.fft import fft
import cv2
from numpy.fft import fftshift
import matplotlib.pyplot as plt
import compute_properties as cp
'''
Pixel to distance conversions
26 px = .045 in = 1.143 mm
1px = 0.04396153846153846 mm = 4.396153846153846 * 10^(-5) m
'''


def compute_fft(y):
    N = len(y)
    T = 333*10**(-6)
    x = np.linspace(0.0, N*T, N)
    yf = fftshift(fft(y))
    xf = np.linspace(0.0, 1.0/(2.0*T), int(N/2))

    return xf, yf, N


P1 = 334*10 ** (-6)  # period between frames in [s]
P2 = 333*10 ** (-6)
PX_TO_MM = 4.396153846153846 * 10**(-2)
DATASET = 'Globular'
with open('Geometry/' + DATASET + '.pickle', 'rb') as f:
    GEOMETRY = pickle.load(f)


def plot_centroids():
    geom = GEOMETRY['centroids']
    for i in range(len(geom)):
        cent = geom[i]
        for j in cent:
            if len(cent) > 1:
                plt.plot(j[0]*PX_TO_MM, -j[1]*PX_TO_MM, 'rx')
            else:
                plt.plot(j[0]*PX_TO_MM, -j[1]*PX_TO_MM, 'k.')
        plt.xlim([0, 352*PX_TO_MM])
        plt.ylim([-288*PX_TO_MM, 0])
        plt.pause(.001)
        plt.cla()
    plt.show()


def plot_perimeters(cont=True):
    geom = GEOMETRY['perimeters']

    if cont:
        _, (ax1, ax2) = plt.subplots(
            2, 1)
        for i in range(len(geom)):
            ar = geom[i]
            if len(ar) > 1:
                for j in ar:
                    ax1.plot(i, j*PX_TO_MM, 'rx')
            elif len(ar) == 1:
                ax1.plot(i, ar[0]*PX_TO_MM, 'k.')
            elif len(ar) == 0:
                ax1.plot(i, 0, 'b*')
            ax2.imshow(cv2.imread('HSV Frames\\Test\\' +
                                  DATASET + '\\' + str(i) + '.jpg', cv2.IMREAD_GRAYSCALE))
            ax1.set_ylabel('Perimeter (mm)')
            plt.pause(.0001)
        plt.show()
    else:
        single_droplet, single_droplet_idx = ([], [])
        multi_droplet, multi_droplet_idx = ([], [])
        zero_droplet, zero_droplet_idx = ([], [])

        for i in range(len(geom)):
            ar = geom[i]
            if len(ar) > 1:
                for j in ar:
                    multi_droplet.append(j*PX_TO_MM)
                    multi_droplet_idx.append(i)
            elif len(ar) == 1:
                single_droplet.append(ar[0]*PX_TO_MM)
                single_droplet_idx.append(i)
            elif len(ar) == 0:
                zero_droplet.append(0)
                zero_droplet_idx.append(i)
        plt.plot(single_droplet_idx, single_droplet, 'k-.')
        plt.plot(multi_droplet_idx, multi_droplet, 'rx')
        plt.plot(zero_droplet_idx, zero_droplet, 'b*')

        plt.ylim([0, 500*PX_TO_MM])
        plt.tight_layout()
        plt.savefig('perimeters.pdf')
        plt.show()


def plot_areas(cont=True):
    geom = GEOMETRY['areas']
    if cont:
        _, (ax1, ax2) = plt.subplots(
            2, 1)
        for i in range(len(geom)):
            ar = geom[i]
            if len(ar) > 1:
                for j in ar:
                    ax1.plot(i, j*PX_TO_MM**2, 'rx')
            elif len(ar) == 1:
                ax1.plot(i, ar[0]*PX_TO_MM**2, 'k.')
            elif len(ar) == 0:
                ax1.plot(i, 0, 'b*')
            ax2.imshow(cv2.imread('HSV Frames\\Test\\' +
                                  DATASET + '\\' + str(i) + '.jpg', cv2.IMREAD_GRAYSCALE))
            ax1.set_ylabel(r'Area ($mm^2$)')
            ax1.set_ylim([0, 500*PX_TO_MM])
            plt.tight_layout()
            plt.savefig(f'Graphs\\{i}.jpg', format='jpg')
    else:

        single_droplet, single_droplet_idx = ([], [])
        multi_droplet, multi_droplet_idx = ([], [])
        zero_droplet, zero_droplet_idx = ([], [])
        dt = cp.time_period([P1, P2, P2, P2])
        for i in range(len(geom)):
            ar = geom[i]
            if len(ar) == 0:
                single_droplet.append(0)
            elif len(ar) > 1:
                single_droplet.append(sum(ar)*PX_TO_MM**2*0.01)
            else:
                single_droplet.append(max(ar)*PX_TO_MM**2*0.01)
            # if len(ar) > 1:
            #     for j in ar:
            #         multi_droplet.append(j*PX_TO_MM**2)
            #         multi_droplet_idx.append(i)
            # elif len(ar) == 1:
            #     single_droplet.append(ar[0]*PX_TO_MM**2)
            #     single_droplet_idx.append(i)
            # elif len(ar) == 0:
            #     zero_droplet.append(0)
            #     zero_droplet_idx.append(i)
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(16, 8))
        # smooth_droplet = smooth(np.array(single_droplet), 100)

        #peaks = scipy.signal.find_peaks(-smooth_droplet, height=-.04)
        # for i, peak in enumerate(peaks[0]):
        #     if i == 0:
        #         ax1.axvline(x=peak, color='b', alpha=0.4, label='detachment')
        #     else:
        #         ax1.axvline(x=peak, color='b', alpha=0.4)
        ax1.plot(range(len(single_droplet)), single_droplet,
                 'k-', alpha=.5, label='area')
        # ax1.plot(range(len(multi_droplet_idx)), multi_droplet,
        #          'rx', alpha=.5, label='area')
        # ax1.plot(range(len(zero_droplet_idx)), zero_droplet,
        #          'bx', alpha=.5, label='area')
        # # ax1.plot(range(len(smooth_droplet)), smooth_droplet,
        #          'k-', label='smoothed area')

        # plt.plot(multi_droplet_idx, multi_droplet, 'rx')
        # plt.plot(zero_droplet_idx, zero_droplet, 'b*')

        ax1.set_ylim([0, 2200*PX_TO_MM**2*0.01])
        ax1.set_ylabel(r'Area ($cm^2$)')
        ax1.legend()
        # plt.savefig('areas.pdf')

        return ax2  # , peaks


def plot_fft():
    xf, yf, N = compute_fft(plot_areas(cont=False))
    plt.plot(xf, 2.0/N * np.abs(yf[0:int(N/2)]), 'k-')
    plt.show()


def main():
    plot_areas(cont=False)


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

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

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError(
            "Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(), s, mode='valid')
    return y


if __name__ == "__main__":
    ax2 = plot_areas(cont=False)
    # vel = cp.compute_vel()
    # idx, vv = ([], [])

    # for i, v in enumerate(vel):
    #     try:
    #         vv.append(max(v))
    #         idx.append(i)
    #     except TypeError:
    #         vv.append(v)
    #         idx.append(i)
    # smooth_vv = smooth(np.array(vv), 100)

    # for i, peak in enumerate(peaks[0]):
    #     if i == 0:
    #         ax2.axvline(x=peak, color='b', alpha=0.4, label='detachment')
    #     else:
    #         ax2.axvline(x=peak, color='b', alpha=0.4)
    # ax2.plot(range(len(vv)), vv, 'r-', alpha=.5)
    # # ax2.plot(range(len(smooth_vv)), smooth_vv, 'k-')
    # ax2.legend()
    # plt.tight_layout()
    plt.show()
