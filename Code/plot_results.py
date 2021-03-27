'''
Functions that output plots and animations of
relevant geometric and kinematic droplet properties
using predicted segmentation maps.
'''
# %% IMPORTS
import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.signal import hann
from scipy import optimize
from scipy.fft import fft, fftfreq
from utils.postprocessing import parse_centroid_coords, smooth_signal, set_size, latex_plot_config, polynomial_function

# video settings
matplotlib.rcParams['animation.ffmpeg_path'] = os.path.abspath(
    'C:\\ffmpeg\\bin\\ffmpeg.exe')
# set style
sns.set_style('whitegrid')
matplotlib.rcParams["axes.unicode_minus"] = False
# %% VAIRABLES
# Pixel to distance conversions
# 26 px = .045 in = 1.143 mm
# 1px = 0.04396153846153846 mm = 4.396153846153846 * 10^(-5) m
#
# LaTex \textwidth = 472.03123 pt. Useful for figure sizing

ARCHITECTURE_NAME = 'unet'
DATASET = 'Globular'
N_FILTERS = 32
BATCH_SIZE_TRAIN = 16
EPOCHS = 200

DATA_DIR_RGB = os.path.join(
    'Data', 'Image', 'Input', f'{DATASET.lower()}_rgb.npz')
DATA_DIR_GRAY = os.path.join(
    'Data', 'Image', 'Input', f'{DATASET.lower()}_gray.npz')
PREDS_DIR = os.path.join('Output', 'Predictions',
                         f'{ARCHITECTURE_NAME.lower()}_{DATASET.lower()}_{N_FILTERS}_{BATCH_SIZE_TRAIN}_{EPOCHS}_preds.npz')
GEOMETRY_DIR = os.path.join('Output', 'Geometry',
                            f'{ARCHITECTURE_NAME.lower()}_{DATASET.lower()}_{N_FILTERS}_{BATCH_SIZE_TRAIN}_{EPOCHS}_geometry.pickle')

FPS = 3000.0
PERIOD = 1/FPS
PX_TO_MM = 1.143/26
OFFSET = 0

with open(GEOMETRY_DIR, 'rb') as f:
    GEOMETRY = pickle.load(f)

CENTROIDS = GEOMETRY['centroids_float'][OFFSET:]
AREAS = GEOMETRY['areas'][OFFSET:]
PERIMETERS = GEOMETRY['perimeters'][OFFSET:]
VOLUMES = GEOMETRY['volumes'][OFFSET:]
TIME = GEOMETRY['time'][OFFSET:]

WRITER = animation.writers['ffmpeg'](fps=30, bitrate=1800)
# %% FUNCTIONS


def plot_hanning(save=False):
    '''
    Shows a hanning window and a noisy sine wave smoothed by a hanning window through 1D convolution.
    '''
    window = hann(51)
    fig, axes = plt.subplots(1, 2, figsize=set_size(aspect_ratio=.3))
    axes[0].plot(window)
    axes[0].set_ylabel('Amplitude')
    axes[1].set_ylabel('Amplitude')
    x_val = np.linspace(0, 8, 100)
    y_val = np.abs(np.sin(x_val))
    y_noise = y_val+np.random.randn(len(x_val))*0.1
    y_smooth = smooth_signal(y_noise)
    axes[1].plot(x_val, y_noise, label='Noisy signal')
    axes[1].plot(x_val, y_smooth, label='Smoothed signal')
    plt.legend()
    if save:
        latex_plot_config()
        fig.savefig(os.path.join('Output', 'Plots', 'hanning.pgf'))
    else:
        plt.show()


def interpolate_outlier(arr, start, outlier_id, axis=1):
    '''
    doc
    '''
    outlier_copy = outlier_id.copy()
    vals, oids = [[], []]
    for oid in outlier_id:
        prev_id = oid-1
        next_id = oid+1
        while prev_id in outlier_copy:
            prev_id -= 1
        while next_id in outlier_copy:
            next_id += 1
        prev_id -= start
        next_id -= start
        oid -= start
        val = np.mean(
            [arr[prev_id], arr[next_id]], axis=axis)
        outlier_copy.pop(0)
        vals.append(val)
        oids.append(oid)
    return oids, vals


def compute_vel(start, end, outlier_id=None, save=False):
    '''
    doc
    '''
    perimeters = np.array(PERIMETERS[start:end])
    centroids = np.array(CENTROIDS[start:end])
    times = np.array(TIME[start:end])*10 ** (-6)
    perimeter_signal = property_signal(
        times, centroids, perimeters, offset=start)

    # interpolate outliers with mean value
    if outlier_id:
        oids, vals = interpolate_outlier(
            perimeter_signal[:, 2], start, outlier_id, axis=0)
        perimeter_signal[oids, 2] = vals

    x_pos, y_pos = parse_centroid_coords(perimeter_signal[:, 2])
    cents = np.array([x_pos, y_pos]).T*PX_TO_MM

    # fit cubic polynomial to x, y positions
    t_interp = np.linspace(times[0], times[-1], num=100)
    fit = np.polyfit(times, cents[:, 0], deg=3, full=True)
    x_coef = fit[0]
    r2x = 1 - fit[1][0]/((cents[:, 0]-cents[:, 0].mean())**2).sum()
    dx_coef = np.array([3*x_coef[0], 2*x_coef[1], x_coef[2]])
    ddx_coef = np.array([6*x_coef[0], 2*x_coef[1]])

    fit = np.polyfit(times, cents[:, 1], deg=3, full=True)
    y_coef = fit[0]
    r2y = 1 - fit[1][0]/((cents[:, 1]-cents[:, 1].mean())**2).sum()
    dy_coef = np.array([3*y_coef[0], 2*y_coef[1], y_coef[2]])
    ddy_coef = np.array([6*y_coef[0], 2*y_coef[1]])

    x_interp = polynomial_function(x_coef, t_interp)
    y_interp = polynomial_function(y_coef, t_interp)
    vx_interp = polynomial_function(dx_coef, t_interp)
    vy_interp = polynomial_function(dy_coef, t_interp)
    ax_interp = polynomial_function(ddx_coef, t_interp)
    ay_interp = polynomial_function(ddy_coef, t_interp)

    cents_interp = np.array([x_interp, y_interp]).T*1
    vel = np.array([vx_interp, vy_interp]).T
    acc = np.array([ax_interp, ay_interp]).T
    vel = vel*10**(-1)
    acc = acc*10**(-3)
    vel_n = np.linalg.norm(vel, axis=1)
    acc_n = np.linalg.norm(acc, axis=1)
    skip = 6

    fig1, axes = plt.subplots(1, 1, figsize=set_size(
        fraction=.8, aspect_ratio=296/352))
    axes.set_xlabel(r'$x$ coordinate [$mm$]')
    axes.set_ylabel(r'$y$ coordinate [$mm$]')
    axes.set_xlim([min(cents[:, 0])-0.5, max(cents[:, 0])+0.5])
    axes.set_ylim([min(cents[:, 1])-0.5, max(cents[:, 1])+0.5])
    axes.invert_yaxis()
    axes.plot(cents[:, 0], cents[:, 1], alpha=.7, linestyle='--')

    quiver_vel = axes.quiver(cents_interp[1:-1:skip, 0], cents_interp[1:-1:skip, 1], acc[1:-1:skip, 0], -
                             acc[1:-1:skip, 1], width=.0045, color='r', scale=400)

    axes.quiverkey(quiver_vel, 0.85, 0.2, 20, r'20 $\frac{{m}}{{s^2}}$',
                   coordinates='figure', labelpos='E')

    quiver_acc = axes.quiver(cents_interp[1:-1:skip, 0], cents_interp[1:-1:skip, 1], vel[1:-1:skip, 0], -
                             vel[1:-1:skip, 1], width=.0045, scale=500)
    axes.quiverkey(quiver_acc, 0.65, 0.2, 30, r'$30 \frac{cm}{s}$',
                   coordinates='figure', labelpos='E')

    axes.scatter(cents[0, 0], cents[0, 1], marker='^',
                 color='xkcd:cerulean', label='Detachment', s=100)
    axes.scatter(cents[-1, 0], cents[-1, 1], marker='*',
                 color='xkcd:orange', label='Landing', s=150)
    plt.legend()

    delta_t = (perimeter_signal[-1, 1]-perimeter_signal[0, 1])*1000
    title = fr'{DATASET}, Frames {start}-{end}, $\Delta t = {delta_t:.2f}\;ms$'
    plt.title(title)
    plt.tight_layout()

    print(f'Frames {start}-{end}')
    print('vmin, vmax:', np.min(vel, axis=0), np.max(vel, axis=0))
    print('vmean, vstd:', np.mean(vel, axis=0), np.std(vel, axis=0))
    print('vn_min, vn_max:', np.min(vel_n), np.max(vel_n))
    print('vn_mean, vn_std:', np.mean(vel_n), np.std(vel_n))
    print('\n')
    print('amin, amax:', np.min(acc, axis=0), np.max(acc, axis=0))
    print('amean, astd:', np.mean(acc, axis=0), np.std(acc, axis=0))
    print('an_min, an_max:', np.min(acc_n), np.max(acc_n))
    print('an_mean, an_std:', np.mean(acc_n), np.std(acc_n))
    print('\n')

    fig2, axes2 = plt.subplots(2, 1, figsize=set_size())
    plt.title(title)
    axes2[0].plot(t_interp, x_interp, color='xkcd:orange',
                  label=fr'Cubic ($R^2={r2x:.3f}$)')
    axes2[0].scatter(times, cents[:, 0], marker='x', label='Measured')
    axes2[0].legend()
    axes2[1].plot(t_interp, y_interp, color='xkcd:orange',
                  label=fr'Cubic ($R^2={r2y:.3f}$)')
    axes2[1].scatter(times, cents[:, 1], marker='x', label='Measured')
    axes2[1].legend()
    fig2.tight_layout()

    fig3, axes3 = plt.subplots(2, 1, figsize=set_size())
    plt.title(title)
    axes33 = axes3[0].twinx()
    axes34 = axes3[1].twinx()

    axes3[0].set_xlabel(r'Time [$s$]')
    axes3[0].set_ylabel(r'Velocity [$\frac{cm}{s}$]')
    axes33.set_ylabel(r'Acceleration [$\frac{m}{s^2}$]')
    axes3[1].set_xlabel(r'Time [$s$]')
    axes3[1].set_ylabel(r'Velocity [$\frac{cm}{s}$]')
    axes34.set_ylabel(r'Acceleration [$\frac{m}{s^2}$]')

    plot_1 = axes3[0].plot(t_interp, vel[:, 0],
                           color='xkcd:orange', label=r'$v_x$')
    plot_2 = axes33.plot(t_interp, acc[:, 0], label=r'$a_y$')
    plots = plot_1 + plot_2
    labs = [l.get_label() for l in plots]
    axes3[0].legend(plots, labs)
    axes33.grid(b=None)

    plot_1 = axes3[1].plot(t_interp, vel[:, 1],
                           color='xkcd:orange', label=r'$v_y$')
    plot_2 = axes34.plot(t_interp, acc[:, 1], label=r'$a_y$')
    plots = plot_1+plot_2
    labs = [l.get_label() for l in plots]
    axes3[1].legend(plots, labs)
    axes34.grid(b=None)
    fig3.tight_layout()

    fig4, axes4 = plt.subplots(1, 1, figsize=set_size())
    plt.title(title)
    axes43 = axes4.twinx()

    axes4.set_xlabel(r'Time [$s$]')
    axes4.set_ylabel(r'Velocity [$\frac{cm}{s}$]')
    axes43.set_ylabel(r'Acceleration [$\frac{m}{s^2}$]')

    plot_1 = axes4.plot(t_interp, vel_n,
                        color='xkcd:orange', label=r'$\|v\|$')
    plot_2 = axes43.plot(t_interp, acc_n, label=r'$\|a\|$')
    plots = plot_1 + plot_2
    labs = [l.get_label() for l in plots]
    axes4.legend(plots, labs)
    axes43.grid(b=None)

    fig4.tight_layout()

    if save:
        figs = [fig1, fig2, fig3, fig4]
        names = ['trajectory', 'interp', 'vel_acc_coords', 'vel_acc_norm']
        for fig, name in zip(figs, names):
            fig.savefig(os.path.join('Output', 'Plots',
                                     'kinematic', f'{name}_{start}-{end}.png'), transparent=True, dpi=300)
        latex_plot_config()
        for fig, name in zip(figs, names):
            fig.savefig(os.path.join('Output', 'Plots',
                                     'kinematic', f'{name}_{start}-{end}.pgf'))
    else:
        plt.show()


def compute_freq(start, end, outlier_id=None, save=False):
    '''
    Takes a frame window from start to end and applies fft to the area.

    Args:
    start {int} -- Starting frame
    end {int} -- last frame

    Kwargs:
    save {bool} -- Whether to save the plot as .pgf if True or show them using plt.show() if False
    '''
    if save:
        latex_plot_config()

    perimeters = np.array(PERIMETERS[start:end])
    volumes = np.array(VOLUMES[start:end])
    centroids = np.array(CENTROIDS[start:end])
    times = np.array(TIME[start:end])*10 ** (-6)

    perimeter_signal = property_signal(
        times, centroids, perimeters, offset=start)
    volume_signal = property_signal(times, centroids, volumes, offset=start)

    if outlier_id:
        outlier_copy = outlier_id.copy()
        for oid in outlier_id:
            prev_id = oid-1
            next_id = oid+1
            while prev_id in outlier_copy:
                prev_id -= 1
            while next_id in outlier_copy:
                next_id += 1
            prev_id -= start
            next_id -= start
            oid -= start
            perimeter_signal[oid, 3] = np.mean(
                [perimeter_signal[prev_id, 3], perimeter_signal[next_id, 3]])
            volume_signal[oid, 3] = np.mean(
                [volume_signal[prev_id, 3], volume_signal[next_id, 3]])
            new_cent = np.mean([perimeter_signal[prev_id, 2],
                                perimeter_signal[next_id, 2]], axis=0)
            perimeter_signal[oid, 2] = new_cent
            volume_signal[oid, 2] = new_cent
            outlier_copy.pop(0)
    zero_id = perimeter_signal[:, 4] == 'zero'
    single_id = perimeter_signal[:, 4] == 'single'
    multi_id = perimeter_signal[:, 4] == 'multi'

    fig, axes = plt.subplots(2, 1, figsize=set_size(aspect_ratio=.6))

    axes[0].set_xlabel(r'Frame')
    axes[0].set_ylabel(r'Perimeter [$mm$]')
    axes[0].set_title(f'Frames {start}-{end}')

    axes[0].scatter(perimeter_signal[zero_id, 0], perimeter_signal[zero_id, 3], s=12,
                    marker='o', color='b')
    axes[0].scatter(perimeter_signal[single_id, 0], perimeter_signal[single_id, 3], s=12, marker='.',
                    color='k')

    axes[0].scatter(perimeter_signal[multi_id, 0], perimeter_signal[multi_id, 3], s=9, marker='x',
                    color='r')
    new_ticks = axes[0].get_xticks().astype(int)[1:-1]

    def tick_function(i):
        del i
        arr = np.array(TIME)[new_ticks]*10**(-6)
        return ['%.3f' % i for i in arr]
    ax0 = axes[0].twiny()

    ax0.set_xlim(axes[0].get_xlim())
    ax0.set_xticks(new_ticks)
    ax0.set_xticklabels(tick_function(new_ticks))
    ax0.set_xlabel(r'Time [$s$]')
    plt.grid(b=None)

    n_samples = len(times)
    period = 1/3000
    y_fft = fft(perimeter_signal[:, 3])
    magnitude = 2.0/n_samples * np.abs(y_fft[0: n_samples//2])
    magnitude_s = magnitude.copy()
    magnitude_s.sort()
    second_max = magnitude_s[-2]
    idx = int(np.where(magnitude == second_max)[0])
    x_fft = fftfreq(n_samples, period)[: n_samples//2]

    axes[1].set_xlabel('Frequency [$Hz$]')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_ylim([-.01, magnitude[idx]*1.1])
    axes[1].plot(x_fft[idx:], magnitude[idx:])
    axes[1].scatter(x_fft[idx], magnitude[idx], color='purple',
                    marker='*', s=30, label=fr'$f={x_fft[idx]:.2f}$ $Hz$')

    def sine_func(x_arr, coef_0, coef_1, coef_2, coef_3):
        return coef_0+coef_1*np.sin(2*np.pi*coef_2*x_arr.astype(float)+coef_3)
    params, *_ = optimize.curve_fit(sine_func, perimeter_signal[:, 1], perimeter_signal[:, 3],
                                    p0=[10, 1, 0, x_fft[idx]])

    vol = np.array([np.mean(volume_signal[:, 3]), np.std(volume_signal[:, 3])])
    rho = 6500
    tau = np.array([1/x_fft[idx], 1/params[3]])
    constant = (3*np.pi/8)*rho/(tau**2)*10**(-9)
    sigma = vol.reshape(2, 1)@constant.reshape(1, 2)
    best_sigma = sigma[:, np.argmin(np.abs(sigma[0, :]-1))]
    worst_sigma = sigma[:, np.argmax(np.abs(sigma[0, :]-1))]
    print([start, end], vol, best_sigma, worst_sigma, params[3])

    axes[1].legend()
    fig.tight_layout()

    if save:
        latex_plot_config()
        fig.savefig(os.path.join('Output', 'Plots',
                                 'fourier', f'{DATASET.lower()}_fft_{start}-{end}.pgf'))

    else:
        plt.show()


def property_signal(times, cents, props, offset, init_id=0):
    '''
    doc
    '''
    signal = []
    for i, tcp in enumerate(zip(times, cents, props)):
        time, cent, prop = tcp
        if len(prop) == 0:
            dtype = 'zero'
        elif len(prop) == 1:
            dtype = 'single'
        else:
            dtype = 'multi'

        if i == 0:
            signal.append(
                [i+offset, time, cent[init_id], prop[init_id], dtype])
            continue
        if len(prop) == 0:
            signal.append([i+offset, time,  [], 0, dtype])
        elif len(prop) == 1:
            signal.append([i+offset, time,  *cent, *prop, dtype])
        else:
            cent_0 = signal[i-1][2]
            distance = []
            for cent_1 in cent:
                distance.append(np.linalg.norm(np.array(cent_1)-cent_0))
            idx = np.argmin(distance)
            signal.append([i+offset, time, cent[idx], prop[idx], dtype])
    signal = np.array(signal)
    return signal


def split_property(times, props, offset, zero_id=True):
    '''
    doc
    '''
    out = []
    for i, time_prop in enumerate(zip(times, props)):
        time, prop = time_prop
        if len(prop) == 0:
            if zero_id:
                out.append([i+offset, time, 0])
            else:
                out.append([i+offset, time, None])
        elif len(prop) == 1:
            out.append([i+offset, time, *prop])
        else:
            for prop_i in prop:
                out.append([i+offset, time, prop_i])
    out = np.array(out)
    return out[:, 0], out[:, 1], out[:, 2]


if __name__ == "__main__":
    # free_flight_timestamps = np.array([(34, 72), (289, 342), (1439, 1502, [1466]), (2279, 2341), (2775, 2859), (4197, 4253), (4553, 4577), (4761, 4853),
    #                                    (4983, 5031), (5278, 5311), (5732, 5816), (6111, 6146), (6589, 6653), (7284, 7337), (7754, 7817), (8795, 8871)])
    free_flight_timestamps = np.array([(26, 34)])
    for timestamps in free_flight_timestamps[:4]:
        matplotlib.use('Qt5Agg')
        compute_vel(*timestamps, save=False)
