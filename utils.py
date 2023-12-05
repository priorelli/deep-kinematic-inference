import numpy as np
from numpy.linalg import norm
import argparse
import sys
import config as c


# Add Gaussian noise to array
def add_gaussian_noise(array, noise):
    sigma = noise ** 0.5
    return array + np.random.normal(0, sigma, np.shape(array))


# Normalize data
def normalize(x, limits):
    limits = np.array(limits)
    x_norm = (x - limits[0]) / (limits[1] - limits[0])
    x_norm = x_norm * 2 - 1
    return x_norm


# Denormalize data
def denormalize(x, limits):
    limits = np.array(limits)
    x_denorm = (x + 1) / 2
    x_denorm = x_denorm * (limits[1] - limits[0]) + limits[0]
    return x_denorm


# Parse arguments for simulation
def get_sim_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--manual-control',
                        action='store_true', help='Start manual control')
    parser.add_argument('-s', '--shallow',
                        action='store_true', help='Start shallow model')
    parser.add_argument('-d', '--deep',
                        action='store_true', help='Start deep model')
    parser.add_argument('-j', '--jacobian',
                        action='store_true', help='Start jacobian model')
    parser.add_argument('-a', '--ask-params',
                        action='store_true', help='Ask parameters')

    args = parser.parse_args()
    return args


# Parse arguments for plots
def get_plot_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dynamics',
                        action='store_true', help='Plot dynamics')
    parser.add_argument('-v', '--video',
                        action='store_true', help='Record video')
    parser.add_argument('-s', '--scores',
                        action='store_true', help='Plots scores')

    args = parser.parse_args()
    return args


# Compute score
def get_score(ground, est, mean=True):
    error = norm(ground - est[:, :, -1], axis=2)
    f_error = error[:, -1]
    acc = (f_error < c.reach_dist) * 100

    time, std = [], []
    for ep in error:
        reached = np.where(ep < c.reach_dist)
        if reached[0].size > 0:
            time.append(reached[0][0])
            std.append(np.std(ep[reached[0][0]:]))

    if mean:
        return np.mean(acc), np.mean(f_error), np.mean(time), np.mean(std)
    else:
        return acc, f_error, time, std


def get_score_length(est, mean=True):
    t_acc, t_err, t_time, t_std = [], [], [], []

    for j in range(c.n_joints):
        error = np.abs(c.lengths[j] - est[:, :, j])
        t_err += list(error[:, -1])
        t_acc += list((error[:, -1] < c.reach_dist) * 100)

        for ep in error:
            reached = np.where(ep < c.reach_dist)
            if reached[0].size > 0:
                t_time.append(reached[0][0])
                t_std.append(np.std(ep[reached[0][0]:]))

    if mean:
        return np.mean(t_acc), np.mean(t_err), np.mean(t_time), np.mean(t_std)
    else:
        return t_acc, t_err, t_time, t_std


# Print score
def print_score(log, time):
    score = np.array((get_score(log.target_pos, log.pos),
                      get_score(log.target_pos, log.est_pos),
                      get_score(log.target_pos, log.est_forward),
                      get_score_length(log.est_lengths)))

    print('\n' + '=' * 30)
    print('\t\tReal Pos\t\tEst Pos\t\tForward Pos\t\tLength')
    for m, measure in enumerate(('Acc', 'Error', 'Time', 'Std')):
        print('{:s}\t\t{:.2f}\t\t\t{:.2f}\t\t{:.2f}\t\t\t{:.2f}'.format(
            measure, *score.T[m]))
    print('Successful trials: {:.2f}%'.format(log.success * 100 / c.n_trials))
    print('Time elapsed: {:.2f}s'.format(time))


# Print simulation info
def print_info(trial, success, step):
    sys.stdout.write('\rTrial: {:4d}({:4d})/{:d} \t '
                     'Step: {:4d}/{:d}'
                     .format(trial, int(success), c.n_trials,
                             step, c.n_steps))
    sys.stdout.flush()


# Print inference info
def print_inference(trial, step, log, arm):
    e_a = np.abs(log.angles[trial, step] - log.est_angles[trial, step])
    e_l = np.abs(arm.size[:, 0] - log.est_lengths[trial, step])

    e_r = np.abs(log.phi[trial, step] - log.est_phi[trial, step])

    # e_p = norm(log.pos[trial, step] - log.est_pos[trial, step], axis=1)
    e_p = norm(log.pos[trial, step] - log.est_forward[trial, step], axis=1)

    sys.stdout.write('\rTime: {:4d}/{:2d}'
                     '  |  Angles: {:+6.1f} {:+6.1f} {:+6.1f} {:+6.1f}'
                     '  |  Phi: {:+6.1f} {:+6.1f} {:+6.1f} {:+6.1f}'
                     '  |  Lengths: {:+6.1f} {:+6.1f} {:+6.1f} {:+6.1f}'
                     '  |  Positions: {:+6.1f} {:+6.1f} {:+6.1f} {:+6.1f}'
                     .format(step, trial, *e_a, *e_r, *e_l, *e_p[1:]))
    sys.stdout.flush()
