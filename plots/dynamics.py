import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas
import config as c


def plot_dynamics(log, width):
    # Load variables
    actions = log['actions'][-1]

    angles = log['angles'][-1]
    est_angles = log['est_angles'][-1]

    pos = log['pos'][-1]
    est_pos = log['est_pos'][-1]

    phi = log['phi'][-1]
    est_phi = log['est_phi'][-1]

    est_lengths = log['est_lengths'][-1]
    est_forward = log['est_forward'][-1]

    target_pos = log['target_pos'][-1]
    obstacle_pos = log['obstacle_pos'][-1]

    # Plots
    e_angles = np.abs(angles - est_angles)
    e_pos = np.linalg.norm(est_forward - pos, axis=2)
    e_final_target = np.linalg.norm(pos[:, -1] - target_pos, axis=1)
    e_final_obstacle = np.linalg.norm(pos[:, -1] - obstacle_pos, axis=1)
    e_lengths = np.abs(c.lengths - est_lengths)
    e_phi = np.abs(est_phi - phi)

    fig, axs = plt.subplots(3, 3, figsize=(48, 32))

    colors = ['Blue', 'Red', 'Green', 'Purple']
    cmaps = [cm.get_cmap('Blues_r'), cm.get_cmap('Reds_r'),
             cm.get_cmap('Greens_r'), cm.get_cmap('Purples_r')]

    for j in range(c.n_joints):
        axs[0, 0].set_title('Position')
        axs[0, 0].set_xlim(-c.width / 2, c.width / 2)
        axs[0, 0].set_ylim(-c.height / 2, 240)
        axs[0, 0].set_xlabel('x')
        axs[0, 0].set_ylabel('y')
        axs[0, 0].scatter(*pos[:, j + 1].T, c=np.arange(c.n_steps),
                          cmap=cmaps[j])
        # axs[0, 0].plot(*obstacle_pos.T, marker='o', markersize=20,
        #                color='g', markerfacecolor='darkgreen')
        axs[0, 0].plot(*target_pos.T, marker='o', markersize=20,
                       color='r', markerfacecolor='darkred')

        # axs[0, 1].set_title('Forward')
        # axs[0, 1].set_xlim(-c.width / 2, c.width / 2)
        # axs[0, 1].set_ylim(-c.height / 2, c.height / 2)
        # axs[0, 1].scatter(*est_forward[:, j + 1].T, c=np.arange(c.n_steps),
        #                   cmap=cmaps[j])
        # axs[0, 1].plot(*target_pos.T, marker='o', markersize=20,
        #                markeredgecolor='red', markerfacecolor='green')

        axs[0, 2].set_title('Position belief')
        axs[0, 2].set_xlim(-c.width / 2, c.width / 2)
        axs[0, 2].set_ylim(-c.height / 2, c.height / 2)
        axs[0, 2].scatter(*est_pos[:, j + 1].T, c=np.arange(c.n_steps),
                          cmap=cmaps[j])
        axs[0, 2].plot(*target_pos.T, marker='o', markersize=20,
                       markerfacecolor='green')

        axs[2, 1].set_title('Lengths belief')
        axs[2, 1].plot(est_lengths[:, j], linewidth=width)

        # axs[0, 1].set_title('Action')
        # axs[0, 1].plot(actions[:, j], linewidth=width, c=colors[j])

        axs[1, 1].set_title('Lengths error')
        axs[1, 1].set_xlabel('t')
        axs[1, 1].set_ylabel('Length (px)')
        axs[1, 1].plot(e_lengths[:, j], linewidth=width)

        axs[1, 0].set_title('Joint error')
        axs[1, 0].set_xlabel('t')
        axs[1, 0].set_ylabel('Angle (°)')
        axs[1, 0].plot(e_angles[:, j], linewidth=width, c=colors[j])

        # axs[0, 1].set_title('Joint belief')
        # axs[0, 1].set_xlabel('t')
        # axs[0, 1].set_ylabel('Angle (°)')
        # axs[0, 1].plot(est_angles[:, j], linewidth=width, c=colors[j])

        # axs[1, 1].set_title('Position error')
        # axs[1, 1].plot(e_pos[:, j], linewidth=width, c=colors[j])

        axs[1, 2].set_title('Phi error')
        axs[1, 2].plot(e_phi[:, j], linewidth=width, c=colors[j])

        # axs[0, 1].set_title('Orientation belief')
        # axs[0, 1].set_xlabel('t')
        # axs[0, 1].set_ylabel('Angle (°)')
        # axs[0, 1].plot(est_phi[:, j], linewidth=width, c=colors[j])

    axs[0, 1].set_title('Final error')
    axs[0, 1].set_xlabel('t')
    axs[0, 1].set_ylabel('L2 Norm (px)')
    axs[0, 1].set_ylim(0, 250)
    axs[0, 1].plot(e_final_target, linewidth=width, color='r')
    axs[0, 1].plot(e_final_obstacle, linewidth=width, color='g')
    axs[0, 1].plot(np.full(c.n_steps, 7.5), linewidth=width + 5,
                   linestyle='--')

    # axs[0, 1].set_title('Obstacle error')
    # axs[0, 1].set_xlabel('t')
    # axs[0, 1].set_ylabel('L2 Norm (px)')
    # axs[0, 1].set_ylim(0, 250)
    # for j in range(c.n_joints):
    #     e_obstacle = np.linalg.norm(pos[:, j+1] - obstacle_pos, axis=1)
    #     axs[0, 1].plot(e_obstacle, linewidth=width)
    # axs[0, 1].plot(np.full(c.n_steps, 7.5), linewidth=width + 5,
    #                linestyle='--')

    plt.tight_layout()
    fig.savefig('plots/plot_dynamics', bbox_inches='tight')
    # plt.show()


def plot_length(log, width):
    # Load variables
    target_pos = log['target_pos']
    est_forward = log['est_forward']

    est_lengths = log['est_lengths']

    e_pos = np.linalg.norm(target_pos - est_forward[:, :, -1], axis=2)
    e_lengths = []
    for j in range(c.n_joints):
        e_lengths += list(np.abs(c.lengths[j] - est_lengths[:, :, j]))

    # Create plot
    labels = ['angles', 'lengths']
    fig, ax = plt.subplots(num='dynamics', figsize=(20, 15))

    df = {'val': [], 'ep': [], 't': [], 'L2 Norm (px)': []}
    for w, error, title in zip(range(2), [e_pos, e_lengths], labels):

        for e, line in enumerate(error):
            for i, val in enumerate(line):
                df['val'].append(title)
                df['ep'].append(e)
                df['t'].append(i)
                df['L2 Norm (px)'].append(val)

    data = pandas.DataFrame.from_dict(df)

    for label, color in zip(labels, ['blue', 'green']):
        df = data.loc[data['val'] == label]
        y_mean = df.groupby('t').mean()['L2 Norm (px)']
        x = y_mean.index

        y_std = df.groupby('t').std()['L2 Norm (px)']
        error = (y_std * 3) / np.sqrt(c.n_trials)
        lower = y_mean - error
        upper = y_mean + error

        ax.plot(x, y_mean, color=color[0], label=label, linewidth=8)
        ax.plot(x, lower, color='tab:' + color, alpha=0.1)
        ax.plot(x, upper, color='tab:' + color, alpha=0.1)
        ax.fill_between(x, lower, upper, alpha=0.2)
        ax.set_xlabel('t')
        ax.set_ylabel('L2 Norm (px)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    ax.set_xlim(-50, 1000)
    ax.set_ylim(-7, 90)
    ax.legend()

    ax.plot(np.arange(-100, 1000), np.full(c.n_steps + 100, 8), '--',
            linewidth='5', color='grey')
    ax.plot(np.arange(-100, 1000), np.full(c.n_steps + 100, 0), color='black')
    ax.axvline(0, color='black')

    fig.savefig('plots/plot_length', bbox_inches='tight')
