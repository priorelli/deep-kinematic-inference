import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pylab import tight_layout
import time
import sys
import config as c
from environment.arm import Arm


def record_video(log, width):
    # Initialize arm
    arm = Arm()

    # Load variables
    n_t = log['angles'].shape[0] * log['angles'].shape[1]

    angles = log['angles'].reshape(n_t, c.n_joints)
    est_angles = log['est_angles'].reshape(n_t, c.n_joints)

    pos = log['pos'].reshape(n_t, c.n_joints + 1, 2)
    forward = log['est_forward'].reshape(n_t, c.n_joints + 1, 2)

    target_pos = log['target_pos'].reshape(n_t, 2)
    obstacle_pos = log['obstacle_pos'].reshape(n_t, 2)

    # Create plot
    fig, axs = plt.subplots(1, figsize=(20, 15))

    def animate(n):
        if (n + 1) % 10 == 0:
            sys.stdout.write('\rTrial: {:d} \tStep: {:d}'
                             .format(int(n / c.n_steps) + 1,
                                     (n % c.n_steps) + 1))
            sys.stdout.flush()

        # Clear plot
        axs.clear()
        axs.get_xaxis().set_visible(False)
        axs.get_yaxis().set_visible(False)
        axs.set_xlim(-130, 270)
        axs.set_ylim(-50, 250)
        tight_layout()

        # Draw arm
        for j in range(c.n_joints):
            axs.plot(*np.array([forward[n, arm.idxs[j] + 1], pos[n, j + 1]]).T,
                     linewidth=arm.size[j, 1] * 3.5, color='lightblue',
                     zorder=1)
            axs.plot(*np.array([pos[n, arm.idxs[j] + 1], pos[n, j + 1]]).T,
                     linewidth=arm.size[j, 1] * 3.5, color='b',
                     zorder=1)

        # Draw target
        if c.task in ['reach', 'both']:
            t_size = c.target_size * 400
            axs.scatter(*target_pos[n], color='r', s=t_size, zorder=0)

        # Draw obstacle
        if c.task in ['avoid', 'both']:
            o_size = c.target_size * 400
            axs.scatter(*obstacle_pos[n], color='g', s=o_size, zorder=0)

        # Draw trajectories
        if c.task in ['reach', 'both']:
            axs.scatter(*target_pos[n - (n % c.n_steps): n + 1].T,
                        color='darkred', linewidth=width + 2, zorder=2)
        if c.task in ['avoid', 'both']:
            axs.scatter(*obstacle_pos[n - (n % c.n_steps): n + 1].T,
                        color='darkgreen', linewidth=width + 2, zorder=2)
        axs.scatter(*forward[n - (n % c.n_steps): n + 1].T,
                    color='cyan', linewidth=width + 2, zorder=2)
        axs.scatter(*pos[n - (n % c.n_steps): n + 1].T,
                    color='darkblue', linewidth=width + 2, zorder=2)

    start = time.time()
    ani = animation.FuncAnimation(fig, animate, n_t, interval=60)
    # writer = animation.writers['ffmpeg'](fps=300)
    # ani.save('plots/video.mp4', writer=writer)
    # print('\nTime elapsed:', time.time() - start)
    animate(c.n_steps - 400)
    plt.savefig('plots/frame')
