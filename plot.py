import seaborn as sns
import numpy as np
import utils
import config as c
from plots.dynamics import plot_dynamics, plot_length
from plots.video import record_video
from plots.scores import plot_scores

sns.set_theme(style='darkgrid', font_scale=2.5)


def main():
    width = 3

    # Parse arguments
    options = utils.get_plot_options()

    # Choose plot to display
    if options.dynamics:
        log = np.load('simulation/log_' + c.log_name + '.npz')
        plot_dynamics(log, width)
        # plot_length(log, width)

    elif options.video:
        log = np.load('simulation/log_' + c.log_name + '.npz')
        record_video(log, width)

    elif options.scores:
        plot_scores()


if __name__ == '__main__':
    main()
