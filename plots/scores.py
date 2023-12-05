import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import utils


def plot_scores():
    ylabels = ['Accuracy (%)', 'Error (px)', 'Time']
    measures = ['acc', 'error', 'time']
    limits = [(80, 100), (0, 3.3), (0, 200)]
    orders_model = ['transp', 'pinv', 'hier', 'deep']
    orders_deep = ['1', '2', '3', '4', '5', '6', '7', '8', '9',
                   '10', '11', '12']

    display_model(measures, ylabels, limits, orders_model)
    # display_deep(measures, ylabels, limits, orders_deep)
    # display_length(measures, ylabels, limits)


def display_model(measures, ylabels, limits, order):
    score = {'model': [], 'task': [], 'acc': [], 'error': [], 'time': []}

    fig, axs = plt.subplots(1, 3, num='model', figsize=(24, 8))
    # fig.set_figwidth(24)

    for log_name in sorted(glob.glob('simulation/log_*.npz')):
        # Load log
        log = np.load(log_name)
        val = re.findall(r'log_(.*)_(.*).npz', log_name)[0]

        if val[1] == 'reach':
            data = utils.get_score(log['target_pos'], log['pos'], mean=False)
        else:
            data = utils.get_score(log['target_pos'], log['est_forward'],
                                   mean=False)

        for i in range(len(data[0])):
            score['model'].append(val[0])
            score['task'].append(val[1])
            for m, measure in enumerate(measures):
                x = data[m][i] if i < len(data[m]) else \
                    np.mean(data[m])
                score[measure].append(x)

    for j, ylabel, measure, limit in zip(range(3), ylabels, measures,
                                         limits):
        axs[j].set_xlabel('Model')
        axs[j].set_ylabel(ylabel)
        axs[j].set_ylim(*limit)
        axs[j].yaxis.set_major_locator(plt.MaxNLocator(4))

        sns.barplot(x='model', y=measure, ax=axs[j], hue='task',
                    data=score, palette=('b', 'r'), order=order)

    plt.tight_layout()
    plt.savefig('plots/plot_model', bbox_inches='tight')
    plt.close()


def display_deep(measures, ylabels, limits, order):
    fig, axs = plt.subplots(2, 3, num='model', figsize=(30, 14))

    for t, task, color in zip(range(2), ['infer', 'reach'], ['Blues_d', 'Reds_d']):
        score = {'DoFs': [], 'acc': [], 'error': [], 'time': []}

        for log_name in sorted(glob.glob('simulation/log_%s_*.npz' % task)):
            # Load log
            log = np.load(log_name)
            val = re.findall(r'log_%s_(.*).npz' % task, log_name)[0]

            if task == 'reach':
                data = utils.get_score(log['target_pos'], log['pos'],
                                       mean=False)
            else:
                data = utils.get_score(log['target_pos'], log['est_forward'],
                                       mean=False)

            for i in range(len(data[0])):
                score['DoFs'].append(val)
                for m, measure in enumerate(measures):
                    x = data[m][i] if i < len(data[m]) else \
                        np.mean(data[m])
                    score[measure].append(x)

        for j, ylabel, measure, limit in zip(range(3), ylabels, measures,
                                             limits):
            axs[t, j].set_xlabel('DoFs')
            axs[t, j].set_ylabel(ylabel)
            axs[t, j].set_ylim(*limit)

            sns.barplot(x='DoFs', y=measure, ax=axs[t, j],
                        data=score, palette=color, order=order)

    plt.savefig('plots/plot_deep', bbox_inches='tight')
    plt.close()


def display_length(measures, ylabels, limits):
    score = {'val': [], 'acc': [], 'error': [], 'time': []}

    fig, axs = plt.subplots(1, 3, num='model', figsize=(24, 8))

    # Load log
    log = np.load('simulation/log_length_infer.npz')

    # print(utils.get_score_length(log['est_lengths']))

    for val in ['length', 'position']:
        if val == 'position':
            data = utils.get_score(log['target_pos'], log['est_forward'],
                                   mean=False)
        else:
            data = list(utils.get_score_length(log['est_lengths'], mean=False))

        for i in range(len(data[0])):
            score['val'].append(val)
            for m, measure in enumerate(measures):
                x = data[m][i] if i < len(data[m]) else \
                    np.mean(data[m])
                score[measure].append(x)

    for j, ylabel, measure, limit in zip(range(3), ylabels, measures,
                                         limits):
        axs[j].set_xlabel('Measure')
        axs[j].set_ylabel(ylabel)
        axs[j].set_ylim(*limit)

        sns.barplot(x='val', y=measure, ax=axs[j])

    plt.tight_layout()
    plt.savefig('plots/plot_length', bbox_inches='tight')
    plt.close()