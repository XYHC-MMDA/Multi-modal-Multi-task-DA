import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

plt_colors = iter(['c', 'r', 'g', 'b', 'y', 'k', 'm', '#2A0134', '#FF00FF', '#800000'])
plt_markers = iter(['*', '.', 'o', '^', 'v', '<', '>', '1', '2', '3', '4', 's', 'p', ','])
font = {
    'size': 18
}

src_domain, tgt_domain = 'usa', 'sng'
sub_dir = 'src_ctr_usa_v1'
log_file1 = f'../checkpoints/fusion_consis/xmuda/{sub_dir}/log.log'  # to add: tgt_val, src_val
log_file2 = f'../checkpoints/fusion_consis/xmuda/{sub_dir}/source_test.log'
log_file3 = f'../checkpoints/fusion_consis/xmuda/{sub_dir}/target_test.log'
log_files = [log_file1, log_file2]


def plot_train(log_file, fig, epochs=24):
    accf = open(log_file, 'r')
    lines = accf.readlines()
    accf.close()
    losses = defaultdict(list)
    for line in lines:
        if 'Epoch' in line:
            splits = line.split(', ')
            for token in splits:
                for loss_type in ['contrast_loss', 'seg_loss', 'tgt_contrast_loss']:
                    if token.startswith(loss_type):
                        losses[loss_type].append(float(token.split(': ')[1]))
        elif 'seg_loss' in line:
            losses['seg_loss'].append(float(line.split(': ')[1]))
    if 'contrast_loss' in losses.keys():
        losses['src_contrast_loss'] = losses['contrast_loss']
        del losses['contrast_loss']
    for k, v in losses.items():
        x_len = len(v)
        break
    # if x_len > epochs:  # train loss
    xs = np.arange(x_len) + 1
    ax = fig.add_subplot(111, label='1')
    ax.set_xlabel('train iter', font)
    ax.set_ylabel('train loss', font)
    for i, (key, val) in enumerate(losses.items()):
        ys = np.array(val)
        # plt.plot(xs, ys, label=key, color=next(plt_colors), linewidth=1.)
        ax.plot(xs, ys, label=key, color=next(plt_colors), linewidth=1.)  # train curve labels are covered; weird
    # plt.xticks(range(0, x_len, x_len // epochs), labels=range(1, epochs+1))
    ax.set_xticks(range(0, x_len, x_len // epochs))  # labels=range(1, epochs+1))


def plot_test(log_file, fig, epochs=24):
    accf = open(log_file, 'r')
    lines = accf.readlines()
    accf.close()
    losses = defaultdict(list)
    for line in lines:
        if 'seg_loss' in line:
            losses['seg_loss'].append(float(line.split(': ')[1]))
    for k, v in losses.items():
        x_len = len(v)
        break
    # if x_len > epochs:  # train loss
    xs = np.arange(x_len) + 1
    ax = fig.add_subplot(111, label='2', frame_on=False)
    ax.xaxis.tick_top()
    ax.yaxis.tick_right()
    ax.set_xlabel('test epoch', font)
    ax.set_ylabel('test loss', font)
    ax.xaxis.set_label_position('top')
    ax.yaxis.set_label_position('right')
    for i, (key, val) in enumerate(losses.items()):
        ys = np.array(val)
        ax.plot(xs, ys, label=key, color=next(plt_colors), linewidth=1.)
    ax.set_xticks(xs)  # labels=range(1, epochs+1))


if __name__ == '__main__':
    # plt.title(sub_dir, font)
    # plt.xlabel('log_interval(every 25 batches)', font)
    # plt.xlabel('epoch', font)
    # plt.ylabel('loss', font)
    fig = plt.figure()
    plot_train(log_file1, fig)
    plot_test(log_file2, fig)

    plt.legend(loc='best', prop=font)
    plt.ylim(bottom=0)
    plt.show()
