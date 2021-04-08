import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os

plt_colors = iter(['b', 'y', 'c', 'm', 'r', 'g', 'k', '#2A0134', '#FF00FF', '#800000'])
plt_markers = iter(['*', '.', 'o', '^', 'v', '<', '>', '1', '2', '3', '4', 's', 'p', ','])
font = {
    'size': 18
}

src_domain, tgt_domain = 'usa', 'sng'
sub_dir = 'new10_contra'
# sub_dir = 'fusion_consis/xmuda/baseline2_usa'
# sub_dir = 'fusion_consis/xmuda/contrast_usa_v0'
# sub_dir = 'fusion_consis/xmuda/src_ctr_usa_v1'
log_train = f'../checkpoints/{sub_dir}/log.log'
log_src_test = f'../checkpoints/{sub_dir}/source_test.log'
log_tgt_test = f'../checkpoints/{sub_dir}/target_test.log'


def plot_train(log_file, epochs=16):
    if not os.path.exists(log_file):
        print(f'File \'{log_file}\' does not exist!')
        return
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
    if 'contrast_loss' in losses.keys():
        losses['train_src_ctrloss'] = losses['contrast_loss']
        del losses['contrast_loss']
    if 'tgt_contrast_loss' in losses.keys():
        losses['train_tgt_ctrloss'] = losses['tgt_contrast_loss']
        del losses['tgt_contrast_loss']
    if 'seg_loss' in losses.keys():
        losses['train_segloss'] = losses['seg_loss']
        del losses['seg_loss']
    for k, v in losses.items():
        x_len = len(v)
        break
    xs = np.arange(x_len) + 1
    for i, (key, val) in enumerate(losses.items()):
        ys = np.array(val)
        plt.plot(xs, ys, label=key, color=next(plt_colors), linewidth=0.9)
    plt.xticks(range(0, x_len, x_len // epochs), labels=range(1, epochs+1))
    return x_len


def plot_test(log_file, iters=1872, curve_name='src'):
    if not os.path.exists(log_file):
        print(f'File \'{log_file}\' does not exist!')
        return
    accf = open(log_file, 'r')
    lines = accf.readlines()
    accf.close()
    losses = defaultdict(list)
    for line in lines:
        if 'seg_loss' in line:
            losses['test_' + curve_name + '_segloss'].append(float(line.split(': ')[1]))
    for k, v in losses.items():
        x_len = len(v)
        break
    xs = np.arange(0, iters, iters // x_len)
    for i, (key, val) in enumerate(losses.items()):
        ys = np.array(val)
        plt.plot(xs, ys, label=key, color=next(plt_colors), linewidth=0.9, linestyle='--')
    # plt.xticks(range(0, x_len, x_len // epochs), labels=range(1, epochs+1))


if __name__ == '__main__':
    plt.figure(figsize=(12, 8))
    plt.title(sub_dir, font)
    plt.xlabel('epoch', font)
    plt.ylabel('loss', font)
    iters = plot_train(log_train)
    plot_test(log_src_test, iters, curve_name='src')
    plot_test(log_tgt_test, iters, curve_name='tgt')

    plt.legend(loc='best', prop=font)
    plt.ylim(bottom=0)
    # plt.ylim(bottom=0, top=0.4)
    plt.show()
