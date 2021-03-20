import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

plt_colors = ['c', 'r', 'g', 'b', 'y', 'k', 'm', '#2A0134', '#FF00FF', '#800000']
plt_markers = ['*', '.', 'o', '^', 'v', '<', '>', '1', '2', '3', '4', 's', 'p', ',']
font = {
    'size': 18
}

src_domain, tgt_domain = 'usa', 'sng'
plt_title = ''
sub_dir = 'contrast_usa_v1'
log_file = f'../checkpoints/fusion_consis/xmuda/{sub_dir}/log.log'  # to add: tgt_val, src_val


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
    losses['src_contrast_loss'] = losses['contrast_loss']
    del losses['contrast_loss']
x_len = len(losses['seg_loss'])
x_range = np.arange(x_len) + 1


if __name__ == '__main__':
    plt.title(plt_title, font)
    # plt.xlabel('log_interval(every 25 batches)', font)
    plt.xlabel('epoch', font)
    plt.ylabel('loss', font)
    for i, (key, val) in enumerate(losses.items()):
        val = np.array(val)
        plt.plot(x_range, val, label=key, color=plt_colors[i], linewidth=1.)
    plt.xticks(range(0, x_len, x_len // 24), labels=range(0, 25))
    plt.legend(loc='best', prop=font)
    plt.ylim(bottom=0)
    plt.show()
