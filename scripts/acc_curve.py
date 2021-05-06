import matplotlib.pyplot as plt
import numpy as np
import os

plt_colors = ['c', 'r', 'g', 'b', 'y', 'k', 'm', '#2A0134', '#FF00FF', '#800000']
plt_markers = ['*', '.', 'o', '^', 'v', '<', '>', '1', '2', '3', '4', 's', 'p', ',']
font = {
    'size': 18
}

src_domain, tgt_domain = 'usa', 'sng'
# log_dir = '../checkpoints/fusion_consis/xmuda/contrast_usa_v1'
# log_dir = '../checkpoints/new10_contra/usa_finetune_v2'
# log_dir = '../checkpoints/new10_contra/vanilla_fusion/baseline_usa_v3_run2'
log_dir = '../checkpoints/new10_contra/contra_usa_v5_p1'
# log_dir = '../checkpoints/pretrain/src_usa_finetune_v0_w1'

test_files = []
splits = ['source_test', 'target_test']
for split in splits:
    # test_files.append((f'../checkpoints/fusion_consis/xmuda/{sub_dir}/{split}.log', split))
    ftuple = (os.path.join(log_dir, f'{split}.log'), split)
    test_files.append(ftuple)


def curves():
    ret = []
    for test_file, split in test_files:
        accf = open(test_file, 'r')
        lines = accf.readlines()
        accf.close()
        y_values = []
        for line in lines:
            if line.startswith('overall_iou'):
                iou = float(line.split()[-1])
                y_values.append(iou)
        # y_values = y_values[:24]
        ret.append([np.array(y_values), split])
    return ret


if __name__ == '__main__':
    plt.figure(figsize=(12, 8))
    cfg_name = os.path.basename(log_dir)
    plt.title(cfg_name, font)
    plt.xlabel('epoch', font)
    plt.ylabel('Seg_mIOU', font)
    y_list = curves()
    x_range = np.arange(len(y_list[0][0])) + 1
    for i, (y, split) in enumerate(y_list):
        print(f'{cfg_name} - {split}: Epoch [{np.argmax(y + 1)}] {np.max(y)}')
        plt.plot(x_range, y, label=split, color=plt_colors[i], linewidth=1.5)
    plt.legend(loc='best', prop=font)
    # plt.xticks(range(0, 25))
    plt.xticks(range(1, len(x_range) + 1))
    plt.ylim(bottom=0.2, top=0.8)
    # plt.ylim(bottom=0., top=0.6)
    plt.show()
