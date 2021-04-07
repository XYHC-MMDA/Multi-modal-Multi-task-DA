import matplotlib.pyplot as plt
import numpy as np

plt_colors = ['c', 'r', 'g', 'b', 'y', 'k', 'm', '#2A0134', '#FF00FF', '#800000']
plt_markers = ['*', '.', 'o', '^', 'v', '<', '>', '1', '2', '3', '4', 's', 'p', ',']
font = {
    'size': 18
}

src_domain, tgt_domain = 'usa', 'sng'
sub_dir = 'contrast_usa_v1'

test_files = []
splits = ['source_test', 'target_test']
for split in splits:
    test_files.append((f'../checkpoints/fusion_consis/xmuda/{sub_dir}/{split}.log', split))


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
        y_values = y_values[:24]
        ret.append([np.array(y_values), split])
    return ret


if __name__ == '__main__':
    plt.title(sub_dir, font)
    plt.xlabel('epoch', font)
    plt.ylabel('Seg_mIOU', font)
    y_list = curves()
    x_range = np.arange(len(y_list[0][0])) + 1
    for i, (y, split) in enumerate(y_list):
        print(f'{sub_dir} - {split}: {max(y)}')
        plt.plot(x_range, y, label=split, color=plt_colors[i], linewidth=0.7)
    plt.legend(loc='best', prop=font)
    plt.xticks(range(0, 25))
    plt.ylim(bottom=0, top=0.7)
    plt.show()
