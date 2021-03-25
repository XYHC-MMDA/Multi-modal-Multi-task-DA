import matplotlib.pyplot as plt
import numpy as np

plt_colors = ['c', 'r', 'g', 'b', 'y', 'k', 'm', '#2A0134', '#FF00FF', '#800000']
plt_markers = ['*', '.', 'o', '^', 'v', '<', '>', '1', '2', '3', '4', 's', 'p', ',']
font = {
    'size': 18
}

src_domain, tgt_domain = 'usa', 'sng'
sub_dir = 'baseline2_usa'
tgt_test_file = f'../checkpoints/fusion_consis/xmuda/{sub_dir}/filter.txt'  # to add: tgt_val, src_val

accf = open(tgt_test_file, 'r')
lines = accf.readlines()
accf.close()
tgt_seg = []
for line in lines:
    if line.startswith('overall_iou'):
        iou = float(line.split()[-1])
        tgt_seg.append(iou)
tgt_seg = tgt_seg[:24]
x_range = np.arange(len(tgt_seg)) + 1


if __name__ == '__main__':
    print(f'{sub_dir}: {max(tgt_seg)}')

    plt.xlabel('epoch', font)
    plt.ylabel('Seg_mIOU', font)
    tgt_seg = np.array(tgt_seg)
    plt.plot(x_range, tgt_seg, label='target_val', color='b', linewidth=0.7)
    plt.legend(loc='best', prop=font)
    plt.xticks(range(0, 25))
    plt.ylim(bottom=0, top=0.7)
    plt.show()
