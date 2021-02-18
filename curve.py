import matplotlib.pyplot as plt
import numpy as np

src_domain, tgt_domain = 'usa', 'sng'
plt_title = f'Seg_Fusion: {src_domain}-{tgt_domain}(test {tgt_domain})'
src_file = f'checkpoints/single_seg/no_src_GANloss/train_{src_domain}_disc_07.txt'
tgt_file = f'checkpoints/single_seg/train_usa_xyz_nodisc.txt'
# plt_title = 'Multi_Task_Fusion: usa-sng(test sng)'
# src_file = 'checkpoints/merge_disc/fusion_usa_rep_new.txt'
# tgt_file = 'checkpoints/merge_disc/oracle_sng.txt'

srcf = open(src_file, 'r')
lines = srcf.readlines()
srcf.close()
src_seg, src_det = [], []
for line in lines:
    if line.startswith('overall_iou'):
        iou = float(line.split()[-1])
        src_seg.append(iou)
    if line.startswith('mAP'):
        mAP = float(line.split()[-1])
        src_det.append(mAP)
if len(src_det) > 0:
    assert len(src_seg) == len(src_det)
src_seg = src_seg[:24]
x_range = np.arange(len(src_seg)) + 1

if tgt_file:
    tgtf = open(tgt_file, 'r')
    lines = tgtf.readlines()
    tgtf.close()
    tgt_seg, tgt_det = [], []
    for line in lines:
        if line.startswith('overall_iou'):
            iou = float(line.split()[-1])
            tgt_seg.append(iou)
        if line.startswith('mAP'):
            mAP = float(line.split()[-1])
            tgt_det.append(mAP)
    if len(tgt_det) > 0:
        assert len(tgt_seg) == len(tgt_det)
        assert len(src_seg) == len(tgt_seg)

if __name__ == '__main__':
    plt.title(plt_title)
    plt.xlabel('epoch')
    plt.ylabel('Det_mAP/Seg_mIOU')
    src_seg = np.array(src_seg)
    plt.plot(x_range, src_seg, label='baseline_seg', color='b', linewidth=0.7)
    tgt_seg = np.array(tgt_seg)
    plt.plot(x_range, tgt_seg, label='oracle_seg', color='b', linestyle='--', linewidth=0.7)
    if len(src_det) > 0:
        src_det = np.array(src_det)
        plt.plot(x_range, src_det, label='baseline_det', color='r', linewidth=0.7)
        tgt_det = np.array(tgt_det)
        plt.plot(x_range, tgt_det, label='oracle_det', color='r', linestyle='--', linewidth=0.7)
    plt.legend(loc='best')
    plt.ylim(bottom=0)
    plt.show()


# [18, 36]; fusion_train_usa, test singapore
# plt_title = 'fusion_train_usa_test_singapore'
# x_range = np.arange(18, 37)
# seg = [54.42, 57.09, 53.51, 35.79, 57.43, 54.98, 59.87, 56.13, 54.80, 58.67, 49.16, 58.43, 59.59, 59.99, 60.22, 59.68, 60.57, 60.39, 60.27]
# det = [45.56, 47.10, 46.88, 39.27, 44.31, 42.49, 49.68, 47.73, 48.58, 49.48, 43.68, 50.71, 50.12, 50.76, 51.26, 50.26, 50.89, 50.55, 50.21]

# [19, 36]
# plt_title = 'fusion_train_day_test_night'
# x_range = np.arange(19, 37)
# seg = [42.79, 41.23, 41.07, 43.18, 40.86, 44.47, 45.13, 41.12, 40.57, 43.80, 45.09, 45.50, 45.66, 45.70, 45.16, 45.58, 45.09, 45.45]
# det = [45.69, 43.37, 41.17, 46.89, 45.11, 53.15, 47.38, 50.23, 48.75, 49.15, 49.85, 48.08, 48.96, 49.60, 51.00, 51.65, 51.56, 51.97]
