import matplotlib.pyplot as plt
import numpy as np

# [18, 36]; fusion_train_usa, test singapore
plt_title = 'fusion_train_usa_test_singapore'
x = np.arange(18, 37)
seg = [54.42, 57.09, 53.51, 35.79, 57.43, 54.98, 59.87, 56.13, 54.80, 58.67, 49.16, 58.43, 59.59, 59.99, 60.22, 59.68, 60.57, 60.39, 60.27]
det = [45.56, 47.10, 46.88, 39.27, 44.31, 42.49, 49.68, 47.73, 48.58, 49.48, 43.68, 50.71, 50.12, 50.76, 51.26, 50.26, 50.89, 50.55, 50.21]


if __name__ == '__main__':
    plt.title(plt_title)
    plt.xlabel('epoch')
    plt.ylabel('Det_mAP/Seg_mIOU')
    seg = np.array(seg)
    det = np.array(det)
    avg = (seg + det) / 2
    plt.plot(x, seg, label='seg')
    plt.plot(x, det, label='det')
    plt.plot(x, avg, label='mean')
    plt.legend(loc='best')
    plt.ylim(bottom=0)
    plt.show()
