from classes_dict import raw_classes, LidarSegChallenge, xMUDA, Contrast
import matplotlib.pyplot as plt

font = {
    'size': 11
}


def xlabels_ys(s, percentage=False):
    xlabels, ys = [], []
    for i, (classname, sub) in enumerate(s):
        xlabels.append(classname)
        num_pts = sum([raw_classes.class_list[k][1] for k in sub])
        ys.append(num_pts)
    if percentage:
        ys = [k * 100 / raw_classes.num_pts for k in ys]
    return xlabels, ys


def plot_hist(cls, ylog=True):
    plt.figure(figsize=cls.fig_size)
    # plt.subplot(121)
    # plt.title('nuScenes LidarSeg Challenge')
    plt.title(type(cls).__name__, font)
    if ylog:
        plt.yscale('log')
        plt.ylabel('log(#points)', font)
    else:
        plt.ylabel('percentage', font)
    class_names, bars = xlabels_ys(cls.class_list, percentage=not ylog)
    plt.bar(range(len(bars)), bars, color=cls.bar_colors)
    plt.xticks(range(len(bars)), class_names, rotation=-30, ha='center', fontsize=font['size'])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_hist(LidarSegChallenge(), ylog=False)
