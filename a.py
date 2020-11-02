from mmdet3d.datasets import NuScenesDataset
from mmdet3d.datasets import build_dataset
from mmcv import Config, DictAction

parser = argparse.ArgumentParser(description='Train a detector')
parser.add_argument('config', help='train config file path')
args = parser.parse_args()
cfg = Config.fromfile(args.config)
dataset = build_dataset(cfg.data.train)
