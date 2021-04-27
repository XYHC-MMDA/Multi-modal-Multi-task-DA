import mmcv
import numpy as np
import pyquaternion
import tempfile
from nuscenes.utils.data_classes import Box as NuScenesBox
from os import path as osp

from mmdet.datasets import DATASETS
from ..core import show_result
from ..core.bbox import Box3DMode, LiDARInstance3DBoxes
from .custom_3d import Custom3DDataset


@DATASETS.register_module()
class ContrastSegDatasetV0(Custom3DDataset):
    CLASSES = ('vehicle', 'pedestrian', 'bike', 'traffic_boundary',
               'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation',
               'ignore')
    SEG_CLASSES = ('vehicle', 'pedestrian', 'bike', 'traffic_boundary',
                   'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation',
                   'ignore')

    def __init__(self,
                 ann_file,
                 pipeline=None,
                 data_root=None,
                 classes=None,
                 modality=None,
                 # box_type_3d='LiDAR',
                 test_mode=False):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            # classes=classes,
            modality=modality,
            # box_type_3d=box_type_3d,
            filter_empty_gt=False,
            test_mode=test_mode)
        if classes is not None:
            ContrastSegDatasetV0.CLASSES = tuple(classes)
            ContrastSegDatasetV0.SEG_CLASSES = tuple(classes)

    def load_annotations(self, ann_file):
        # init: self.data_infos = self.load_annotations()
        data = mmcv.load(ann_file)  # dict with keys=('infos', 'metadata')
        data_infos = list(data['infos'])  # list of info dict
        # self.version = 'v1.0-trainval'
        return data_infos

    def get_data_info(self, index):
        '''
            input_dict = self.get_data_info(index)
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            return example
        '''
        info = self.data_infos[index]
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            seglabel_filename=info['lidarseg_path'],
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])

                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                lidar2cam_rts.append(lidar2cam_rt)

                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    lidar2cam=lidar2cam_rts,
                ))
        return input_dict
