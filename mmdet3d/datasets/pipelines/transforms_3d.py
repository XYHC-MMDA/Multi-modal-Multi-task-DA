import numpy as np
from mmcv import is_tuple_of
from mmcv.utils import build_from_cfg
from PIL import Image

from mmdet3d.core import VoxelGenerator
from mmdet3d.core.bbox import box_np_ops
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import RandomFlip
from ..registry import OBJECTSAMPLERS
from .data_augment_utils import noise_per_object_v3_


@PIPELINES.register_module()
class RandomFlip3D(RandomFlip):
    """Flip the points & bbox.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        sync_2d (bool, optional): Whether to apply flip according to the 2D
            images. If True, it will apply the same flip as that to 2D images.
            If False, it will decide whether to flip randomly and independently
            to that of 2D images. Defaults to True.
        flip_ratio_bev_horizontal (float, optional): The flipping probability
            in horizontal direction. Defaults to 0.0.
        flip_ratio_bev_vertical (float, optional): The flipping probability
            in vertical direction. Defaults to 0.0.
    """

    def __init__(self,
                 sync_2d=True,
                 flip_ratio_bev_horizontal=0.0,
                 flip_ratio_bev_vertical=0.0,
                 **kwargs):
        super(RandomFlip3D, self).__init__(
            flip_ratio=flip_ratio_bev_horizontal, **kwargs)
        self.sync_2d = sync_2d
        self.flip_ratio_bev_vertical = flip_ratio_bev_vertical
        if flip_ratio_bev_horizontal is not None:
            assert isinstance(
                flip_ratio_bev_horizontal,
                (int, float)) and 0 <= flip_ratio_bev_horizontal <= 1
        if flip_ratio_bev_vertical is not None:
            assert isinstance(
                flip_ratio_bev_vertical,
                (int, float)) and 0 <= flip_ratio_bev_vertical <= 1

    def random_flip_data_3d(self, input_dict, direction='horizontal'):
        """Flip 3D data randomly.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            direction (str): Flip direction. Default: horizontal.

        Returns:
            dict: Flipped results, 'points', 'bbox3d_fields' keys are \
                updated in the result dict.
        """
        assert direction in ['horizontal', 'vertical']
        if len(input_dict['bbox3d_fields']) == 0:  # test mode
            input_dict['bbox3d_fields'].append('empty_box3d')
            input_dict['empty_box3d'] = input_dict['box_type_3d'](
                np.array([], dtype=np.float32))
        assert len(input_dict['bbox3d_fields']) == 1  # ['gt_bboxes_3d']
        for key in input_dict['bbox3d_fields']:
            input_dict['points'] = input_dict[key].flip(
                direction, points=input_dict['points'])

    def __call__(self, input_dict):
        """Call function to flip points, values in the ``bbox3d_fields`` and \
        also flip 2D image and its annotations.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction', \
                'pcd_horizontal_flip' and 'pcd_vertical_flip' keys are added \
                into result dict.
        """
        # filp 2D image and its annotations
        super(RandomFlip3D, self).__call__(input_dict)

        if self.sync_2d:
            input_dict['pcd_horizontal_flip'] = input_dict['flip']
            input_dict['pcd_vertical_flip'] = False
        else:
            if 'pcd_horizontal_flip' not in input_dict:
                flip_horizontal = True if np.random.rand(
                ) < self.flip_ratio else False
                input_dict['pcd_horizontal_flip'] = flip_horizontal
            if 'pcd_vertical_flip' not in input_dict:
                flip_vertical = True if np.random.rand(
                ) < self.flip_ratio_bev_vertical else False
                input_dict['pcd_vertical_flip'] = flip_vertical

        if input_dict['pcd_horizontal_flip']:
            self.random_flip_data_3d(input_dict, 'horizontal')
        if input_dict['pcd_vertical_flip']:
            self.random_flip_data_3d(input_dict, 'vertical')
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += '(sync_2d={},'.format(self.sync_2d)
        repr_str += 'flip_ratio_bev_vertical={})'.format(
            self.flip_ratio_bev_vertical)
        return repr_str


@PIPELINES.register_module()
class ObjectSample(object):
    """Sample GT objects to the data.

    Args:
        db_sampler (dict): Config dict of the database sampler.
        sample_2d (bool): Whether to also paste 2D image patch to the images
            This should be true when applying multi-modality cut-and-paste.
            Defaults to False.
    """

    def __init__(self, db_sampler, sample_2d=False):
        self.sampler_cfg = db_sampler
        self.sample_2d = sample_2d
        if 'type' not in db_sampler.keys():
            db_sampler['type'] = 'DataBaseSampler'
        self.db_sampler = build_from_cfg(db_sampler, OBJECTSAMPLERS)

    @staticmethod
    def remove_points_in_boxes(points, boxes):
        """Remove the points in the sampled bounding boxes.

        Args:
            points (np.ndarray): Input point cloud array.
            boxes (np.ndarray): Sampled ground truth boxes.

        Returns:
            np.ndarray: Points with those in the boxes removed.
        """
        masks = box_np_ops.points_in_rbbox(points, boxes)
        points = points[np.logical_not(masks.any(-1))]
        return points

    def __call__(self, input_dict):
        """Call function to sample ground truth objects to the data.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after object sampling augmentation, \
                'points', 'gt_bboxes_3d', 'gt_labels_3d' keys are updated \
                in the result dict.
        """
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']

        # change to float for blending operation
        points = input_dict['points']
        if self.sample_2d:
            img = input_dict['img']
            gt_bboxes_2d = input_dict['gt_bboxes']
            # Assume for now 3D & 2D bboxes are the same
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d.tensor.numpy(),
                gt_labels_3d,
                gt_bboxes_2d=gt_bboxes_2d,
                img=img)
        else:
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d.tensor.numpy(), gt_labels_3d, img=None)

        if sampled_dict is not None:
            sampled_gt_bboxes_3d = sampled_dict['gt_bboxes_3d']
            sampled_points = sampled_dict['points']
            sampled_gt_labels = sampled_dict['gt_labels_3d']

            gt_labels_3d = np.concatenate([gt_labels_3d, sampled_gt_labels],
                                          axis=0)
            gt_bboxes_3d = gt_bboxes_3d.new_box(
                np.concatenate(
                    [gt_bboxes_3d.tensor.numpy(), sampled_gt_bboxes_3d]))

            points = self.remove_points_in_boxes(points, sampled_gt_bboxes_3d)
            # check the points dimension
            dim_inds = points.shape[-1]
            points = np.concatenate([sampled_points[:, :dim_inds], points],
                                    axis=0)

            if self.sample_2d:
                sampled_gt_bboxes_2d = sampled_dict['gt_bboxes_2d']
                gt_bboxes_2d = np.concatenate(
                    [gt_bboxes_2d, sampled_gt_bboxes_2d]).astype(np.float32)

                input_dict['gt_bboxes'] = gt_bboxes_2d
                input_dict['img'] = sampled_dict['img']

        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d.astype(np.long)
        input_dict['points'] = points

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f' sample_2d={self.sample_2d},'
        repr_str += f' data_root={self.sampler_cfg.data_root},'
        repr_str += f' info_path={self.sampler_cfg.info_path},'
        repr_str += f' rate={self.sampler_cfg.rate},'
        repr_str += f' prepare={self.sampler_cfg.prepare},'
        repr_str += f' classes={self.sampler_cfg.classes},'
        repr_str += f' sample_groups={self.sampler_cfg.sample_groups}'
        return repr_str


@PIPELINES.register_module()
class ObjectNoise(object):
    """Apply noise to each GT objects in the scene.

    Args:
        translation_std (list[float], optional): Standard deviation of the
            distribution where translation noise are sampled from.
            Defaults to [0.25, 0.25, 0.25].
        global_rot_range (list[float], optional): Global rotation to the scene.
            Defaults to [0.0, 0.0].
        rot_range (list[float], optional): Object rotation range.
            Defaults to [-0.15707963267, 0.15707963267].
        num_try (int, optional): Number of times to try if the noise applied is
            invalid. Defaults to 100.
    """

    def __init__(self,
                 translation_std=[0.25, 0.25, 0.25],
                 global_rot_range=[0.0, 0.0],
                 rot_range=[-0.15707963267, 0.15707963267],
                 num_try=100):
        self.translation_std = translation_std
        self.global_rot_range = global_rot_range
        self.rot_range = rot_range
        self.num_try = num_try

    def __call__(self, input_dict):
        """Call function to apply noise to each ground truth in the scene.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after adding noise to each object, \
                'points', 'gt_bboxes_3d' keys are updated in the result dict.
        """
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        points = input_dict['points']

        # TODO: check this inplace function
        numpy_box = gt_bboxes_3d.tensor.numpy()
        noise_per_object_v3_(
            numpy_box,
            points,
            rotation_perturb=self.rot_range,
            center_noise_std=self.translation_std,
            global_random_rot_range=self.global_rot_range,
            num_try=self.num_try)

        input_dict['gt_bboxes_3d'] = gt_bboxes_3d.new_box(numpy_box)
        input_dict['points'] = points
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += '(num_try={},'.format(self.num_try)
        repr_str += ' translation_std={},'.format(self.translation_std)
        repr_str += ' global_rot_range={},'.format(self.global_rot_range)
        repr_str += ' rot_range={})'.format(self.rot_range)
        return repr_str


@PIPELINES.register_module()
class GlobalRotScaleTrans(object):
    """Apply global rotation, scaling and translation to a 3D scene.

    Args:
        rot_range (list[float]): Range of rotation angle.
            Defaults to [-0.78539816, 0.78539816] (close to [-pi/4, pi/4]).
        scale_ratio_range (list[float]): Range of scale ratio.
            Defaults to [0.95, 1.05].
        translation_std (list[float]): The standard deviation of ranslation
            noise. This apply random translation to a scene by a noise, which
            is sampled from a gaussian distribution whose standard deviation
            is set by ``translation_std``. Defaults to [0, 0, 0]
        shift_height (bool): Whether to shift height.
            (the fourth dimension of indoor points) when scaling.
            Defaults to False.
    """

    def __init__(self,
                 rot_range=[-0.78539816, 0.78539816],
                 scale_ratio_range=[0.95, 1.05],
                 translation_std=[0, 0, 0],
                 shift_height=False):
        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = translation_std
        self.shift_height = shift_height

    def _trans_bbox_points(self, input_dict):
        """Private function to translate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after translation, 'points', 'pcd_trans' \
                and keys in input_dict['bbox3d_fields'] are updated \
                in the result dict.
        """
        if not isinstance(self.translation_std, (list, tuple, np.ndarray)):
            translation_std = [
                self.translation_std, self.translation_std,
                self.translation_std
            ]
        else:
            translation_std = self.translation_std
        translation_std = np.array(translation_std, dtype=np.float32)
        trans_factor = np.random.normal(scale=translation_std, size=3).T

        input_dict['points'][:, :3] += trans_factor
        input_dict['pcd_trans'] = trans_factor
        for key in input_dict['bbox3d_fields']:
            input_dict[key].translate(trans_factor)

    def _rot_bbox_points(self, input_dict):
        """Private function to rotate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after rotation, 'points', 'pcd_rotation' \
                and keys in input_dict['bbox3d_fields'] are updated \
                in the result dict.
        """
        rotation = self.rot_range
        if not isinstance(rotation, list):
            rotation = [-rotation, rotation]
        noise_rotation = np.random.uniform(rotation[0], rotation[1])

        for key in input_dict['bbox3d_fields']:
            if len(input_dict[key].tensor) != 0:
                points, rot_mat_T = input_dict[key].rotate(
                    noise_rotation, input_dict['points'])
                input_dict['points'] = points
                input_dict['pcd_rotation'] = rot_mat_T

    def _scale_bbox_points(self, input_dict):
        """Private function to scale bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points'and keys in \
                input_dict['bbox3d_fields'] are updated in the result dict.
        """
        scale = input_dict['pcd_scale_factor']
        input_dict['points'][:, :3] *= scale
        if self.shift_height:
            input_dict['points'][:, -1] *= scale

        for key in input_dict['bbox3d_fields']:
            input_dict[key].scale(scale)

    def _random_scale(self, input_dict):
        """Private function to randomly set the scale factor.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'pcd_scale_factor' are updated \
                in the result dict.
        """
        scale_factor = np.random.uniform(self.scale_ratio_range[0],
                                         self.scale_ratio_range[1])
        input_dict['pcd_scale_factor'] = scale_factor

    def __call__(self, input_dict):
        """Private function to rotate, scale and translate bounding boxes and \
        points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points', 'pcd_rotation',
                'pcd_scale_factor', 'pcd_trans' and keys in \
                input_dict['bbox3d_fields'] are updated in the result dict.
        """
        self._rot_bbox_points(input_dict)

        if 'pcd_scale_factor' not in input_dict:
            self._random_scale(input_dict)
        self._scale_bbox_points(input_dict)

        self._trans_bbox_points(input_dict)
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += '(rot_range={},'.format(self.rot_range)
        repr_str += ' scale_ratio_range={},'.format(self.scale_ratio_range)
        repr_str += ' translation_std={})'.format(self.translation_std)
        repr_str += ' shift_height={})'.format(self.shift_height)
        return repr_str


@PIPELINES.register_module()
class PointShuffle(object):
    """Shuffle input points."""

    def __call__(self, input_dict):
        """Call function to shuffle points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'points' keys are updated \
                in the result dict.
        """
        idx = np.arange(input_dict['points'].shape[0])
        np.random.shuffle(idx)
        for key in ['points', 'pts_indices']:
            if key not in input_dict.keys():
                continue
            input_dict[key] = input_dict[key][idx]
        # np.random.shuffle(input_dict['points'])
        return input_dict

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class ObjectRangeFilter(object):
    """Filter objects by the range.

    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)
        self.bev_range = self.pcd_range[[0, 1, 3, 4]]

    def __call__(self, input_dict):
        """Call function to filter objects by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']
        mask = gt_bboxes_3d.in_range_bev(self.bev_range)
        gt_bboxes_3d = gt_bboxes_3d[mask]
        # mask is a torch tensor but gt_labels_3d is still numpy array
        # using mask to index gt_labels_3d will cause bug when
        # len(gt_labels_3d) == 1, where mask=1 will be interpreted
        # as gt_labels_3d[1] and cause out of index error
        gt_labels_3d = gt_labels_3d[mask.numpy().astype(np.bool)]

        # limit rad to [-pi, pi]
        gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)
        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += '(point_cloud_range={})'.format(self.pcd_range.tolist())
        return repr_str


@PIPELINES.register_module()
class PointsRangeFilter(object):
    """Filter points by the range.

    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(
            point_cloud_range, dtype=np.float32)[np.newaxis, :]

    def __call__(self, input_dict):
        """Call function to filter points by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'points' keys are updated \
                in the result dict.
        """
        points = input_dict['points']
        points_mask = ((points[:, :3] >= self.pcd_range[:, :3])
                       & (points[:, :3] < self.pcd_range[:, 3:]))
        points_mask = points_mask[:, 0] & points_mask[:, 1] & points_mask[:, 2]
        clean_points = points[points_mask, :]
        input_dict['points'] = clean_points
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += '(point_cloud_range={})'.format(self.pcd_range.tolist())
        return repr_str


@PIPELINES.register_module()
class FrontImageFilter(object):
    def __init__(self, resize=(400, 225)):
        self.resize = resize

    def __call__(self, results):
        filepath = results['img_filename'][0]  # front image
        rot = results['lidar2img'][0]
        seg_pts = results['seg_points']  # (N, 4)
        seg_label = results['seg_label']
        img = Image.open(filepath)
        img_size = img.size  # (1600, 900)

        # img_indices
        num_points = seg_pts.shape[0]
        pts_cam = np.concatenate([seg_pts[:, :3], np.ones((num_points, 1))], axis=1) @ rot.T
        pts_img = pts_cam[:, :3]

        # calc mask
        pts_img[:, 0] /= pts_img[:, 2]
        pts_img[:, 1] /= pts_img[:, 2]
        mask = ((0, 0) < pts_img[:, :2]) & (pts_img[:, :2] < img_size)
        # mask = mask[:, 0] & mask[:, 1]
        mask = mask[:, 0] & mask[:, 1] & (pts_img[:, 2] > 0)

        # filter
        seg_pts_indices = pts_img[mask][:, :2]  # (col, row)
        seg_pts = seg_pts[mask]
        seg_label = seg_label[mask]

        if self.resize:
            img = img.resize(self.resize, Image.BILINEAR)
            seg_pts_indices = seg_pts_indices * self.resize / img_size

        # img = np.array(img, dtype=np.float32) / 255.  # shape=(225, 400, 3)
        seg_pts_indices = np.fliplr(seg_pts_indices).astype(np.int64)  # (row, col)
        results['img'] = img
        results['seg_pts_indices'] = seg_pts_indices  # (N, 2): (row, column)
        results['seg_points'] = seg_pts  # pts inside front camera; lidar coordinate
        results['seg_label'] = seg_label

        return results


@PIPELINES.register_module()
class PointsSensorFilter(object):
    """Currently only implement front camera. Filter points inside front camera
    """

    def __init__(self, img_size=(1600, 900), resize=(400, 225)):
        self.img_size = img_size
        self.resize = resize

    def __call__(self, results):
        rot = results['lidar2img'][0]
        pts_lidar = results['points']
        num_points = pts_lidar.shape[0]

        pts_cam = np.concatenate([pts_lidar[:, :3], np.ones((num_points, 1))], axis=1) @ rot.T
        pts_img = pts_cam[:, :3]
        pts_img[:, 0] /= pts_img[:, 2]
        pts_img[:, 1] /= pts_img[:, 2]
        mask = ((0, 0) < pts_img[:, :2]) & (pts_img[:, :2] < self.img_size)
        # mask = mask[:, 0] & mask[:, 1]
        mask = mask[:, 0] & mask[:, 1] & (pts_img[:, 2] > 0)

        pts_lidar = pts_lidar[mask]
        pts_indices = pts_img[:, :2][mask]
        if self.resize:
            pts_indices = pts_indices * self.resize / self.img_size
        pts_indices = np.fliplr(pts_indices).astype(np.int64)  # (row, col)

        results['points'] = pts_lidar
        results['pts_indices'] = pts_indices
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += '(point_cloud_range={})'.format(self.pcd_range.tolist())
        return repr_str


@PIPELINES.register_module()
class PointsSensorFilterVer2(object):
    """Currently only implement front camera. Filter points inside front camera
    """

    def __init__(self, img_size=(1600, 900), resize=(400, 225)):
        self.img_size = img_size
        self.resize = resize

    def __call__(self, results):
        rot = results['lidar2img'][0]
        pts_lidar = results['points']
        num_points = pts_lidar.shape[0]
        num_seg_pts = results['num_seg_pts']
        seg_label = results['seg_label']

        pts_cam = np.concatenate([pts_lidar[:, :3], np.ones((num_points, 1))], axis=1) @ rot.T
        pts_img = pts_cam[:, :3]
        pts_img[:, 0] /= pts_img[:, 2]
        pts_img[:, 1] /= pts_img[:, 2]
        mask = ((0, 0) < pts_img[:, :2]) & (pts_img[:, :2] < self.img_size)
        # mask = mask[:, 0] & mask[:, 1]
        mask = mask[:, 0] & mask[:, 1] & (pts_img[:, 2] > 0)
        seg_label = seg_label[mask[:num_seg_pts]]
        num_seg_pts = np.sum(mask[:num_seg_pts])
        # assert num_seg_pts == len(seg_label)

        pts_lidar = pts_lidar[mask]
        pts_indices = pts_img[:, :2][mask]
        if self.resize:
            pts_indices = pts_indices * self.resize / self.img_size
        pts_indices = np.fliplr(pts_indices).astype(np.int64)  # (row, col)

        results['points'] = pts_lidar
        results['pts_indices'] = pts_indices
        results['num_seg_pts'] = num_seg_pts
        results['seg_label'] = seg_label
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += '(point_cloud_range={})'.format(self.pcd_range.tolist())
        return repr_str


@PIPELINES.register_module()
class SegDetPointsRangeFilter(object):
    """Filter points by the range.

    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(
            point_cloud_range, dtype=np.float32)[np.newaxis, :]

    def __call__(self, input_dict):
        points = input_dict['points']
        points_mask = ((points[:, :3] >= self.pcd_range[:, :3])
                       & (points[:, :3] < self.pcd_range[:, 3:]))
        points_mask = points_mask[:, 0] & points_mask[:, 1] & points_mask[:, 2]
        for key in ['points', 'pts_indices']:
            if key not in input_dict.keys():
                continue
            input_dict[key] = input_dict[key][points_mask]

        seg_pts = input_dict['seg_points']
        pts_mask = ((seg_pts[:, :3] >= self.pcd_range[:, :3])
                    & (seg_pts[:, :3] < self.pcd_range[:, 3:]))
        pts_mask = pts_mask[:, 0] & pts_mask[:, 1] & pts_mask[:, 2]
        for key in ['seg_points', 'seg_label', 'seg_pts_indices']:
            if key not in input_dict.keys():
                continue
            input_dict[key] = input_dict[key][pts_mask]

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += '(point_cloud_range={})'.format(self.pcd_range.tolist())
        return repr_str


@PIPELINES.register_module()
class PointsRangeFilterVer2(object):
    """Filter points by the range; generate 'seg_pts', 'seg_pts_indices'; update 'seg_label'

    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)[np.newaxis, :]

    def __call__(self, input_dict):
        points = input_dict['points']
        num_seg_pts = input_dict['num_seg_pts']
        seg_label = input_dict['seg_label']  # assert len(seg_label) == num_seg_pts

        # points_mask
        points_mask = ((points[:, :3] >= self.pcd_range[:, :3]) &
                       (points[:, :3] < self.pcd_range[:, 3:]))
        points_mask = points_mask[:, 0] & points_mask[:, 1] & points_mask[:, 2]
        input_dict['points'] = points[points_mask]
        input_dict['pts_indices'] = input_dict['pts_indices'][points_mask]

        # seg_pts_mask, seg_label
        seg_pts_mask = points_mask[:num_seg_pts]
        seg_label = seg_label[seg_pts_mask]
        num_seg_pts = np.sum(seg_pts_mask)  # assert num_seg_pts == len(seg_label)
        input_dict['seg_label'] = seg_label
        input_dict['num_seg_pts'] = num_seg_pts

        # seg_points, seg_pts_indices
        input_dict['seg_points'] = input_dict['points'][:num_seg_pts].copy()
        input_dict['seg_pts_indices'] = input_dict['pts_indices'][:num_seg_pts].copy()

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += '(point_cloud_range={})'.format(self.pcd_range.tolist())
        return repr_str


@PIPELINES.register_module()
class GetSegFromPoints(object):
    '''
        new 'seg_points', 'seg_pts_indices'
    '''

    def __init__(self):
        pass

    def __call__(self, input_dict):
        num_seg_pts = input_dict['num_seg_pts']

        # seg_points, seg_pts_indices
        input_dict['seg_points'] = input_dict['points'][:num_seg_pts].copy()
        input_dict['seg_pts_indices'] = input_dict['pts_indices'][:num_seg_pts].copy()

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class MergeCat(object):
    def __init__(self):
        # order according to config.class_names
        # vehicle(0): car, truck, bus, trailer, construction vehicle
        # pedestrian(1): pedestrian
        # bike(2): motorcycle, bicycle
        # traffic_boundary(3): traffic cone, barrier
        # background(4)
        merge_map = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0],
                     [5, 1],
                     [6, 2], [7, 2],
                     [8, 3], [9, 3],
                     [10, 4]]
        self.merge_map = dict()
        for x, y in merge_map:
            self.merge_map[x] = y

    def __call__(self, input_dict):
        seg_label = input_dict['seg_label']
        seg_label = np.array(list(map(lambda x: self.merge_map[x], seg_label)))
        input_dict['seg_label'] = seg_label
        if 'gt_labels_3d' in input_dict:
            det_label = input_dict['gt_labels_3d']
            det_label = np.array(list(map(lambda x: self.merge_map[x], det_label)))
            input_dict['gt_labels_3d'] = det_label
        return input_dict


@PIPELINES.register_module()
class Aug2D(object):
    def __init__(self, fliplr=0.0, color_jitter=None):
        self.fliplr = fliplr
        from torchvision import transforms as T
        self.color_jitter = T.ColorJitter(*color_jitter) if color_jitter else None

    def __call__(self, input_dict):
        # 2D augmentation
        image = input_dict['img']  # PIL.Image
        if self.color_jitter is not None:
            image = self.color_jitter(image)
        image = np.array(image, dtype=np.float32) / 255.  # shape=(225, 400, 3); RGB in [0, 1]

        if np.random.rand() < self.fliplr:
            image = np.ascontiguousarray(np.fliplr(image))
            for key in ['seg_pts_indices', 'pts_indices']:
                if key not in input_dict.keys():
                    continue
                pts_indices = input_dict[key]
                pts_indices[:, 1] = image.shape[1] - 1 - pts_indices[:, 1]
                input_dict[key] = pts_indices

        input_dict['img'] = image
        return input_dict


@PIPELINES.register_module()
class XmudaAug3D(object):
    # xmuda/data/utils/augmentation_3d.py
    def __init__(self, scale, full_scale, noisy_rot=0.0, flip_x=0.0, flip_y=0.0, rot_z=0.0, transl=False):
        self.scale = scale
        self.full_scale = full_scale
        self.noisy_rot = noisy_rot
        self.flip_x = flip_x
        self.flip_y = flip_y
        self.rot_z = rot_z
        self.transl = transl

    def __call__(self, input_dict):
        scn_coords = input_dict['points'][:, :3].copy()
        rot_matrix = np.eye(3, dtype=np.float32)
        if self.noisy_rot > 0:
            # add noise to rotation matrix
            rot_matrix += np.random.randn(3, 3) * self.noisy_rot
        if self.flip_x > 0:
            # flip x axis: multiply element at (0, 0) with 1 or -1
            rot_matrix[0][0] *= np.random.randint(0, 2) * 2 - 1
        if self.flip_y > 0:
            # flip y axis: multiply element at (1, 1) with 1 or -1
            rot_matrix[1][1] *= np.random.randint(0, 2) * 2 - 1
        if self.rot_z > 0:
            # rotate around z-axis (up-axis)
            theta = np.random.rand() * self.rot_z
            z_rot_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                     [np.sin(theta), np.cos(theta), 0],
                                     [0, 0, 1]], dtype=np.float32)
            rot_matrix = rot_matrix.dot(z_rot_matrix)
        scn_coords = scn_coords.dot(rot_matrix)

        # scale with inverse voxel size (e.g. 20 corresponds to 5cm)
        scn_coords = scn_coords * self.scale
        # translate points to positive octant (receptive field of SCN in x, y, z coords is in interval [0, full_scale])
        scn_coords -= scn_coords.min(0)

        if self.transl:
            # random translation inside receptive field of SCN
            offset = np.clip(self.full_scale - scn_coords.max(0) - 0.001, a_min=0, a_max=None) * np.random.rand(3)
            scn_coords += offset

        # cast to integer
        scn_coords = scn_coords.astype(np.int64)

        # only use voxels inside receptive field
        idxs = (scn_coords.min(1) >= 0) * (scn_coords.max(1) < self.full_scale)
        scn_coords = scn_coords[idxs]
        for key in ['seg_label', 'points', 'pts_indices']:
            if key not in input_dict.keys():
                continue
            input_dict[key] = input_dict[key][idxs]

        input_dict['scn_coords'] = scn_coords
        input_dict['num_seg_pts'] = len(scn_coords)
        # input_dict['scn_feats'] = np.ones([len(scn_coords), 1], np.float32)  # simply use 1 as feature
        return input_dict


@PIPELINES.register_module()
class DetLabelFilter(object):
    def __call__(self, input_dict):
        gt_labels_3d = input_dict['gt_labels_3d']
        mask = gt_labels_3d >= 0  # get_anno_info
        input_dict['gt_bboxes_3d'] = input_dict['gt_bboxes_3d'][mask]
        input_dict['gt_labels_3d'] = input_dict['gt_labels_3d'][mask]

        return input_dict


@PIPELINES.register_module()
class ObjectNameFilter(object):
    """Filter GT objects by their names.

    Args:
        classes (list[str]): List of class names to be kept for training.
    """

    def __init__(self, classes):
        self.classes = classes
        self.labels = list(range(len(self.classes)))

    def __call__(self, input_dict):
        """Call function to filter objects by their names.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        gt_labels_3d = input_dict['gt_labels_3d']
        gt_bboxes_mask = np.array([n in self.labels for n in gt_labels_3d],
                                  dtype=np.bool_)
        input_dict['gt_bboxes_3d'] = input_dict['gt_bboxes_3d'][gt_bboxes_mask]
        input_dict['gt_labels_3d'] = input_dict['gt_labels_3d'][gt_bboxes_mask]

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(classes={self.classes})'
        return repr_str


@PIPELINES.register_module()
class IndoorPointSample(object):
    """Indoor point sample.

    Sampling data to a certain number.

    Args:
        name (str): Name of the dataset.
        num_points (int): Number of points to be sampled.
    """

    def __init__(self, num_points):
        self.num_points = num_points

    def points_random_sampling(self,
                               points,
                               num_samples,
                               replace=None,
                               return_choices=False):
        """Points random sampling.

        Sample points to a certain number.

        Args:
            points (np.ndarray): 3D Points.
            num_samples (int): Number of samples to be sampled.
            replace (bool): Whether the sample is with or without replacement.
            Defaults to None.
            return_choices (bool): Whether return choice. Defaults to False.

        Returns:
            tuple[np.ndarray] | np.ndarray:

                - points (np.ndarray): 3D Points.
                - choices (np.ndarray, optional): The generated random samples.
        """
        if replace is None:
            replace = (points.shape[0] < num_samples)
        choices = np.random.choice(
            points.shape[0], num_samples, replace=replace)
        if return_choices:
            return points[choices], choices
        else:
            return points[choices]

    def __call__(self, results):
        """Call function to sample points to in indoor scenes.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after sampling, 'points', 'pts_instance_mask' \
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        points = results['points']
        points, choices = self.points_random_sampling(
            points, self.num_points, return_choices=True)
        pts_instance_mask = results.get('pts_instance_mask', None)
        pts_semantic_mask = results.get('pts_semantic_mask', None)
        results['points'] = points

        if pts_instance_mask is not None and pts_semantic_mask is not None:
            pts_instance_mask = pts_instance_mask[choices]
            pts_semantic_mask = pts_semantic_mask[choices]
            results['pts_instance_mask'] = pts_instance_mask
            results['pts_semantic_mask'] = pts_semantic_mask

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += '(num_points={})'.format(self.num_points)
        return repr_str


@PIPELINES.register_module()
class BackgroundPointsFilter(object):
    """Filter background points near the bounding box.

    Args:
        bbox_enlarge_range (tuple[float], float): Bbox enlarge range.
    """

    def __init__(self, bbox_enlarge_range):
        assert (is_tuple_of(bbox_enlarge_range, float)
                and len(bbox_enlarge_range) == 3) \
            or isinstance(bbox_enlarge_range, float), \
            f'Invalid arguments bbox_enlarge_range {bbox_enlarge_range}'

        if isinstance(bbox_enlarge_range, float):
            bbox_enlarge_range = [bbox_enlarge_range] * 3
        self.bbox_enlarge_range = np.array(
            bbox_enlarge_range, dtype=np.float32)[np.newaxis, :]

    def __call__(self, input_dict):
        """Call function to filter points by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'points' keys are updated \
                in the result dict.
        """
        points = input_dict['points']
        gt_bboxes_3d = input_dict['gt_bboxes_3d']

        gt_bboxes_3d_np = gt_bboxes_3d.tensor.numpy()
        gt_bboxes_3d_np[:, :3] = gt_bboxes_3d.gravity_center.numpy()
        enlarged_gt_bboxes_3d = gt_bboxes_3d_np.copy()
        enlarged_gt_bboxes_3d[:, 3:6] += self.bbox_enlarge_range
        foreground_masks = box_np_ops.points_in_rbbox(points, gt_bboxes_3d_np)
        enlarge_foreground_masks = box_np_ops.points_in_rbbox(
            points, enlarged_gt_bboxes_3d)
        foreground_masks = foreground_masks.max(1)
        enlarge_foreground_masks = enlarge_foreground_masks.max(1)
        valid_masks = ~np.logical_and(~foreground_masks,
                                      enlarge_foreground_masks)

        input_dict['points'] = points[valid_masks]
        pts_instance_mask = input_dict.get('pts_instance_mask', None)
        if pts_instance_mask is not None:
            input_dict['pts_instance_mask'] = pts_instance_mask[valid_masks]

        pts_semantic_mask = input_dict.get('pts_semantic_mask', None)
        if pts_semantic_mask is not None:
            input_dict['pts_semantic_mask'] = pts_semantic_mask[valid_masks]
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += '(bbox_enlarge_range={})'.format(
            self.bbox_enlarge_range.tolist())
        return repr_str


@PIPELINES.register_module()
class VoxelBasedPointSampler(object):
    """Voxel based point sampler.

    Apply voxel sampling to multiple sweep points.

    Args:
        cur_sweep_cfg (dict): Config for sampling current points.
        prev_sweep_cfg (dict): Config for sampling previous points.
        time_dim (int): Index that indicate the time dimention
            for input points.
    """

    def __init__(self, cur_sweep_cfg, prev_sweep_cfg=None, time_dim=3):
        self.cur_voxel_generator = VoxelGenerator(**cur_sweep_cfg)
        self.cur_voxel_num = self.cur_voxel_generator._max_voxels
        self.time_dim = time_dim
        if prev_sweep_cfg is not None:
            assert prev_sweep_cfg['max_num_points'] == \
                cur_sweep_cfg['max_num_points']
            self.prev_voxel_generator = VoxelGenerator(**prev_sweep_cfg)
            self.prev_voxel_num = self.prev_voxel_generator._max_voxels
        else:
            self.prev_voxel_generator = None
            self.prev_voxel_num = 0

    def _sample_points(self, points, sampler, point_dim):
        """Sample points for each points subset.

        Args:
            points (np.ndarray): Points subset to be sampled.
            sampler (VoxelGenerator): Voxel based sampler for
                each points subset.
            point_dim (int): The dimention of each points

        Returns:
            np.ndarray: Sampled points.
        """
        voxels, coors, num_points_per_voxel = sampler.generate(points)
        if voxels.shape[0] < sampler._max_voxels:
            padding_points = np.zeros([
                sampler._max_voxels - voxels.shape[0], sampler._max_num_points,
                point_dim
            ],
                                      dtype=points.dtype)
            padding_points[:] = voxels[0]
            sample_points = np.concatenate([voxels, padding_points], axis=0)
        else:
            sample_points = voxels

        return sample_points

    def __call__(self, results):
        """Call function to sample points from multiple sweeps.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after sampling, 'points', 'pts_instance_mask' \
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        points = results['points']
        original_dim = points.shape[1]

        # TODO: process instance and semantic mask while _max_num_points
        # is larger than 1
        # Extend points with seg and mask fields
        map_fields2dim = []
        start_dim = original_dim
        extra_channel = [points]
        for idx, key in enumerate(results['pts_mask_fields']):
            map_fields2dim.append((key, idx + start_dim))
            extra_channel.append(results[key][..., None])

        start_dim += len(results['pts_mask_fields'])
        for idx, key in enumerate(results['pts_seg_fields']):
            map_fields2dim.append((key, idx + start_dim))
            extra_channel.append(results[key][..., None])

        points = np.concatenate(extra_channel, axis=-1)

        # Split points into two part, current sweep points and
        # previous sweeps points.
        # TODO: support different sampling methods for next sweeps points
        # and previous sweeps points.
        cur_points_flag = (points[:, self.time_dim] == 0)
        cur_sweep_points = points[cur_points_flag]
        prev_sweeps_points = points[~cur_points_flag]
        if prev_sweeps_points.shape[0] == 0:
            prev_sweeps_points = cur_sweep_points

        # Shuffle points before sampling
        np.random.shuffle(cur_sweep_points)
        np.random.shuffle(prev_sweeps_points)

        cur_sweep_points = self._sample_points(cur_sweep_points,
                                               self.cur_voxel_generator,
                                               points.shape[1])
        if self.prev_voxel_generator is not None:
            prev_sweeps_points = self._sample_points(prev_sweeps_points,
                                                     self.prev_voxel_generator,
                                                     points.shape[1])

            points = np.concatenate([cur_sweep_points, prev_sweeps_points], 0)
        else:
            points = cur_sweep_points

        if self.cur_voxel_generator._max_num_points == 1:
            points = points.squeeze(1)
        results['points'] = points[..., :original_dim]

        # Restore the correspoinding seg and mask fields
        for key, dim_index in map_fields2dim:
            results[key] = points[..., dim_index]

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""

        def _auto_indent(repr_str, indent):
            repr_str = repr_str.split('\n')
            repr_str = [' ' * indent + t + '\n' for t in repr_str]
            repr_str = ''.join(repr_str)[:-1]
            return repr_str

        repr_str = self.__class__.__name__
        indent = 4
        repr_str += '(\n'
        repr_str += ' ' * indent + f'num_cur_sweep={self.cur_voxel_num},\n'
        repr_str += ' ' * indent + f'num_prev_sweep={self.prev_voxel_num},\n'
        repr_str += ' ' * indent + f'time_dim={self.time_dim},\n'
        repr_str += ' ' * indent + 'cur_voxel_generator=\n'
        repr_str += f'{_auto_indent(repr(self.cur_voxel_generator), 8)},\n'
        repr_str += ' ' * indent + 'prev_voxel_generator=\n'
        repr_str += f'{_auto_indent(repr(self.prev_voxel_generator), 8)})'
        return repr_str
