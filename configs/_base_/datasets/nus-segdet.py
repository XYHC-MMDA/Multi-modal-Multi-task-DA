# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-50, 0, -5, 50, 50, 3]
# For nuScenes we usually do 10-class detection
# class_names = [
#     'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
#     'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
# ]
class_names = [
    "car",  # 0
    "truck",  # 1
    "bus",  # 2
    "trailer",  # 3
    "construction_vehicle",  # 4
    "pedestrian",  # 5
    "motorcycle",  # 6
    "bicycle",  # 7
    "traffic_cone",  # 8
    "barrier",  # 9
    # "background"
]
dataset_type = 'NuscMultiModalDataset'
data_root = '/home/xyyue/xiangyu/nuscenes_unzip/'
# Input modality for nuScenes dataset, this is consistent with the submission
# format which requires the information in input_modality.
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')
# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.
# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/nuscenes/': 's3://nuscenes/nuscenes/',
#         'data/nuscenes/': 's3://nuscenes/nuscenes/'
#     }))
train_pipeline = [
    dict(
        type='LoadSegDetPointsFromFile',  # new
        load_dim=5,
        use_dim=5),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        file_client_args=file_client_args),
    dict(type='LoadFrontImage'),  # new
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='SegDetPointsRangeFilter', point_cloud_range=point_cloud_range),  # new
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='SegDetFormatBundle', class_names=class_names),
    dict(type='Collect3D', keys=['img', 'img_indices', 'seg_label', 'points', 'gt_bboxes_3d', 'gt_labels_3d'])  # TODO
]
test_pipeline = [
    dict(
        type='LoadSegDetPointsFromFile',
        load_dim=5,
        use_dim=5),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        file_client_args=file_client_args),
    dict(type='LoadFrontImage'),  # new
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='SegDetPointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='SegDetFormatBundle',
                class_names=class_names),
            dict(type='Collect3D', keys=['img', 'img_indices', 'seg_label', 'points'])
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    mini_train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_mini_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    test_on_train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_train.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    mini_test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_mini_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'))
# For nuScenes dataset, we usually evaluate the model at the end of training.
# Since the models are trained by 24 epochs by default, we set evaluation
# interval to be 24. Please change the interval accordingly if you do not
# use a default schedule.
evaluation = dict(interval=25)
