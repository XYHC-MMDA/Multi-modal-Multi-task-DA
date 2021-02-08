# variants
runner = 'SingleSegRunner'
seg_discriminator = None
seg_optimizer = None
model_type = 'SingleSegXYZ'
lambda_GANLoss = 0.1


point_cloud_range = [-50, 0, -5, 50, 50, 3]
anchor_generator_ranges = [[-50, 0, -1.8, 50, 50, -1.8]]
scatter_shape = [200, 400]
voxel_size = [0.25, 0.25, 8]
src_train = 'mmda_xmuda_split/train_usa.pkl'
tgt_train = 'mmda_xmuda_split/train_singapore.pkl'
ann_val = 'mmda_xmuda_split/test_singapore.pkl'

img_feat_channels = 64
model = dict(
    type=model_type,
    img_backbone=dict(
        type='UNetResNet34',
        out_channels=img_feat_channels,
        pretrained=True),
    img_seg_head=dict(
        type='ImageSegHead',
        img_feat_dim=img_feat_channels,
        seg_pts_dim=4,
        num_classes=5,
        lidar_fc=[64, 64],
        concat_fc=[128, 64],
        class_weights=[2.68678412, 4.36182969, 5.47896839, 3.89026883, 1.])
)

# train_cfg = dict()
# test_cfg = dict()

class_names = [
    'vehicle',  # car, truck, bus, trailer, cv
    'pedestrian',  # pedestrian
    'bike',  # motorcycle, bicycle
    'traffic_boundary'  # traffic_cone, barrier
    # background
]

data_root = '/home/xyyue/xiangyu/nuscenes_unzip/'
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadSegDetPointsFromFile',  # 'points', 'seg_points', 'seg_label'
         load_dim=5,
         use_dim=5),
    # dict(
    #     type='LoadPointsFromMultiSweeps',
    #     sweeps_num=10,
    #     file_client_args=file_client_args),
    # dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='LoadFrontImage'),  # filter 'seg_points', 'seg_label', 'seg_pts_indices' inside front camera; 'img'
    # dict(type='PointsSensorFilter'),  # filter 'points', 'pts_indices' inside front camera
    # dict(
    #     type='GlobalRotScaleTrans',
    #     rot_range=[-0.7854, 0.7854],
    #     scale_ratio_range=[0.9, 1.1],
    #     translation_std=[0.2, 0.2, 0.2]),
    # dict(
    #     type='RandomFlip3D',
    #     flip_ratio_bev_horizontal=0.5,
    #     flip_ratio_bev_vertical=0.5),
    dict(type='SegDetPointsRangeFilter', point_cloud_range=point_cloud_range),  # filter 'points', 'seg_points' within point_cloud_range
    # dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='DetLabelFilter'),
    # dict(type='PointShuffle'),
    dict(type='MergeCat'),  # merge 'seg_label', 'gt_labels_3d' (from 10 classes to 4 classes, without background)
    dict(type='SegDetFormatBundle'),
    dict(type='Collect3D', keys=['img', 'seg_points', 'seg_pts_indices', 'seg_label'])
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
    dict(type='LoadFrontImage'),  # filter 'seg_points', 'seg_label', 'seg_pts_indices' inside front camera; 'img'
    dict(type='PointsSensorFilter'),  # filter 'points', 'pts_indices' inside front camera
    dict(type='SegDetPointsRangeFilter', point_cloud_range=point_cloud_range),  # filter 'points', 'seg_points' within point_cloud_range
    dict(type='MergeCat'),  # merge 'seg_label', 'gt_labels_3d' (from 10 classes to 4 classes, without background)
    dict(type='SegDetFormatBundle'),
    dict(type='Collect3D', keys=['img', 'seg_points', 'seg_pts_indices', 'seg_label'])
    # dict(
    #     type='MultiScaleFlipAug3D',
    #     img_scale=(1333, 800),
    #     pts_scale_ratio=1,
    #     flip=False,
    #     transforms=[
    #         dict(
    #             type='GlobalRotScaleTrans',
    #             rot_range=[0, 0],
    #             scale_ratio_range=[1., 1.],
    #             translation_std=[0, 0, 0]),
    #         dict(type='RandomFlip3D'),
    #         dict(type='SegDetPointsRangeFilter', point_cloud_range=point_cloud_range),
    #         dict(type='MergeCat'),
    #         dict(type='SegDetFormatBundle'),
    #         dict(type='Collect3D', keys=['img', 'seg_points', 'seg_pts_indices', 'seg_label',
    #                                      'points', 'pts_indices'])
    #     ])
]

# dataset
dataset_type = 'MMDAMergeCatDataset'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    source_train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + src_train,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        box_type_3d='LiDAR'),
    target_train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + tgt_train,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + ann_val,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + ann_val,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'))
evaluation = dict(interval=100)

# shedule_2x.py
optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.01)
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[12, 18])
momentum_config = None
total_epochs = 24

# default_runtime.py
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]

