# new train_pipeline

##############################################
# variants: Runner, model
# options: train-test split; class_weights
##############################################
# runner = 'TargetConsistencyRunner'
model_type = 'FusionBaseline4'
# lambda_consis = 1.0

src_train = 'mmda_xmuda_split/train_usa.pkl'
# tgt_train = 'mmda_xmuda_split/train_singapore.pkl'
ann_val = 'mmda_xmuda_split/test_singapore.pkl'
daynight_weights = [2.68678412, 4.36182969, 5.47896839, 3.89026883, 1.]
usasng_weights = [2.47956584, 4.26788384, 5.71114131, 3.80241668, 1.]
class_weights = usasng_weights

lr_step = [16, 22]
total_epochs = 24
# target_start_epoch = lr_step[0]

##########################################################
point_cloud_range = [-50, 0, -5, 50, 50, 3]
anchor_generator_ranges = [[-50, 0, -1.8, 50, 50, -1.8]]
scatter_shape = [200, 400]
voxel_size = [0.25, 0.25, 8]

img_feat_channels = 64
pts_feat_dim = 64
voxel_feat_dim = 128
det_pts_dim = 3  # (x, y, z, timestamp); (x, y, z, reflectance) for seg_pts

backbone_arch = 'regnetx_3.2gf'
arch_map = {'regnetx_1.6gf': [168, 408, 912], 'regnetx_3.2gf': [192, 432, 1008]}
FPN_in_channels = arch_map[backbone_arch]

model = dict(
    type=model_type,
    pts_fc=[3, 64, pts_feat_dim],
    img_backbone=dict(
        type='UNetResNet34',
        out_channels=img_feat_channels,
        pretrained=True),
    img_seg_head=dict(
        type='ImageSegHead',
        img_feat_dim=img_feat_channels,
        seg_pts_dim=pts_feat_dim,  # concat_feats_dim 
        num_classes=5,
        lidar_fc=[],  # no fc before fusion
        concat_fc=[128, 64],
        class_weights=class_weights),
    pts_voxel_layer=dict(
        max_num_points=64,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(30000, 40000)),
    pts_voxel_encoder=dict(
        type='HardVFE',
        in_channels=det_pts_dim + pts_feat_dim + img_feat_channels,
        feat_channels=[128, voxel_feat_dim],
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        # norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01)),
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01)),
    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=voxel_feat_dim, output_shape=scatter_shape),
    # pretrained=dict(pts='open-mmlab://regnetx_1.6gf'),
    pretrained=dict(pts='open-mmlab://' + backbone_arch),
    pts_backbone=dict(
        type='NoStemRegNet',
        arch=backbone_arch,
        out_indices=(1, 2, 3),
        frozen_stages=-1,
        strides=(1, 2, 2, 2),
        base_channels=128,
        stem_channels=128,
        # norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.01),
        norm_eval=False,
        style='pytorch'),
    pts_neck=dict(
        type='FPN',
        # norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.01),
        act_cfg=dict(type='ReLU'),
        in_channels=FPN_in_channels,
        out_channels=256,
        start_level=0,
        num_outs=3),
    pts_bbox_head=dict(
        type='FreeAnchor3DHead',
        num_classes=4,
        in_channels=256,
        feat_channels=256,
        use_direction_classifier=True,
        pre_anchor_topk=25,
        bbox_thr=0.5,
        gamma=2.0,
        alpha=0.5,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=anchor_generator_ranges,
            scales=[1, 2, 4],
            sizes=[
                [0.8660, 2.5981, 1.],  # 1.5/sqrt(3)
                [0.5774, 1.7321, 1.],  # 1/sqrt(3)
                [1., 1., 1.],
                [0.4, 0.4, 1],
            ],
            custom_values=[0, 0],
            rotations=[0, 1.57],
            reshape_out=True),
        assigner_per_size=False,
        diff_rad_by_sin=True,
        dir_offset=0.7854,  # pi/4
        dir_limit_offset=0,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=9),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.8),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)))

train_cfg = dict(
    pts=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            iou_calculator=dict(type='BboxOverlapsNearest3D'),
            pos_iou_thr=0.6,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        allowed_border=0,
        code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.25, 0.25],
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    pts=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_pre=1000,
        nms_thr=0.2,
        score_thr=0.05,
        min_bbox_size=0,
        max_num=500))

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

img_size = (400, 225)
train_pipeline = [
    dict(
        type='LoadSegDetPointsFromFile',  # new 'points'
        load_dim=5,
        use_dim=5),
    dict(
        type='LoadMaskedMultiSweeps',  # modify 'points'; new 'num_seg_pts'
        sweeps_num=10,
        file_client_args=file_client_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),  # new 'gt_bboxes_3d', 'gt_labels_3d'
    dict(type='LoadImgSegLabel'),  # new 'img'(PIL.Image), 'seg_label'
    dict(type='PointsSensorFilterVer2', resize=img_size),  # filter 'points'; new 'pts_indices'
    dict(type='Aug2D', fliplr=0.5, color_jitter=(0.4, 0.4, 0.4)),
    # fliplr & color jitter; 'img': PIL.Image to np.array; update 'seg_pts_indices', 'pts_indices' accordingly;
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.7854, 0.7854],
        scale_ratio_range=[0.9, 1.1],
        translation_std=[0.2, 0.2, 0.2]),  # 3D Rot, Scale, Trans for 'points'
    dict(
        type='RandomFlip3D',
        # flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),  # do nothing; to read further
    dict(type='PointsRangeFilterVer2', point_cloud_range=point_cloud_range),
    # filter 'points', 'pts_indices', 'seg_label'; new 'seg_points', 'seg_pts_indices'
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='DetLabelFilter'),  # Filter labels == -1; not in TEN_CLASSES
    dict(type='PointShuffle'),  # shuffle 'points', 'pts_indices'; make sure no index op after shuffle
    dict(type='MergeCat'),  # merge 'seg_label', 'gt_labels_3d'
    dict(type='SegDetFormatBundle'),
    dict(type='Collect3D', keys=['img', 'seg_points', 'seg_pts_indices', 'seg_label',
                                 'points', 'pts_indices', 'gt_bboxes_3d', 'gt_labels_3d'])
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
    dict(type='FrontImageFilter', resize=img_size),  # 'seg_pts_indices'
    dict(type='PointsSensorFilter', resize=img_size),  # 'pts_indices'
    dict(type='Aug2D'),  # No Aug2D in test
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            # dict(
            #     type='GlobalRotScaleTrans',
            #     rot_range=[0, 0],
            #     scale_ratio_range=[1., 1.],
            #     translation_std=[0, 0, 0]),
            # dict(type='RandomFlip3D'),
            dict(type='SegDetPointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(type='MergeCat'),
            dict(type='SegDetFormatBundle'),
            dict(type='Collect3D', keys=['img', 'seg_points', 'seg_pts_indices', 'seg_label',
                                         'points', 'pts_indices'])
        ])
]

# dataset
dataset_type = 'MMDAMergeCatDataset'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + src_train,
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
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=lr_step)
optimizer_config = dict()
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
momentum_config = None

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

