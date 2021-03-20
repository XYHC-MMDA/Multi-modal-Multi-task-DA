# SegFusionV3; introduce groups;  1024 -> 256 * 4

##############################################
# variants: Runner, model
# options: train-test split; class_weights
##############################################
# runner
runner = 'XmudaRunner'  # for any customized runner, use general_train.py
only_contrast = False  # default False

# model args; if no contrast, just set contrast_criterion to None; assert contrast_criterion is not None or not only_contrast
model_type = 'SegFusionV3'
contrast_criterion = dict(type='NT_Xent', temperature=0.1, normalize=True, contrast_mode='cross_entropy')
max_pts, groups = 256, 4
lambda_contrast = 0.01

img_dim, pts_dim = 64, 16
prelogits_dim = img_dim + pts_dim

# XmudaAug3D, UNetSCN
scn_scale = 20
scn_full_scale = 4096

# split
src_train = 'mmda_xmuda_split/train_usa.pkl'
tgt_train = 'mmda_xmuda_split/train_singapore.pkl'
ann_val = 'mmda_xmuda_split/test_singapore.pkl'

# class_weights
daynight_weights = [2.68678412, 4.36182969, 5.47896839, 3.89026883, 1.]
usasng_weights = [2.47956584, 4.26788384, 5.71114131, 3.80241668, 1.]
class_weights = usasng_weights

# lr_scheduler
lr_step = [16, 22]
total_epochs = 24

#######################################################
# model
#######################################################
model = dict(
    type=model_type,
    img_backbone=dict(
        type='UNetResNet34',
        out_channels=img_dim,
        pretrained=True),
    pts_backbone=dict(
        type='UNetSCN',
        in_channels=1,
        full_scale=scn_full_scale),
    num_classes=5,
    prelogits_dim=prelogits_dim,
    class_weights=class_weights,
    contrast_criterion=contrast_criterion,
    max_pts=max_pts,
    groups=groups,
    lambda_contrast=lambda_contrast,
    img_fcs=[img_dim, img_dim, pts_dim],
    pts_fcs=[pts_dim, pts_dim, pts_dim]
)

train_cfg = None
test_cfg = None

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

img_size = (1600, 900)
resize = (400, 225)
train_pipeline = [
    dict(
        type='LoadPointsFromFileVer2',  # new 'points', 'num_seg_pts'
        load_dim=5,
        use_dim=5),
    # dict(
    #     type='LoadMaskedMultiSweeps',  # modify 'points'; new 'num_seg_pts'
    #     sweeps_num=10,
    #     file_client_args=file_client_args),
    # dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),  # new 'gt_bboxes_3d', 'gt_labels_3d'
    dict(type='LoadImgSegLabel', resize=resize),  # new 'img'(PIL.Image), 'seg_label'
    dict(type='PointsSensorFilterVer2', img_size=img_size, resize=resize),
    # filter 'points'; new 'pts_indices'; modify 'num_seg_pts', 'seg_label'
    dict(type='Aug2D', fliplr=0.5, color_jitter=(0.4, 0.4, 0.4)),
    # fliplr & color jitter; 'img': PIL.Image to np.array; update 'seg_pts_indices', 'pts_indices' accordingly;
    dict(type='XmudaAug3D', scale=scn_scale, full_scale=scn_full_scale,
         noisy_rot=0.1, flip_x=0.5, flip_y=0.5, rot_z=6.2831, transl=True),  # new 'scn_coords'
    # dict(
    #     type='GlobalRotScaleTrans',
    #     rot_range=[-0.7854, 0.7854],
    #     scale_ratio_range=[0.9, 1.1],
    #     translation_std=[0.2, 0.2, 0.2]),  # 3D Rot, Scale, Trans for 'points'
    # dict(
    #     type='RandomFlip3D',
    #     # flip_ratio_bev_horizontal=0.5,
    #     flip_ratio_bev_vertical=0.5),  # do nothing; to read further
    # dict(type='PointsRangeFilterVer2', point_cloud_range=point_cloud_range),
    # # filter 'points', 'pts_indices', 'seg_label'; new 'seg_points', 'seg_pts_indices'
    dict(type='GetSegFromPoints'),  # new 'seg_points', 'seg_pts_indices'
    # dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='DetLabelFilter'),  # Filter labels == -1; not in TEN_CLASSES
    # dict(type='PointShuffle'),  # shuffle 'points', 'pts_indices'; make sure no index op after shuffle
    dict(type='MergeCat'),  # merge 'seg_label'
    dict(type='SegDetFormatBundle'),
    dict(type='Collect3D', keys=['img', 'seg_points', 'seg_pts_indices', 'seg_label', 'scn_coords'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFileVer2',
        load_dim=5,
        use_dim=5),
    # dict(
    #     type='LoadMaskedMultiSweeps',
    #     sweeps_num=10,
    #     file_client_args=file_client_args),
    dict(type='LoadImgSegLabel', resize=resize),  # new 'img'(PIL.Image), 'seg_label'
    dict(type='PointsSensorFilterVer2', img_size=img_size, resize=resize),
    # filter 'points'; new 'pts_indices'; modify 'num_seg_pts', 'seg_label'
    dict(type='Aug2D'),  # No Aug2D in test; just PIL.Image to np.ndarray
    dict(type='XmudaAug3D', scale=scn_scale, full_scale=scn_full_scale),  # new 'scn_coords'; no aug3d in test
    dict(type='GetSegFromPoints'),  # new 'seg_points', 'seg_pts_indices'
    dict(type='MergeCat'),  # merge 'seg_label'
    dict(type='SegDetFormatBundle'),
    dict(type='Collect3D', keys=['img', 'seg_points', 'seg_pts_indices', 'seg_label', 'scn_coords'])
    # dict(
    #     type='MultiScaleFlipAug3D',
    #     img_scale=(1333, 800),
    #     pts_scale_ratio=1,
    #     flip=False,
    #     transforms=[
    #         # dict(type='PointsRangeFilterVer2', point_cloud_range=point_cloud_range),
    #         dict(type='MergeCat'),
    #         dict(type='SegDetFormatBundle'),
    #         dict(type='Collect3D', keys=['img', 'seg_points', 'seg_pts_indices', 'seg_label'])
    #     ])
]

# dataset
dataset_type = 'MMDAMergeCatDataset'
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    source_train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + src_train,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        filter_empty_gt=False,
        box_type_3d='LiDAR'),
    target_train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + tgt_train,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        filter_empty_gt=False,
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
    interval=25,
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

