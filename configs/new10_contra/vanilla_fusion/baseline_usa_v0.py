# contrastive on new class labels;
# todo: class_weights

##############################################
# variants: Runner, model
# options: train-test split; class_weights
##############################################
# runner
runner = 'XmudaRunner'  # for any customized runner, use general_train.py
only_contrast = False  # default False

# model args; if no contrast, just set contrast_criterion to None; assert contrast_criterion is not None or not only_contrast
model_type = 'SegFusionV3'
# contrast_criterion = dict(type='NT_Xent', temperature=0.1, normalize=True, contrast_mode='cross_entropy')
max_pts, groups = 2048, 1
lambda_contrast = 0.01
contrast_criterion = None

img_dim, pts_dim = 64, 16 
prelogits_dim = img_dim + pts_dim

# XmudaAug3D, UNetSCN
scn_scale = 20
scn_full_scale = 4096

# source/target domain
# src, tgt = 'day', 'night'
src, tgt = 'usa', 'singapore'

# lr_scheduler
lr_step = [16, 22]
total_epochs = 24

# class_weights
daynight_weights = [2.167, 3.196, 4.054, 2.777, 1., 2.831, 2.089, 2.047, 1.534, 1.534, 2.345]
usasng_weights = [2.154, 3.298, 4.447, 2.855, 1., 2.841, 2.152, 2.758, 1.541, 1.845, 2.257]
class_weights = usasng_weights if src == 'usa' else daynight_weights

# splits' paths
source_train = f'mmda_xmuda_split/train_{src}.pkl'
source_test = f'mmda_xmuda_split/test_{src}.pkl'
target_train = f'mmda_xmuda_split/train_{tgt}.pkl'
target_test = f'mmda_xmuda_split/test_{tgt}.pkl'
target_val = f'mmda_xmuda_split/val_{tgt}.pkl'
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
        m=pts_dim,
        full_scale=scn_full_scale),
    num_classes=11,
    prelogits_dim=prelogits_dim,
    class_weights=class_weights,
    contrast_criterion=contrast_criterion,
    max_pts=max_pts,
    groups=groups,
    lambda_contrast=lambda_contrast,
    img_fcs=(img_dim, img_dim, pts_dim),
    pts_fcs=(pts_dim, pts_dim, pts_dim)
)

train_cfg = None
test_cfg = None

# class_names = [
#     'vehicle',  # car, truck, bus, trailer, cv
#     'pedestrian',  # pedestrian
#     'bike',  # motorcycle, bicycle
#     'traffic_boundary'  # traffic_cone, barrier
#     # background
# ]

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
    dict(type='LoadPointsFromFileVer2', load_dim=5, use_dim=5),  # new 'points', 'num_seg_pts'
    dict(type='LoadImgSegLabelVer2', resize=resize),  # new 'img'(PIL.Image), 'seg_label'
    dict(type='PointsSensorFilterVer2', img_size=img_size, resize=resize),
    # filter 'points'; new 'pts_indices'; modify 'num_seg_pts', 'seg_label'
    dict(type='Aug2D', fliplr=0.5, color_jitter=(0.4, 0.4, 0.4)),
    # fliplr & color jitter; 'img': PIL.Image to np.array; update 'seg_pts_indices', 'pts_indices' accordingly;
    dict(type='XmudaAug3D', scale=scn_scale, full_scale=scn_full_scale,
         noisy_rot=0.1, flip_x=0.5, flip_y=0.5, rot_z=6.2831, transl=True),  # new 'scn_coords'
    # filter 'points', 'pts_indices', 'seg_label'; new 'seg_points', 'seg_pts_indices'
    dict(type='GetSegFromPoints'),  # new 'seg_points', 'seg_pts_indices'
    # dict(type='MergeCat'),  # merge 'seg_label'
    dict(type='SegDetFormatBundle'),
    dict(type='Collect3D', keys=['img', 'seg_points', 'seg_pts_indices', 'seg_label', 'scn_coords'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFileVer2',
        load_dim=5,
        use_dim=5),
    dict(type='LoadImgSegLabelVer2', resize=resize),  # new 'img'(PIL.Image), 'seg_label'
    dict(type='PointsSensorFilterVer2', img_size=img_size, resize=resize),
    # filter 'points'; new 'pts_indices'; modify 'num_seg_pts', 'seg_label'
    dict(type='Aug2D'),  # No Aug2D in test; just PIL.Image to np.ndarray
    dict(type='XmudaAug3D', scale=scn_scale, full_scale=scn_full_scale),  # new 'scn_coords'; no aug3d in test
    dict(type='GetSegFromPoints'),  # new 'seg_points', 'seg_pts_indices'
    # dict(type='MergeCat'),  # merge 'seg_label'
    dict(type='SegDetFormatBundle'),
    dict(type='Collect3D', keys=['img', 'seg_points', 'seg_pts_indices', 'seg_label', 'scn_coords'])
]


# dataset
dataset_type = 'ContrastSegDatasetV0'
data_root = '/home/xyyue/xiangyu/nuscenes_unzip/'
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    source_train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + source_train,
        pipeline=train_pipeline,
        # classes=class_names,
        modality=input_modality,
        test_mode=False),
    target_train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + target_train,
        pipeline=train_pipeline,
        # classes=class_names,
        modality=input_modality,
        test_mode=False),
    source_test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + source_test,
        pipeline=test_pipeline,
        # classes=class_names,
        modality=input_modality,
        test_mode=True),
    target_test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + target_test,
        pipeline=test_pipeline,
        # classes=class_names,
        modality=input_modality,
        test_mode=True),
    target_val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + target_val,
        pipeline=test_pipeline,
        # classes=class_names,
        modality=input_modality,
        test_mode=True)
)
evaluation = dict(interval=100)

# shedule_2x.py
optimizer = dict(type='Adam', lr=0.001, weight_decay=0.01)
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

