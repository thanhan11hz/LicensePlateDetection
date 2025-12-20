# _base_ = [
#     '../_base_/models/rtmdet_s.py',
#     '../_base_/datasets/coco_detection.py',
#     '../_base_/default_runtime.py'
# ]

# # --------------------
# # Dataset
# # --------------------
# dataset_type = 'CocoDataset'
# data_root = 'data/license_plate/'

# metainfo = dict(classes=('license_plate',))

# train_dataloader = dict(
#     batch_size=32,
#     num_workers=8,
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file='annotations/instances_train.json',
#         data_prefix=dict(img='train/'),
#         metainfo=metainfo
#     )
# )

# val_dataloader = dict(
#     batch_size=32,
#     num_workers=8,
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file='annotations/instances_val.json',
#         data_prefix=dict(img='val/'),
#         metainfo=metainfo
#     )
# )

# # test_dataloader = val_dataloader

# test_dataloader = test_dataloader = dict(
#     batch_size=32,
#     num_workers=8,
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file='annotations/instances_test.json',
#         data_prefix=dict(img='test/'),
#         metainfo=metainfo
#     )
# )

# # --------------------
# # Model
# # --------------------
# model = dict(
#     bbox_head=dict(
#         num_classes=1
#     )
# )

# # --------------------
# # Optimizer
# # --------------------
# optim_wrapper = dict(
#     optimizer=dict(
#         type='SGD',
#         lr=0.04,
#         momentum=0.9,
#         weight_decay=0.0001
#     )
# )

# # --------------------
# # Training schedule
# # --------------------
# train_cfg = dict(max_epochs=20)

# param_scheduler = dict(
#     type='MultiStepLR',
#     milestones=[14, 18],
#     gamma=0.1
# )

# # --------------------
# # Evaluation
# # --------------------
# val_evaluator = dict(type='CocoMetric', metric='bbox')
# # test_evaluator = val_evaluator
# test_evaluator = dict(type='CocoMetric', metric='bbox')

# default_hooks = dict(
#     checkpoint=dict(
#         type='CheckpointHook',
#         interval=1,
#         save_best='bbox_mAP'
#     )
# )

# # --------------------
# # Pretrained
# # --------------------
# load_from = (
#     'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/'
#     'rtmdet_s_8xb32-300e_coco/rtmdet_s_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
# )

# work_dir = './work_dirs/license_plate/rtmdet'

checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-s_imagenet_600e.pth'

model = dict(
    type='RTMDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        batch_augments=None),
    backbone=dict(
        type='CSPNeXt', # <-- Đã thêm chữ 'e' theo tài liệu bạn gửi
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.33,
        widen_factor=0.5,
        channel_attention=True,
        norm_cfg=dict(type='BN'), # Đổi SyncBN -> BN cho 1 GPU
        act_cfg=dict(type='SiLU', inplace=True),
        init_cfg=dict(type='Pretrained', prefix='backbone.', checkpoint=checkpoint)),
    neck=dict(
        type='CSPNeXtPAFPN', # <-- Thêm chữ 'e'
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1,
        expand_ratio=0.5,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='RTMDetSepBNHead',
        num_classes=1, # 1 lớp: license_plate
        in_channels=128,
        stacked_convs=2,
        feat_channels=128,
        anchor_generator=dict(
            type='MlvlPointGenerator', # Dùng generator từ tài liệu
            strides=[8, 16, 32],
            offset=5),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        with_objectness=False,
        exp_on_reg=False,
        share_conv=True,
        pred_kernel_size=1,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    train_cfg=dict(
        assigner=dict(type='DynamicSoftLabelAssigner', topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=30000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100)
)

# --------------------
# DATA & RUNTIME (Rút gọn để chạy nhanh)
# --------------------
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CachedMosaic', img_scale=(640, 640), pad_val=114.0),
    dict(type='RandomResize', scale=(1280, 1280), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=32, 
    num_workers=16,
    dataset=dict(
        type='CocoDataset',
        data_root='data/',
        ann_file='COCO_labels/instances_train.json',
        data_prefix=dict(img='images/train/'),
        metainfo=dict(classes=('license_plate', )),
        pipeline=train_pipeline))

# 1. Pipeline cho Validation (thường đơn giản hơn Train)
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs')
]

# 2. Bộ nạp dữ liệu Val
val_dataloader = dict(
    batch_size=32, # Val không cần batch size quá lớn
    num_workers=16,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root='data/',
        ann_file='COCO_labels/instances_val.json', # File nhãn tập Val
        data_prefix=dict(img='images/val/'),       # Thư mục ảnh tập Val
        metainfo=dict(classes=('license_plate', )),
        pipeline=val_pipeline))

# 3. Bộ chấm điểm (Metric) - Quan trọng nhất để biết có Overfit không
val_evaluator = dict(
    type='CocoMetric',
    ann_file='data/COCO_labels/instances_val.json',
    metric='bbox', # Chấm điểm mAP cho Bounding Box
    format_only=False)

# Test cũng dùng tương tự Val
test_dataloader = val_dataloader
test_evaluator = val_evaluator

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.05))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=10, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto'))

load_from = '.cache/torch/hub/checkpoints/cspnext-s_imagenet_600e.pth'
work_dir = './work_dirs/license_plate/rtmdet_s_lpr'