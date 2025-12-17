_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/default_runtime.py'
]

# --------------------
# Dataset
# --------------------
dataset_type = 'CocoDataset'
data_root = 'data/license_plate/'

metainfo = dict(classes=('license_plate',))

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='train/'),
        metainfo=metainfo
    )
)

val_dataloader = dict(
    batch_size=32,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='val/'),
        metainfo=metainfo
    )
)

# test_dataloader = val_dataloader

test_dataloader = test_dataloader = dict(
    batch_size=32,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_test.json',
        data_prefix=dict(img='test/'),
        metainfo=metainfo
    )
)

# --------------------
# Model
# --------------------
model = dict(
    bbox_head=dict(
        num_classes=1
    )
)

# --------------------
# Optimizer
# --------------------
optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.04,   # batch 32 → scaled
        momentum=0.9,
        weight_decay=0.0001
    )
)

# --------------------
# Training schedule
# --------------------
train_cfg = dict(max_epochs=20)

param_scheduler = dict(
    type='MultiStepLR',
    milestones=[14, 18],
    gamma=0.1
)

# --------------------
# Evaluation
# --------------------
val_evaluator = dict(type='CocoMetric', metric='bbox')
# test_evaluator = val_evaluator
test_evaluator = dict(type='CocoMetric', metric='bbox')

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='bbox_mAP'
    )
)

# --------------------
# Pretrained
# --------------------
load_from = (
    'https://download.openmmlab.com/mmdetection/v2.0/retinanet/'
    'retinanet_r50_fpn_1x_coco/'
    'retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'
)

work_dir = './work_dirs/license_plate/retinanet'
