_base_ = [
    '../_base_/models/rtmdet_s.py',
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
        lr=0.04,
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
    'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/'
    'rtmdet_s_8xb32-300e_coco/rtmdet_s_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
)

work_dir = './work_dirs/license_plate/rtmdet'
