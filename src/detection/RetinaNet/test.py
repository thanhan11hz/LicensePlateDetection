"""Test script for RetinaNet on CCPD2019 test sets."""
import argparse
import os
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.apis import init_detector, inference_detector
from mmdet.evaluation import CocoMetric
from mmengine.evaluator import Evaluator
from mmdet.registry import DATASETS
from mmengine.dataset import DefaultSampler


def test_on_dataset(config_path, checkpoint_path, ann_file, img_prefix, output_dir):
    """Run evaluation on a test dataset."""
    cfg = Config.fromfile(config_path)

    # Update test dataset config
    cfg.test_dataloader.dataset.ann_file = ann_file
    cfg.test_dataloader.dataset.data_prefix = dict(img=img_prefix)
    cfg.test_evaluator.ann_file = ann_file

    cfg.load_from = checkpoint_path
    cfg.work_dir = output_dir

    # Build runner and test
    runner = Runner.from_cfg(cfg)
    metrics = runner.test()

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Test RetinaNet on CCPD2019')
    parser.add_argument('--config', default='configs/retinanet_ccpd.py',
                        help='Config file path')
    parser.add_argument('--checkpoint', default='work_dirs/retinanet_ccpd/best_coco_bbox_mAP_epoch_14.pth',
                        help='Checkpoint file path')
    parser.add_argument('--test-set', default='all',
                        choices=['all', 'normal', 'blur', 'tilt', 'rotate', 'fn', 'db', 'weather', 'challenge'],
                        help='Which test set to evaluate')
    args = parser.parse_args()

    # Test set configurations
    test_sets = {
        'normal': 'instances_test_normal.json',
        'blur': 'instances_test_blur.json',
        'tilt': 'instances_test_tilt.json',
        'rotate': 'instances_test_rotate.json',
        'fn': 'instances_test_fn.json',
        'db': 'instances_test_db.json',
        'weather': 'instances_test_weather.json',
        'challenge': 'instances_test_challenge.json',
    }

    ann_dir = '/home/quang/Apps/Khoi/COCO_labels-20251212T112049Z-3-001/COCO_labels/'
    # Note: You may need to adjust img_prefix based on where your test images are located
    img_prefix = '/home/quang/Apps/Khoi/CCPD2019/images/test/'

    if args.test_set == 'all':
        sets_to_test = test_sets.keys()
    else:
        sets_to_test = [args.test_set]

    print(f"\nUsing checkpoint: {args.checkpoint}")
    print("=" * 60)

    for test_name in sets_to_test:
        print(f"\n>>> Testing on: {test_name}")
        ann_file = os.path.join(ann_dir, test_sets[test_name])
        output_dir = f'work_dirs/test_results/{test_name}'

        try:
            metrics = test_on_dataset(
                args.config,
                args.checkpoint,
                ann_file,
                img_prefix,
                output_dir
            )
            print(f"Results for {test_name}: {metrics}")
        except Exception as e:
            print(f"Error testing {test_name}: {e}")


if __name__ == '__main__':
    main()
