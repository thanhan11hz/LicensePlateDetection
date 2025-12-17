"""Training script for RetinaNet on CCPD2019."""
import argparse
from mmengine.config import Config
from mmengine.runner import Runner


def main():
    parser = argparse.ArgumentParser(description='Train RetinaNet on CCPD2019')
    parser.add_argument('--config', default='configs/retinanet_ccpd.py',
                        help='Config file path')
    parser.add_argument('--work-dir', default='work_dirs/retinanet_ccpd',
                        help='Working directory')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest checkpoint')
    parser.add_argument('--gpus', type=int, default=3,
                        help='Number of GPUs to use')
    args = parser.parse_args()

    # Load config
    cfg = Config.fromfile(args.config)

    # Update work directory
    cfg.work_dir = args.work_dir

    # Set up distributed training if multiple GPUs
    if args.gpus > 1:
        cfg.launcher = 'pytorch'
        cfg.env_cfg = dict(
            cudnn_benchmark=False,
            mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
            dist_cfg=dict(backend='nccl'),
        )

    # Handle resume
    if args.resume:
        cfg.resume = True

    # Build runner and start training
    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()
