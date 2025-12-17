"""Run inference on test images."""
import argparse
import os
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
import mmcv


def main():
    parser = argparse.ArgumentParser(description='Run inference on images')
    parser.add_argument('--config', default='configs/retinanet_ccpd.py',
                        help='Config file path')
    parser.add_argument('--checkpoint', default='work_dirs/retinanet_ccpd/best_coco_bbox_mAP_epoch_14.pth',
                        help='Checkpoint file path')
    parser.add_argument('--img-dir', default='/home/quang/Apps/Khoi/testimages',
                        help='Directory containing test images')
    parser.add_argument('--output-dir', default='/home/quang/Apps/Khoi/inference_results',
                        help='Directory to save results')
    parser.add_argument('--score-thr', type=float, default=0.5,
                        help='Score threshold for detections')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize model
    print(f"Loading model from {args.checkpoint}...")
    model = init_detector(args.config, args.checkpoint, device='cuda:0')

    # Initialize visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    # Get all images
    img_files = [f for f in os.listdir(args.img_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    print(f"Found {len(img_files)} images")
    print(f"Score threshold: {args.score_thr}")
    print("-" * 50)

    for img_file in img_files:
        img_path = os.path.join(args.img_dir, img_file)
        print(f"\nProcessing: {img_file}")

        # Run inference
        result = inference_detector(model, img_path)

        # Get predictions above threshold
        pred_instances = result.pred_instances
        scores = pred_instances.scores.cpu().numpy()
        bboxes = pred_instances.bboxes.cpu().numpy()
        labels = pred_instances.labels.cpu().numpy()

        # Filter by score threshold
        mask = scores >= args.score_thr
        filtered_scores = scores[mask]
        filtered_bboxes = bboxes[mask]

        print(f"  Detections: {len(filtered_scores)}")
        for i, (bbox, score) in enumerate(zip(filtered_bboxes, filtered_scores)):
            print(f"    [{i+1}] bbox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}], score: {score:.3f}")

        # Visualize and save
        img = mmcv.imread(img_path, channel_order='rgb')
        visualizer.add_datasample(
            name=img_file,
            image=img,
            data_sample=result,
            draw_gt=False,
            pred_score_thr=args.score_thr,
            show=False,
            out_file=os.path.join(args.output_dir, img_file)
        )

    print("\n" + "=" * 50)
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
