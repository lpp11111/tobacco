import os
import shutil
import yaml
import torch
from multiprocessing import freeze_support
from ultralytics import YOLO


def seg_to_det(label_file, output_file, class_mapping):
    """Convert YOLO segmentation labels to detection labels with class mapping"""
    lines = []
    if not os.path.exists(label_file):
        with open(output_file, 'w') as f:
            pass
        return
    with open(label_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue

            cls = int(parts[0])
            coords = [float(x) for x in parts[1:]]

            if len(coords) >= 8:
                xs = coords[0::2]
                ys = coords[1::2]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                w = x_max - x_min
                h = y_max - y_min
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
            elif len(coords) >= 4:
                x_center, y_center, w, h = coords[0], coords[1], coords[2], coords[3]
            else:
                continue

            new_cls = class_mapping.get(cls, -1)
            if new_cls < 0:
                continue

            lines.append(f"{new_cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    with open(output_file, 'w') as f:
        if lines:
            f.write('\n'.join(lines) + '\n')


def create_finetune_dataset():
    """Create fine-tuning dataset with unified categories (2 classes matching medium model)"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    t_dir = os.path.join(base_dir, 't')
    output_dir = os.path.join(base_dir, 'finetune', 'finetune_dataset')

    # Clear any previous stale outputs so we truly start from data
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    class_mapping = {
        0: 0,  # grow_tobacco -> healthy
        1: 1,  # disease_tobacco -> disease
        # 2=others is skipped
    }

    splits = ['train', 'val']

    for split in splits:
        src_img_dir = os.path.join(t_dir, 'data', split, 'images')
        src_label_dir = os.path.join(t_dir, 'data', split, 'labels')
        dst_img_dir = os.path.join(output_dir, split, 'images')
        dst_label_dir = os.path.join(output_dir, split, 'labels')

        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_label_dir, exist_ok=True)

        img_files = sorted([f for f in os.listdir(src_img_dir)
                          if f.endswith(('.png', '.jpg', '.jpeg'))])

        converted_count = 0
        for img_file in img_files:
            img_path = os.path.join(src_img_dir, img_file)
            label_file = os.path.splitext(img_file)[0] + '.txt'
            src_label_path = os.path.join(src_label_dir, label_file)
            dst_img_path = os.path.join(dst_img_dir, img_file)
            dst_label_path = os.path.join(dst_label_dir, label_file)

            shutil.copy2(img_path, dst_img_path)

            seg_to_det(src_label_path, dst_label_path, class_mapping)
            if os.path.exists(dst_label_path) and os.path.getsize(dst_label_path) > 0:
                converted_count += 1

        print(f"  {split}: {len(img_files)} images, {converted_count} labels converted")

    data_yaml = {
        'path': output_dir,
        'train': 'train/images',
        'val': 'val/images',
        'nc': 2,
        'names': ['healthy', 'disease'],
        'ch': 3,
    }

    data_yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

    print(f"\n数据集已创建: {output_dir}")
    print(f"data.yaml: {data_yaml_path}")

    return output_dir


def evaluate_model(model_path, data_yaml, split='val'):
    """Evaluate model on dataset"""
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, split=split, verbose=False)

    results = {
        'mAP50': float(metrics.box.map50) if hasattr(metrics.box, 'map50') else 0,
        'mAP50_95': float(metrics.box.map) if hasattr(metrics.box, 'map') else 0,
        'precision': float(metrics.box.mp) if hasattr(metrics.box, 'mp') else 0,
        'recall': float(metrics.box.mr) if hasattr(metrics.box, 'mr') else 0,
    }

    return results


def main():
    freeze_support()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print("=" * 70)
    print("中尺度模型微调训练脚本 (2类: healthy, disease) —  从数据重建开始")
    print("=" * 70)

    print("\n[Step 1] 创建微调数据集（2类 + 格式转换）")
    print("-" * 50)
    dataset_dir = create_finetune_dataset()

    data_yaml = os.path.join(dataset_dir, 'data.yaml')

    print("\n[Step 2] 加载中尺度模型进行微调")
    print("-" * 50)

    medium_model_path = os.path.join(base_dir, 'models', 'best_medium.pt')

    if os.path.exists(medium_model_path):
        print(f"  加载中尺度模型: {medium_model_path}")
        model = YOLO(medium_model_path)
    else:
        print(f"  警告: 中尺度模型不存在 ({medium_model_path})")
        print(f"  使用YOLOv8m基础模型")
        model = YOLO('yolov8m.pt')

    print("\n[Step 3] 在t数据集上微调")
    print("-" * 50)

    finetune_output = os.path.join(base_dir, 'finetune', 'finetune_medium')
    # Clear previous finetune outputs to truly restart
    if os.path.exists(finetune_output):
        shutil.rmtree(finetune_output)
    os.makedirs(finetune_output, exist_ok=True)

    device = 0 if torch.cuda.is_available() else "cpu"

    model.train(
        data=data_yaml,
        epochs=50,
        patience=15,
        imgsz=640,
        batch=4,
        workers=0,
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        cos_lr=True,
        mosaic=1.0,
        mixup=0.15,
        close_mosaic=10,
        box=0.85,
        cls=0.6,
        dfl=1.0,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0005,
        flipud=0.3,
        fliplr=0.5,
        val=True,
        plots=True,
        save=True,
        save_period=10,
        project=finetune_output,
        name='finetune_medium',
        exist_ok=True,
        device=device,
        verbose=True,
    )

    print(f"  微调完成")

    print("\n[Step 4] 评估微调后模型的精度")
    print("-" * 50)

    finetuned_model_path = os.path.join(finetune_output, 'finetune_medium', 'weights', 'best.pt')

    if os.path.exists(finetuned_model_path):
        after_metrics = evaluate_model(finetuned_model_path, data_yaml)
        print(f"  mAP50:      {after_metrics['mAP50']:.4f}")
        print(f"  mAP50-95:   {after_metrics['mAP50_95']:.4f}")
        print(f"  Precision:  {after_metrics['precision']:.4f}")
        print(f"  Recall:     {after_metrics['recall']:.4f}")

        dst_path = os.path.join(base_dir, 'models', 'best_medium_finetuned.pt')
        shutil.copy2(finetuned_model_path, dst_path)
        print(f"\n  微调后模型已保存: {dst_path}")
    else:
        print("  警告: 未找到微调后的模型文件")

    print("\n" + "=" * 70)
    print("微调训练完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()
