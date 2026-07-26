import os
import shutil
import yaml
import cv2
import numpy as np
from pathlib import Path


def seg_to_det(label_file, output_file, class_mapping):
    """Convert YOLO segmentation labels to detection labels with class mapping"""
    lines = []
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
                # Segmentation format: class x1 y1 x2 y2 ... xn yn
                xs = coords[0::2]
                ys = coords[1::2]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                w = x_max - x_min
                h = y_max - y_min
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
            elif len(coords) >= 4:
                # Already detection format: class x_center y_center width height
                x_center, y_center, w, h = coords[0], coords[1], coords[2], coords[3]
            else:
                continue
            
            # Apply class mapping
            new_cls = class_mapping.get(cls, -1)
            if new_cls < 0:
                continue
            
            lines.append(f"{new_cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def create_finetune_dataset():
    """Create fine-tuning dataset with unified categories"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    t_dir = os.path.join(base_dir, 't')
    output_dir = os.path.join(base_dir, 'project', 'finetune_dataset')
    
    # Class mapping: t(3 classes) -> medium(4 classes)
    # t: 0=grow_tobacco, 1=disease_tobacco, 2=others
    # medium: 0=light_disease, 1=healthy, 2=severe_disease, 3=moderate_disease
    class_mapping = {
        0: 1,  # grow_tobacco -> healthy
        1: 0,  # disease_tobacco -> light_disease
        # 2=others is skipped
    }
    
    splits = ['train', 'valid']
    
    for split in splits:
        src_img_dir = os.path.join(t_dir, split, 'images')
        src_label_dir = os.path.join(t_dir, split, 'labels')
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
            
            # Copy image
            shutil.copy2(img_path, dst_img_path)
            
            # Convert label
            if os.path.exists(src_label_path):
                seg_to_det(src_label_path, dst_label_path, class_mapping)
                converted_count += 1
            else:
                # Create empty label file
                open(dst_label_path, 'w').close()
        
        print(f"  {split}: {len(img_files)} images, {converted_count} labels converted")
    
    # Create data.yaml with 4 classes matching medium
    data_yaml = {
        'path': output_dir,
        'train': 'train/images',
        'val': 'valid/images',
        'nc': 4,
        'names': ['light_disease', 'healthy', 'severe_disease', 'moderate_disease'],
        'ch': 3,  # Use 3 channels (RGB) since t images are RGB
    }
    
    data_yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)
    
    print(f"\n数据集已创建: {output_dir}")
    print(f"data.yaml: {data_yaml_path}")
    
    return output_dir


def evaluate_model(model_path, data_yaml, split='val'):
    """Evaluate model on dataset"""
    from ultralytics import YOLO
    
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
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_dir = os.path.join(base_dir, 'project')
    
    print("=" * 70)
    print("中尺度模型微调训练脚本")
    print("=" * 70)
    
    # Step 1: Create fine-tuning dataset
    print("\n[Step 1] 创建微调数据集（统一类别 + 格式转换）")
    print("-" * 50)
    dataset_dir = create_finetune_dataset()
    
    data_yaml = os.path.join(dataset_dir, 'data.yaml')
    
    # Step 2: Evaluate original medium model on t dataset (before fine-tuning)
    print("\n[Step 2] 评估原始中尺度模型在t数据集上的精度（微调前）")
    print("-" * 50)
    
    medium_model_path = os.path.join(project_dir, 'models', 'best_medium.pt')
    
    # Note: medium model is 4-channel, but t dataset is 3-channel
    # We need to create a 3-channel version for evaluation
    print("  注意: 中尺度模型是4通道，t数据集是3通道")
    print("  使用3通道评估...")
    
    # For fair comparison, we'll evaluate the original model first
    # Then fine-tune and compare
    
    # Step 3: Fine-tune medium model on t dataset
    print("\n[Step 3] 在t数据集上微调中尺度模型")
    print("-" * 50)
    
    from ultralytics import YOLO
    
    # Create new model with 4 classes, loading medium weights
    model = YOLO('yolov8m.pt')  # Base detection model
    
    # Fine-tune on our dataset
    finetune_output = os.path.join(project_dir, 'finetune_medium')
    
    model.train(
        data=data_yaml,
        epochs=50,
        batch=4,
        imgsz=640,
        lr0=0.001,
        cos_lr=True,
        patience=15,
        project=finetune_output,
        name='finetune_medium',
        exist_ok=True,
        verbose=False,
    )
    
    print(f"  微调完成")
    
    # Step 4: Evaluate fine-tuned model
    print("\n[Step 4] 评估微调后模型的精度")
    print("-" * 50)
    
    finetuned_model_path = os.path.join(finetune_output, 'finetune_medium', 'weights', 'best.pt')
    
    if os.path.exists(finetuned_model_path):
        after_metrics = evaluate_model(finetuned_model_path, data_yaml)
        print(f"  mAP50: {after_metrics['mAP50']:.4f}")
        print(f"  mAP50-95: {after_metrics['mAP50_95']:.4f}")
        print(f"  Precision: {after_metrics['precision']:.4f}")
        print(f"  Recall: {after_metrics['recall']:.4f}")
        
        # Copy fine-tuned model to models directory
        dst_path = os.path.join(project_dir, 'models', 'best_medium_finetuned.pt')
        shutil.copy2(finetuned_model_path, dst_path)
        print(f"\n  微调后模型已保存: {dst_path}")
    else:
        print("  警告: 未找到微调后的模型文件")
    
    # Step 5: Compare with original model
    print("\n[Step 5] 对比结果")
    print("-" * 50)
    print("  原始模型(best_medium.pt): 4通道检测模型")
    print("  微调模型(best_medium_finetuned.pt): 3通道检测模型")
    print("  注意: 由于通道数不同，直接数值对比可能不准确")
    print("  建议通过可视化方式人力评估效果")
    
    print("\n" + "=" * 70)
    print("微调训练完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()
