import os
import cv2
import numpy as np
import pandas as pd
import argparse
from ultralytics import YOLO


def load_labels(label_path):
    labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cls = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    
                    if len(coords) >= 8 and len(coords) % 2 == 0:
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
                    
                    labels.append((cls, x_center, y_center, w, h))
    return labels


def calculate_iou(box1, box2):
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2
    
    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def main():
    parser = argparse.ArgumentParser(description='检查large模型迁移能力')
    parser.add_argument('--split', default='valid', help='数据集分割 (train/val/valid)')
    parser.add_argument('--iou_threshold', type=float, default=0.5, help='IoU阈值')
    parser.add_argument('--conf_threshold', type=float, default=0.25, help='置信度阈值')
    
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    t_dataset_path = os.path.join(base_dir, 't', 'data')
    model_path = os.path.join(base_dir, 'models', 'best_large.pt')
    
    t_classes = ["grow tobacco", "disease tobacco", "others"]
    t_nc = 3
    
    split_map = {'valid': 'val'}
    split = split_map.get(args.split, args.split)
    
    image_dir = os.path.join(t_dataset_path, split, 'images')
    label_dir = os.path.join(t_dataset_path, split, 'labels')
    
    print(f"\n{'='*70}")
    print(f"迁移能力检查 - large模型 -> t数据集")
    print(f"数据集分割: {split}, IoU阈值: {args.iou_threshold}, 置信度阈值: {args.conf_threshold}")
    print(f"{'='*70}")
    
    print("\n类别映射规则 (large -> t):")
    print("  large(1类): ['tobacco']")
    print("  t(3类): ['grow tobacco', 'disease tobacco', 'others']")
    print("  0(tobacco) -> 0(grow tobacco)")
    print("  其他 -> 2(others)")
    
    print(f"\n加载模型: {model_path}")
    model = YOLO(model_path)
    print("  模型加载成功")
    
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    results = {
        'model': 'large',
        'images': len(image_files),
        'tp': 0,
        'fp': 0,
        'fn': 0,
        'total_gt': 0,
        'total_pred': 0,
        'class_tp': [0] * t_nc,
        'class_fp': [0] * t_nc,
        'class_fn': [0] * t_nc,
        'class_gt': [0] * t_nc,
    }
    
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(label_dir, label_file)
        
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        gt_labels = load_labels(label_path)
        
        for cls, _, _, _, _ in gt_labels:
            if cls < t_nc:
                results['class_gt'][cls] += 1
        results['total_gt'] += len(gt_labels)
        
        pred_results = model.predict(image, conf=args.conf_threshold, verbose=False)
        
        predictions = []
        for r in pred_results:
            if r.boxes is not None:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x_center = (box.xywh[0][0] / image.shape[1]).item()
                    y_center = (box.xywh[0][1] / image.shape[0]).item()
                    width = (box.xywh[0][2] / image.shape[1]).item()
                    height = (box.xywh[0][3] / image.shape[0]).item()
                    mapped_cls = 0 if cls == 0 else 2
                    predictions.append((mapped_cls, x_center, y_center, width, height, conf))
        
        results['total_pred'] += len(predictions)
        
        matched_gt = [False] * len(gt_labels)
        matched_pred = [False] * len(predictions)
        
        for i, (gt_cls, gt_x, gt_y, gt_w, gt_h) in enumerate(gt_labels):
            for j, (pred_cls, pred_x, pred_y, pred_w, pred_h, _) in enumerate(predictions):
                if not matched_gt[i] and not matched_pred[j]:
                    iou = calculate_iou((gt_x, gt_y, gt_w, gt_h), (pred_x, pred_y, pred_w, pred_h))
                    if iou >= args.iou_threshold and gt_cls == pred_cls:
                        matched_gt[i] = True
                        matched_pred[j] = True
                        results['tp'] += 1
                        if gt_cls < t_nc:
                            results['class_tp'][gt_cls] += 1
        
        for i, matched in enumerate(matched_gt):
            if not matched:
                gt_cls = gt_labels[i][0]
                results['fn'] += 1
                if gt_cls < t_nc:
                    results['class_fn'][gt_cls] += 1
        
        for j, matched in enumerate(matched_pred):
            if not matched:
                pred_cls = predictions[j][0]
                results['fp'] += 1
                if pred_cls < t_nc:
                    results['class_fp'][pred_cls] += 1
    
    precision = results['tp'] / (results['tp'] + results['fp']) if (results['tp'] + results['fp']) > 0 else 0
    recall = results['tp'] / (results['tp'] + results['fn']) if (results['tp'] + results['fn']) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n总体指标:")
    print(f"  图片数: {results['images']}")
    print(f"  真实标注: {results['total_gt']}")
    print(f"  预测标注: {results['total_pred']}")
    print(f"  TP: {results['tp']}, FP: {results['fp']}, FN: {results['fn']}")
    print(f"  精确率: {precision:.4f}")
    print(f"  召回率: {recall:.4f}")
    print(f"  F1值: {f1:.4f}")
    
    print(f"\n各类别指标:")
    for i in range(t_nc):
        cls_tp = results['class_tp'][i]
        cls_fp = results['class_fp'][i]
        cls_fn = results['class_fn'][i]
        cls_gt = results['class_gt'][i]
        cls_precision = cls_tp / (cls_tp + cls_fp) if (cls_tp + cls_fp) > 0 else 0
        cls_recall = cls_tp / (cls_tp + cls_fn) if (cls_tp + cls_fn) > 0 else 0
        cls_f1 = 2 * cls_precision * cls_recall / (cls_precision + cls_recall) if (cls_precision + cls_recall) > 0 else 0
        print(f"  {i}({t_classes[i]}): GT={cls_gt}, TP={cls_tp}, FP={cls_fp}, FN={cls_fn}, P={cls_precision:.4f}, R={cls_recall:.4f}, F1={cls_f1:.4f}")
    
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'check', 'transfer_check')
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.DataFrame([{
        '模型': 'large',
        '图片数': results['images'],
        '真实标注': results['total_gt'],
        '预测标注': results['total_pred'],
        'TP': results['tp'],
        'FP': results['fp'],
        'FN': results['fn'],
        '精确率': precision,
        '召回率': recall,
        'F1值': f1,
    }])
    
    csv_path = os.path.join(output_dir, 'transfer_results_large.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到: {csv_path}")


if __name__ == '__main__':
    main()
