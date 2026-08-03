import os
import cv2
import yaml
import argparse
import random
import numpy as np


def load_config(data_yaml):
    with open(data_yaml, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config.get('names', []), config.get('nc', 0)


def load_labels(label_path):
    labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        labels.append((cls, x_center, y_center, width, height))
    return labels


def draw_boxes(image, labels, class_names):
    h, w = image.shape[:2]
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 255, 0)]

    for cls, x_center, y_center, width, height in labels:
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)

        color = colors[cls % len(colors)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        class_name = class_names[cls] if cls < len(class_names) else f'class_{cls}'
        label = f'{class_name}'
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - label_height - 5), (x1 + label_width, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return image


def main():
    parser = argparse.ArgumentParser(description='检查t数据集标注 (nano训练 - 缺苗数识别)')
    parser.add_argument('--split', default='train', help='数据集分割 (train/val/test)')
    parser.add_argument('--max_images', type=int, default=20, help='最大检查图片数')
    parser.add_argument('--output', default='annotation_check_t_n', help='输出目录')
    parser.add_argument('--mode', default='mixed', choices=['mixed', 'positive', 'negative', 'all'],
                        help='采样模式: mixed=混合正负样本, positive=仅正样本, negative=仅负样本, all=全部')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    args = parser.parse_args()
    random.seed(args.seed)

    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_path = os.path.join(project_dir, 't', 'data')
    data_yaml = os.path.join(dataset_path, 'data.yaml')

    class_names, nc = load_config(data_yaml)

    split = args.split
    image_dir = os.path.join(dataset_path, split, 'images')
    label_dir = os.path.join(dataset_path, split, 'labels')
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output, split)
    os.makedirs(output_path, exist_ok=True)

    all_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tif'))]

    positive_files = []
    negative_files = []
    for f in all_files:
        label_file = os.path.splitext(f)[0] + '.txt'
        label_path = os.path.join(label_dir, label_file)
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            positive_files.append(f)
        else:
            negative_files.append(f)

    random.shuffle(positive_files)
    random.shuffle(negative_files)

    max_n = args.max_images
    if args.mode == 'positive':
        selected = positive_files[:max_n]
    elif args.mode == 'negative':
        selected = negative_files[:max_n]
    elif args.mode == 'all':
        selected = all_files[:max_n]
    else:
        pos_count = min(int(max_n * 0.7), len(positive_files))
        neg_count = min(max_n - pos_count, len(negative_files))
        if pos_count == 0 and negative_files:
            neg_count = min(max_n, len(negative_files))
        if neg_count == 0 and positive_files:
            pos_count = min(max_n, len(positive_files))
        selected = positive_files[:pos_count] + negative_files[:neg_count]
        random.shuffle(selected)

    total_images = len(all_files)
    total_labels = 0
    class_counts = {i: 0 for i in range(nc)}

    print(f"\n{'='*60}")
    print(f"数据集: t (YOLOv8n nano训练 - 缺苗数识别)")
    print(f"类别数: {nc}")
    print(f"类别: {class_names}")
    print(f"采样模式: {args.mode}")
    print(f"正样本: {len(positive_files)} 张, 负样本: {len(negative_files)} 张")
    print(f"已选: {len(selected)} 张")
    print(f"{'='*60}")

    for i, image_file in enumerate(selected):
        image_path = os.path.join(image_dir, image_file)
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(label_dir, label_file)

        try:
            image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            image = None
        if image is None:
            print(f"  跳过: 无法读取 {image_file}")
            continue

        labels = load_labels(label_path)

        for cls, _, _, _, _ in labels:
            if cls < nc:
                class_counts[cls] += 1
        total_labels += len(labels)

        image_with_boxes = draw_boxes(image.copy(), labels, class_names)

        sample_type = "pos" if len(labels) > 0 else "neg"
        output_file = os.path.join(output_path, f"{i:02d}_{sample_type}_{image_file}")
        ext = os.path.splitext(output_file)[1]
        success, encoded = cv2.imencode(ext, image_with_boxes)
        if success:
            with open(output_file, 'wb') as f:
                f.write(encoded.tobytes())

        sample_label = "正样本" if len(labels) > 0 else "负样本"
        print(f"  [{i+1}/{len(selected)}] {image_file}: {len(labels)} 个标注 ({sample_label})")

    print(f"\n{'='*60}")
    print(f"统计结果:")
    print(f"  图片总数: {total_images}")
    print(f"  正样本数: {len(positive_files)}")
    print(f"  负样本数: {len(negative_files)}")
    print(f"  标注总数: {total_labels}")
    print(f"  类别分布:")
    for i in range(nc):
        print(f"    {i}: {class_names[i]} = {class_counts[i]}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
