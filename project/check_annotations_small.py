import os
import cv2
import yaml
import argparse


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
    parser = argparse.ArgumentParser(description='检查small数据集标注')
    parser.add_argument('--split', default='train', help='数据集分割 (train/val/valid/test)')
    parser.add_argument('--max_images', type=int, default=20, help='最大检查图片数')
    parser.add_argument('--output', default='annotation_check_small', help='输出目录')
    
    args = parser.parse_args()
    
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'small')
    data_yaml = os.path.join(dataset_path, 'data.yaml')
    
    class_names, nc = load_config(data_yaml)
    
    split_map = {'val': 'valid'}
    split = split_map.get(args.split, args.split)
    
    image_dir = os.path.join(dataset_path, split, 'images')
    label_dir = os.path.join(dataset_path, split, 'labels')
    project_path = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(project_path, args.output, split)
    os.makedirs(output_path, exist_ok=True)
    
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tif'))])
    
    print(f"\n{'='*60}")
    print(f"数据集: small")
    print(f"类别数: {nc}")
    print(f"类别: {class_names}")
    print(f"{'='*60}")
    
    total_images = len(image_files)
    total_labels = 0
    class_counts = {i: 0 for i in range(nc)}
    
    for i, image_file in enumerate(image_files[:args.max_images]):
        image_path = os.path.join(image_dir, image_file)
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(label_dir, label_file)
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"  跳过: 无法读取 {image_file}")
            continue
        
        labels = load_labels(label_path)
        
        for cls, _, _, _, _ in labels:
            if cls < nc:
                class_counts[cls] += 1
        total_labels += len(labels)
        
        image_with_boxes = draw_boxes(image.copy(), labels, class_names)
        
        output_file = os.path.join(output_path, image_file)
        cv2.imwrite(output_file, image_with_boxes)
        
        print(f"  [{i+1}/{total_images}] {image_file}: {len(labels)} 个标注")
    
    print(f"\n{'='*60}")
    print(f"统计结果:")
    print(f"  图片总数: {total_images}")
    print(f"  标注总数: {total_labels}")
    print(f"  类别分布:")
    for i in range(nc):
        print(f"    {i}: {class_names[i]} = {class_counts[i]}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
