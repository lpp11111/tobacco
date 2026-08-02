import os
import yaml
import json
from collections import Counter

def parse_yolo_label(line):
    parts = line.strip().split()
    if len(parts) >= 5:
        cls = int(parts[0])
        return cls
    return None

def count_labels(labels_dir):
    class_counts = Counter()
    total_labels = 0
    num_files = 0
    
    if not os.path.exists(labels_dir):
        return class_counts, total_labels, num_files
    
    for fname in os.listdir(labels_dir):
        if not fname.endswith('.txt'):
            continue
        num_files += 1
        fpath = os.path.join(labels_dir, fname)
        with open(fpath, 'r') as f:
            for line in f:
                cls = parse_yolo_label(line)
                if cls is not None:
                    class_counts[cls] += 1
                    total_labels += 1
    
    return class_counts, total_labels, num_files

def get_dataset_info(data_dir):
    yaml_path = os.path.join(data_dir, 'data.yaml')
    if not os.path.exists(yaml_path):
        return None
    
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return {
        'nc': config.get('nc', 0),
        'names': config.get('names', []),
    }

def print_dataset_stats(name, data_dir, output_dir):
    info = get_dataset_info(data_dir)
    if info is None:
        print(f"未找到 data.yaml")
        return
    
    names = info['names']
    nc = info['nc']
    
    result = {
        'dataset': name,
        'nc': nc,
        'names': names,
        'splits': {}
    }
    
    splits = {
        'train': 'train/labels',
        'valid': 'valid/labels',
        'test': 'test/labels'
    }
    
    total_class_counts = Counter()
    total_all = 0
    total_files = 0
    
    print("=" * 60)
    print(f"  数据集: {name}")
    print(f"  类别数: {nc}")
    print(f"  类别: {names}")
    print("=" * 60)
    
    for split_name, label_path in splits.items():
        full_path = os.path.join(data_dir, label_path)
        class_counts, count, files = count_labels(full_path)
        
        split_data = {
            'files': files,
            'labels': count,
            'class_counts': {}
        }
        
        if count > 0:
            print(f"\n  [{split_name}] ({files} 张图片, {count} 个标注)")
            print(f"  {'类别':<30} {'数量':>8} {'百分比':>10}")
            print(f"  {'-'*50}")
        
        for cls_idx in range(nc):
            cls_name = names[cls_idx] if cls_idx < len(names) else f'class_{cls_idx}'
            cls_count = class_counts.get(cls_idx, 0)
            pct = round(cls_count / count * 100, 1) if count > 0 else 0
            split_data['class_counts'][cls_name] = {
                'count': cls_count,
                'percentage': pct
            }
            if count > 0:
                print(f"  {cls_name:<30} {cls_count:>8} {pct:>9.1f}%")
        
        result['splits'][split_name] = split_data
        total_class_counts.update(class_counts)
        total_all += count
        total_files += files
    
    result['total'] = {
        'files': total_files,
        'labels': total_all,
        'class_counts': {}
    }
    
    print(f"\n  {'='*60}")
    print(f"  [总计] ({total_files} 张图片, {total_all} 个标注)")
    print(f"  {'类别':<30} {'数量':>8} {'百分比':>10}")
    print(f"  {'-'*50}")
    
    for cls_idx in range(nc):
        cls_name = names[cls_idx] if cls_idx < len(names) else f'class_{cls_idx}'
        cls_count = total_class_counts.get(cls_idx, 0)
        pct = round(cls_count / total_all * 100, 1) if total_all > 0 else 0
        result['total']['class_counts'][cls_name] = {
            'count': cls_count,
            'percentage': pct
        }
        print(f"  {cls_name:<30} {cls_count:>8} {pct:>9.1f}%")
    
    output_json = os.path.join(output_dir, f'stats_{name}.json')
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {output_json}")

if __name__ == '__main__':
    dataset_name = 'medium'
    data_dir = r'e:\tobacco\train\data\medium'
    output_dir = r'e:\tobacco\train\statistics'
    
    print_dataset_stats(dataset_name, data_dir, output_dir)
