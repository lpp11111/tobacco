import os
import pandas as pd
import argparse
import tempfile
import shutil
import yaml
from ultralytics import YOLO


def create_temp_data_yaml_and_labels(t_dataset_path, model_classes, class_mapping, split='valid'):
    """Create a temporary dataset with remapped labels and data.yaml"""
    tmp_dir = os.path.join(tempfile.gettempdir(), 'tobacco_eval')
    os.makedirs(tmp_dir, exist_ok=True)
    
    for data_split in ['train', split]:
        src_img_dir = os.path.join(t_dataset_path, data_split, 'images')
        src_label_dir = os.path.join(t_dataset_path, data_split, 'labels')
        dst_img_dir = os.path.join(tmp_dir, data_split, 'images')
        dst_label_dir = os.path.join(tmp_dir, data_split, 'labels')
        
        if not os.path.exists(src_img_dir):
            continue
        
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_label_dir, exist_ok=True)
        
        for img_file in os.listdir(src_img_dir):
            if img_file.endswith(('.png', '.jpg', '.jpeg')):
                src_img = os.path.join(src_img_dir, img_file)
                dst_img = os.path.join(dst_img_dir, img_file)
                if not os.path.exists(dst_img):
                    shutil.copy2(src_img, dst_img)
        
        for label_file in os.listdir(src_label_dir):
            if label_file.endswith('.txt') and not label_file.endswith('.cache'):
                src_label = os.path.join(src_label_dir, label_file)
                dst_label = os.path.join(dst_label_dir, label_file)
                
                lines = []
                with open(src_label, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) < 2:
                            continue
                        orig_cls = int(parts[0])
                        new_cls = class_mapping.get(orig_cls, -1)
                        if new_cls >= 0 and new_cls < len(model_classes):
                            parts[0] = str(new_cls)
                            lines.append(' '.join(parts))
                
                with open(dst_label, 'w') as f:
                    f.write('\n'.join(lines) + '\n')
    
    data_yaml = {
        'path': tmp_dir.replace('\\', '/'),
        'train': 'train/images',
        'val': f'{split}/images',
        'nc': len(model_classes),
        'names': model_classes,
    }
    
    yaml_path = os.path.join(tmp_dir, 'eval_data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)
    
    return yaml_path


def evaluate_model_val(model_path, t_dataset_path, model_name, model_classes, 
                        class_mapping, model_type='det', split='valid', imgsz=640, batch=4):
    """Evaluate model using model.val() for proper metrics"""
    print(f"\n加载模型: {model_path}")
    if not os.path.exists(model_path):
        print(f"  警告: 模型文件不存在，跳过")
        return None
    
    model = YOLO(model_path)
    print(f"  模型加载成功 (类型: {model_type})")
    
    yaml_path = create_temp_data_yaml_and_labels(t_dataset_path, model_classes, class_mapping, split)
    print(f"  配置: {yaml_path}")
    
    try:
        metrics = model.val(data=yaml_path, imgsz=imgsz, batch=batch, verbose=False, plots=False)
    except Exception as e:
        print(f"  ⚠️  验证出错: {e}")
        return {
            'model': model_name,
            'type': model_type,
            'precision': 0,
            'recall': 0,
            'box_map50': 0,
            'box_map50_95': 0,
            'mask_precision': 0,
            'mask_recall': 0,
            'mask_map50': 0,
            'mask_map50_95': 0,
            'error': str(e),
        }
    
    results = {
        'model': model_name,
        'type': model_type,
        'precision': float(metrics.box.mp) if hasattr(metrics, 'box') else 0,
        'recall': float(metrics.box.mr) if hasattr(metrics, 'box') else 0,
        'box_map50': float(metrics.box.map50) if hasattr(metrics.box, 'map50') else 0,
        'box_map50_95': float(metrics.box.map) if hasattr(metrics.box, 'map') else 0,
    }
    
    if model_type == 'seg' and hasattr(metrics, 'mask'):
        results['mask_precision'] = float(metrics.mask.mp) if hasattr(metrics.mask, 'mp') else 0
        results['mask_recall'] = float(metrics.mask.mr) if hasattr(metrics.mask, 'mr') else 0
        results['mask_map50'] = float(metrics.mask.map50) if hasattr(metrics.mask, 'map50') else 0
        results['mask_map50_95'] = float(metrics.mask.map) if hasattr(metrics.mask, 'map') else 0
    else:
        results['mask_precision'] = 0
        results['mask_recall'] = 0
        results['mask_map50'] = 0
        results['mask_map50_95'] = 0
    
    print(f"\n  {model_name} ({model_type}) 验证结果:")
    print(f"  Box - Precision: {results['precision']:.4f}, Recall: {results['recall']:.4f}")
    print(f"  Box - mAP50: {results['box_map50']:.4f}, mAP50-95: {results['box_map50_95']:.4f}")
    if model_type == 'seg':
        print(f"  Mask - Precision: {results['mask_precision']:.4f}, Recall: {results['mask_recall']:.4f}")
        print(f"  Mask - mAP50: {results['mask_map50']:.4f}, mAP50-95: {results['mask_map50_95']:.4f}")
    
    return results


def main():
    tmp_dir = os.path.join(tempfile.gettempdir(), 'tobacco_eval')
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)
    
    parser = argparse.ArgumentParser(description='检查t数据集模型性能 - 综合评估')
    parser.add_argument('--split', default='valid', help='数据集分割 (train/val/valid)')
    parser.add_argument('--imgsz', type=int, default=640, help='图像尺寸')
    parser.add_argument('--batch', type=int, default=4, help='批大小')
    parser.add_argument('--model', default='all', help='评估的模型 (all/medium/small/large/scratch/transfer/finetune)')
    
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    t_dataset_path = os.path.join(base_dir, 't', 'data')
    models_dir = os.path.join(base_dir, 'models')
    
    split_map = {'valid': 'val'}
    split = split_map.get(args.split, args.split)
    
    print(f"\n{'='*70}")
    print(f"t数据集模型性能综合评估")
    print(f"数据集路径: {t_dataset_path}")
    print(f"数据集分割: {split}, imgsz: {args.imgsz}, batch: {args.batch}")
    print(f"{'='*70}")
    
    models_config = {
        'extra_small': {
            'path': os.path.join(models_dir, 'best_extra_small.pt'),
            'classes': ['healthy', 'disease'],
            'type': 'det',
            'mapping': {0: 0, 1: 1, 2: -1},  # t: 0->healthy, 1->disease, 2->ignore
        },
        'medium': {
            'path': os.path.join(models_dir, 'best_medium.pt'),
            'classes': ['healthy', 'disease'],
            'type': 'det',
            'mapping': {0: 0, 1: 1, 2: -1},  # t: 0->healthy, 1->disease, 2->ignore
        },
        'small': {
            'path': os.path.join(models_dir, 'best_small.pt'),
            'classes': ['light_disease', 'healthy', 'severe_disease', 'moderate_disease'],
            'type': 'seg',
            'mapping': {0: 1, 1: 0, 2: 1, 3: 1},
        },
        'large': {
            'path': os.path.join(models_dir, 'best_large.pt'),
            'classes': ['tobacco'],
            'type': 'seg',
            'mapping': {0: 0, 1: 0, 2: -1},
        },
        'scratch': {
            'path': os.path.join(base_dir, 't', 't_train_scratch', 'weights', 'best.pt'),
            'classes': ['grow_tobacco', 'disease_tobacco', 'others'],
            'type': 'det',
            'mapping': {0: 0, 1: 1, 2: 2},
        },
        'transfer': {
            'path': os.path.join(base_dir, 't', 't_train_transfer', 'weights', 'best.pt'),
            'classes': ['grow_tobacco', 'disease_tobacco', 'others'],
            'type': 'det',
            'mapping': {0: 0, 1: 1, 2: 2},
        },
        'finetune': {
            'path': os.path.join(models_dir, 'best_medium_finetuned.pt'),
            'classes': ['healthy', 'disease'],
            'type': 'det',
            'mapping': {0: 0, 1: 1, 2: -1},
        },
        't_nano': {
            'path': os.path.join(models_dir, 'best_t_nano.pt'),
            'classes': ['grow_tobacco', 'disease_tobacco', 'others'],
            'type': 'det',
            'mapping': {0: 0, 1: 1, 2: 2},
        },
        't_m': {
            'path': os.path.join(models_dir, 'best_t_m.pt'),
            'classes': ['grow_tobacco', 'disease_tobacco', 'others'],
            'type': 'det',
            'mapping': {0: 0, 1: 1, 2: 2},
        },
    }
    
    all_results = []
    det_results = []
    seg_results = []
    
    models_to_eval = list(models_config.keys()) if args.model == 'all' else [args.model]
    
    for model_name in models_to_eval:
        if model_name not in models_config:
            print(f"未知模型: {model_name}")
            continue
        
        config = models_config[model_name]
        
        print(f"\n{'='*50}")
        print(f"评估模型: {model_name} ({config['type']})")
        print(f"  类别: {config['classes']}")
        print(f"{'='*50}")
        
        results = evaluate_model_val(
            model_path=config['path'],
            t_dataset_path=t_dataset_path,
            model_name=model_name,
            model_classes=config['classes'],
            class_mapping=config['mapping'],
            model_type=config['type'],
            split=split,
            imgsz=args.imgsz,
            batch=args.batch,
        )
        
        if results:
            all_results.append(results)
            if config['type'] == 'seg':
                seg_results.append(results)
            else:
                det_results.append(results)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'check', 'transfer_check')
    os.makedirs(output_dir, exist_ok=True)
    
    if det_results:
        print(f"\n{'='*70}")
        print(f"📊 检测模型对比 (Detection Models):")
        print(f"{'='*70}")
        det_df = pd.DataFrame([{
            '模型': r['model'],
            '类型': r['type'],
            'Box_Precision': r['precision'],
            'Box_Recall': r['recall'],
            'Box_mAP50': r['box_map50'],
            'Box_mAP50-95': r['box_map50_95'],
        } for r in det_results])
        print(det_df.to_string(index=False))
        
        det_csv = os.path.join(output_dir, 'det_models_comparison.csv')
        det_df.to_csv(det_csv, index=False, encoding='utf-8-sig')
        print(f"\n检测结果已保存到: {det_csv}")
    
    if seg_results:
        print(f"\n{'='*70}")
        print(f"🎨 分割模型对比 (Segmentation Models):")
        print(f"{'='*70}")
        seg_df = pd.DataFrame([{
            '模型': r['model'],
            '类型': r['type'],
            'Box_Precision': r['precision'],
            'Box_Recall': r['recall'],
            'Box_mAP50': r['box_map50'],
            'Mask_Precision': r['mask_precision'],
            'Mask_Recall': r['mask_recall'],
            'Mask_mAP50': r['mask_map50'],
            'Mask_mAP50-95': r['mask_map50_95'],
        } for r in seg_results])
        print(seg_df.to_string(index=False))
        
        seg_csv = os.path.join(output_dir, 'seg_models_comparison.csv')
        seg_df.to_csv(seg_csv, index=False, encoding='utf-8-sig')
        print(f"\n分割结果已保存到: {seg_csv}")
    
    # 生成各模型的单独评估 CSV 到 check/transfer_check/
    print(f"\n{'='*70}")
    print(f"📁 生成各模型单独评估结果")
    print(f"{'='*70}")
    for result in all_results:
        model_name = result['model']
        
        single_df = pd.DataFrame([{
            '模型': result['model'],
            '类型': result['type'],
            'Box_Precision': result['precision'],
            'Box_Recall': result['recall'],
            'Box_mAP50': result['box_map50'],
            'Box_mAP50-95': result['box_map50_95'],
            'Mask_Precision': result.get('mask_precision', 0),
            'Mask_Recall': result.get('mask_recall', 0),
            'Mask_mAP50': result.get('mask_map50', 0),
            'Mask_mAP50-95': result.get('mask_map50_95', 0),
        }])
        
        csv_path = os.path.join(output_dir, f'transfer_results_{model_name}.csv')
        single_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"  transfer_results_{model_name}.csv ✓")
    
    if all_results:
        print(f"\n{'='*70}")
        print(f"📋 总结与建议:")
        print(f"{'='*70}")
        
        if det_results:
            best_det = max(det_results, key=lambda x: x['box_map50'])
            print(f"\n  🏆 最佳检测模型: {best_det['model']}")
            print(f"     Box mAP50: {best_det['box_map50']:.4f}")
        
        if seg_results:
            best_seg = max(seg_results, key=lambda x: x['mask_map50'])
            print(f"\n  🏆 最佳分割模型: {best_seg['model']}")
            print(f"     Mask mAP50: {best_seg['mask_map50']:.4f}")
        
        print(f"\n  💡 使用建议:")
        print(f"     - 若需要检测功能: 使用 finetune 模型 (最佳检测性能)")
        print(f"     - 若需要分割功能: 使用 scratch 或 transfer 模型")
        print(f"     - 检测 + 分割联合任务: 推荐先检测(finetune)再分割(seg模型)")
        
        print(f"\n使用说明:")
        print(f"  python check/check_transfer_t.py --model all          # 评估所有模型")
        print(f"  python check/check_transfer_t.py --model finetune     # 仅评估finetune模型")
        print(f"  python check/check_transfer_t.py --model scratch     # 仅评估scratch模型")
        print(f"  python check/check_transfer_t.py --split train        # 评估训练集")
        print(f"  python check/check_transfer_t.py --imgsz 1024         # 使用1024尺寸")


if __name__ == '__main__':
    main()
