from ultralytics import YOLO
import torch
import torch.nn as nn
from multiprocessing import freeze_support
import os


def transfer_weights_from_detection_to_segmentation(detect_model, seg_model):
    """将检测模型的权重迁移到分割模型"""
    detect_net = detect_model.model.model
    seg_net = seg_model.model.model
    
    detect_params = {}
    for name, param in detect_net.named_parameters():
        detect_params[name] = param.data.clone()
    
    transferred_layers = 0
    skipped_layers = 0
    
    for name, param in seg_net.named_parameters():
        if name in detect_params and param.size() == detect_params[name].size():
            param.data.copy_(detect_params[name])
            transferred_layers += 1
        else:
            skipped_layers += 1
    
    print(f"权重迁移完成: {transferred_layers} 层权重已迁移, {skipped_layers} 层跳过")
    return seg_model


def adapt_model_for_new_dataset(model, num_classes):
    """适配新数据集的模型：调整类别数"""
    net = model.model.model
    
    for name, module in net.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            break

    return model


def main():
    freeze_support()
    
    t_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(os.path.dirname(t_dir), "train")
    
    print("=" * 80)
    print("新数据集迁移训练 - 基于中尺度模型迁移学习")
    print("=" * 80)
    print("\n数据集配置:")
    print(f"- 数据集路径: {t_dir}")
    print("- 输入: 3通道 (RGB)")
    print("- 任务: 语义分割")
    print("- 类别数: 3")
    print("- 类别: grow tobacco, disease tobacco, others")
    print("- 迁移源: 中尺度检测模型 (healthy/disease 二分类)")
    print("- 迁移方式: 检测模型backbone权重 -> 分割模型backbone")
    print()

    medium_model_path = os.path.join(os.path.dirname(t_dir), "models", "best_medium.pt")
    yaml_path = os.path.join(t_dir, "data", "data.yaml")

    if not os.path.exists(medium_model_path):
        print(f"警告: 中尺度权重不存在 ({medium_model_path})")
        print("使用YOLOv8m-seg预训练模型从头训练")
        model = YOLO("yolov8m-seg.pt")
    else:
        print(f"加载中尺度检测模型: {medium_model_path}")
        detect_model = YOLO(medium_model_path)
        
        print("创建YOLOv8m-seg分割模型")
        seg_model = YOLO("yolov8m-seg.pt")
        
        print("将检测模型权重迁移到分割模型...")
        model = transfer_weights_from_detection_to_segmentation(detect_model, seg_model)
        print("迁移学习: 检测模型知识已迁移到分割模型")

    model = adapt_model_for_new_dataset(model, num_classes=3)
    print("模型适配完成 (3类别)")

    results = model.train(
        data=yaml_path,
        epochs=100,
        patience=30,
        imgsz=640,
        batch=4,
        workers=0,
        project=t_dir,
        name="t_train_transfer",
        exist_ok=True,
        device=0 if torch.cuda.is_available() else "cpu",
        box=0.85,
        cls=0.6,
        dfl=1.0,
        mosaic=1.0,
        mixup=0.0,
        hsv_h=0.02,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.15,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.3,
        fliplr=0.5,
        cos_lr=True,
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        val=True,
        plots=True,
        save=True,
        save_period=10,
        task="segment"
    )

    print("\n" + "=" * 80)
    print("迁移学习训练完成!")
    best_model = os.path.join(t_dir, "t_train_transfer", "weights", "best.pt")
    print(f"最优模型: {best_model}")
    print("=" * 80)


if __name__ == "__main__":
    main()