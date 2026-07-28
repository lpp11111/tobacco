from ultralytics import YOLO
import torch
import torch.nn as nn
from multiprocessing import freeze_support
import os


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
    
    print("=" * 80)
    print("新数据集从头训练 - YOLOv8m-seg原始预训练")
    print("=" * 80)
    print("\n数据集配置:")
    print(f"- 数据集路径: {t_dir}")
    print("- 输入: 3通道 (RGB)")
    print("- 任务: 语义分割")
    print("- 类别数: 3")
    print("- 类别: grow tobacco, disease tobacco, others")
    print("- 预训练: YOLOv8m-seg原始COCO预训练")
    print()

    yaml_path = os.path.join(t_dir, "data", "data.yaml")

    print(f"加载YOLOv8m-seg原始预训练模型")
    model = YOLO("yolov8m-seg.pt")

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
        name="t_train_scratch",
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
    print("从头训练完成!")
    best_model = os.path.join(t_dir, "t_train_scratch", "weights", "best.pt")
    print(f"最优模型: {best_model}")
    print("=" * 80)


if __name__ == "__main__":
    main()