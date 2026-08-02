from ultralytics import YOLO
import torch
from multiprocessing import freeze_support
import os
import shutil


def main():
    freeze_support()

    t_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(t_dir)

    print("=" * 80)
    print("t数据集从头训练 - YOLOv8m 检测模型")
    print("=" * 80)
    print("\n数据集配置:")
    print(f"- 数据集路径: {t_dir}")
    print("- 输入: 3通道 (RGB)")
    print("- 任务: 目标检测")
    print("- 类别数: 3")
    print("- 类别: grow_tobacco, disease_tobacco, others")
    print("- 预训练: YOLOv8m 原始COCO预训练")
    print(f"- 数据: train=89 / val=11 / test=12")
    print()

    yaml_path = os.path.join(t_dir, "data", "data.yaml")

    if not os.path.exists(yaml_path):
        print(f"错误: 数据集配置文件不存在: {yaml_path}")
        return

    print(f"加载YOLOv8m预训练模型")
    model = YOLO(os.path.join(base_dir, "yolov8m.pt"))

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
        mixup=0.15,
        copy_paste=0.1,
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
        cos_lr=True,
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        val=True,
        plots=True,
        save=True,
        save_period=10,
        close_mosaic=15,
    )

    print("\n" + "=" * 80)
    print("从头训练完成!")
    best_model = os.path.join(t_dir, "t_train_scratch", "weights", "best.pt")
    print(f"最优模型: {best_model}")

    dst_path = os.path.join(base_dir, "models", "best_t_scratch.pt")
    if os.path.exists(best_model):
        shutil.copy2(best_model, dst_path)
        print(f"模型已复制到: {dst_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
