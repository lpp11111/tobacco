from ultralytics import YOLO
import torch
from multiprocessing import freeze_support
import os


def main():
    freeze_support()
    project_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(project_dir)

    print("=" * 80)
    print("中尺度模型训练 - 健康/病害二分类")
    print("=" * 80)
    print("\n模型配置:")
    print("- 模型: YOLOv8m (检测模型)")
    print("- 输入: 3通道 (RGB)")
    print("- 任务: 健康/病害二分类检测")
    print("- 类别: ['healthy', 'disease']")
    print("- 优化: 针对中尺度病害检测优化")
    print()

    model_path = os.path.join(base_dir, "yolov8m.pt")
    yaml_path = os.path.join(base_dir, "train", "data", "medium", "data.yaml")

    print(f"加载预训练权重: {model_path}")
    model = YOLO(model_path)

    # 中尺度专用训练参数
    results = model.train(
        data=yaml_path,
        epochs=100,
        patience=30,
        imgsz=640,
        batch=4,
        workers=0,
        project=os.path.join(project_dir, "data_result"),
        name="medium_train",
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
        save_period=10
    )

    print("\n" + "=" * 80)
    print("中尺度训练完成!")
    best_model = os.path.join(project_dir, "data_result", "medium_train", "weights", "best.pt")
    print(f"最优模型: {best_model}")
    print("=" * 80)


if __name__ == "__main__":
    main()