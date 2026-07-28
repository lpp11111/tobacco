from ultralytics import YOLO
import torch
from multiprocessing import freeze_support
import os
import shutil


def main():
    freeze_support()
    project_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(project_dir)

    print("=" * 80)
    print("病害识别模型训练 - 叶片病害精细检测")
    print("=" * 80)
    print("模型配置:")
    print("- 模型: YOLOv8m (检测模型)")
    print("- 输入: 3通道 (RGB) - 近景叶片病害识别")
    print("- 任务: 叶片病害检测与分类")
    print("- 类别: 4类 (健康/细菌病/真菌病/病毒病)")
    print("- 用途: 平台识图预测功能")
    print()

    model_path = os.path.join(base_dir, "yolov8m.pt")
    yaml_path = os.path.join(project_dir, "data", "extra_small", "data.yaml")
    
    if not os.path.exists(yaml_path):
        print(f"错误: 数据集配置文件不存在: {yaml_path}")
        return

    print(f"加载预训练权重: {model_path}")
    model = YOLO(model_path)

    results = model.train(
        data=yaml_path,
        epochs=100,
        patience=25,
        imgsz=640,
        batch=4,
        workers=0,
        project=os.path.join(project_dir, "data_result"),
        name="extra_small_train",
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
    print("病害识别训练完成!")
    best_model = os.path.join(project_dir, "data_result", "extra_small_train", "weights", "best.pt")
    print(f"最优模型: {best_model}")
    
    dst_path = os.path.join(base_dir, "models", "best_extra_small.pt")
    if os.path.exists(best_model):
        shutil.copy2(best_model, dst_path)
        print(f"模型已复制到: {dst_path}")
    
    print("\n类别映射:")
    print("  0: healthy (健康)")
    print("  1: bacterial_disease (细菌病害)")
    print("  2: fungal_disease (真菌病害)")
    print("  3: viral_disease (病毒病害)")
    
    print("\n使用方法:")
    print("  from predict import predict")
    print("  result = predict('leaf.jpg', model_type='extra_small')")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
