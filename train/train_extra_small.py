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
    print("烟草目标检测模型训练 - extra_small 数据集")
    print("=" * 80)
    print("模型配置:")
    print("- 模型: YOLOv8m (检测模型)")
    print("- 输入: 3通道 (RGB) - 烟草叶片/茎秆/杂草多类目标检测")
    print("- 任务: 叶片状态检测与分类")
    print("- 类别: 10类 (batang/bayam/daun bintik kuning/daun kecil/daun matang/daun sehat/daun_berlubang/daunberlubang/daunhama/penyakit k)")
    print("- 数据: train=254 / valid=71 / test=37")
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
    print("extra_small 模型训练完成!")
    best_model = os.path.join(project_dir, "data_result", "extra_small_train", "weights", "best.pt")
    print(f"最优模型: {best_model}")
    
    dst_path = os.path.join(base_dir, "models", "best_extra_small.pt")
    if os.path.exists(best_model):
        shutil.copy2(best_model, dst_path)
        print(f"模型已复制到: {dst_path}")
    
    print("\n类别映射:")
    print("  0: batang (茎秆)")
    print("  1: bayam (杂草/苋菜)")
    print("  2: daun bintik kuning (黄斑叶)")
    print("  3: daun kecil (小叶)")
    print("  4: daun matang (成熟叶)")
    print("  5: daun sehat (健康叶)")
    print("  6: daun_berlubang (穿孔叶)")
    print("  7: daunberlubang (穿孔叶-变体)")
    print("  8: daunhama (虫害叶)")
    print("  9: penyakit k (病害k)")
    
    print("\n使用方法:")
    print("  from predict import predict")
    print("  result = predict('leaf.jpg', model_type='extra_small')")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
