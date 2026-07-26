from ultralytics import YOLO
import torch
import torch.nn as nn
from multiprocessing import freeze_support
import os


def adapt_4channel_model(model):
    """适配4通道输入的模型"""
    net = model.model.model
    first_conv = None
    for m in net.modules():
        if isinstance(m, nn.Conv2d) and m.in_channels == 3:
            first_conv = m
            break
    if first_conv is not None:
        new_conv = nn.Conv2d(
            in_channels=4,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None
        )
        new_conv.weight.data[:, :3, :, :] = first_conv.weight.data
        new_conv.weight.data[:, 3:4, :, :] = torch.mean(first_conv.weight.data, dim=1, keepdim=True)
        if first_conv.bias is not None:
            new_conv.bias.data = first_conv.bias.data
        for name, module in net.named_modules():
            if module is first_conv:
                parent_name = name.rsplit(".", 1)[0]
                layer_name = name.rsplit(".", 1)[1]
                parent_module = net
                for part in parent_name.split("."):
                    parent_module = getattr(parent_module, part)
                setattr(parent_module, layer_name, new_conv)
                break
    return model


def main():
    freeze_support()
    project_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(project_dir)

    print("=" * 80)
    print("中尺度模型训练 - 病害检测")
    print("=" * 80)
    print("\n模型配置:")
    print("- 模型: YOLOv8m (检测模型)")
    print("- 输入: 4通道 (RGB + NDVI)")
    print("- 任务: 病害检测")
    print("- 优化: 针对中尺度病害检测优化")
    print()

    model_path = os.path.join(base_dir, "yolov8m.pt")
    yaml_path = os.path.join(base_dir, "medium", "data.yaml")

    print(f"加载预训练权重: {model_path}")
    model = YOLO(model_path)

    model = adapt_4channel_model(model)
    print("4通道输入适配完成")

    # 中尺度专用训练参数
    results = model.train(
        data=yaml_path,
        epochs=100,
        patience=30,
        imgsz=640,
        batch=4,
        workers=0,
        project=os.path.join(project_dir, "medium"),
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
        lr0=0.01,
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
    best_model = os.path.join(project_dir, "medium", "medium_train", "weights", "best.pt")
    print(f"最优模型: {best_model}")
    print("=" * 80)


if __name__ == "__main__":
    main()
