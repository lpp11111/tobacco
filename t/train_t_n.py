from ultralytics import YOLO
import warnings
import os
import shutil

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
# 加载一个模型
        model = YOLO(r"e:\tobacco\yolov8n.pt")  # 加载n规模的小模型
# 训练模型
        results = model.train(
            task='detect',  # 检测任务
            mode='train',  # 训练模式
            data=r'e:\tobacco\t\data\data.yaml',  # 数据集配置文件
            epochs=300,  # 训练轮数
            imgsz=640,  # 输入图像尺寸
            device='0',  # 使用GPU 0（如果是CPU则设为'cpu'）
            project=r'e:\tobacco\t',  # 输出项目目录
            name='t_train_nano',  # 实验名称
            exist_ok=True,  # 允许覆盖已存在目录
        )

        t_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(t_dir)
        best_model = os.path.join(t_dir, 't_train_nano', 'weights', 'best.pt')
        dst_path = os.path.join(base_dir, 'models', 'best_t_nano.pt')
        if os.path.exists(best_model):
            shutil.copy2(best_model, dst_path)
            print(f"模型已复制到: {dst_path}")