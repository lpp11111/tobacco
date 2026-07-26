# Tobacco Project - 烟草检测系统

## 项目概述

本项目基于YOLOv8模型实现烟草检测功能，包含三个尺度的预训练模型和一个微调模型。

## 目录结构

```
project/
├── models/                          # 模型文件
│   ├── best_large.pt               # 大尺度模型 (1类: tobacco)
│   ├── best_medium.pt              # 中尺度模型 (4类: light_disease, healthy, severe_disease, moderate_disease)
│   ├── best_small.pt               # 小尺度模型 (4类)
│   └── best_medium_finetuned.pt    # 微调后模型 (在t数据集上微调)
├── predict.py                       # 预测函数封装
├── finetune_medium.py              # 微调训练脚本
├── check_annotations_*.py          # 标注检查脚本 (3个)
├── check_transfer_*.py             # 迁移能力检查脚本 (3个)
├── annotation_check_*/              # 标注检查结果
├── transfer_check/                  # 迁移检查结果
├── finetune_dataset/               # 微调数据集
├── finetune_medium/                 # 微调训练结果
└── 脚本说明.md                      # 脚本说明文档
```

---

## 一、模型文件清单

| 模型 | 文件路径 | 说明 |
|------|----------|------|
| 大尺度 | `models/best_large.pt` | 原始大尺度数据集训练，1类(tobacco)，4通道输入 |
| 中尺度 | `models/best_medium.pt` | 原始中尺度数据集训练，4类，4通道输入 |
| 小尺度 | `models/best_small.pt` | 原始小尺度数据集训练，4类，4通道输入 |
| 微调 | `models/best_medium_finetuned.pt` | 中尺度模型在t数据集上微调，4类，3通道输入 |

**模型类别定义：**
- large: `['tobacco']`
- medium/small/finetuned: `['light_disease', 'healthy', 'severe_disease', 'moderate_disease']`

---

## 二、预测函数使用方法

### 方式1：命令行调用

```bash
# 使用中尺度模型预测
python project/predict.py image.jpg

# 指定模型类型
python project/predict.py image.jpg --model_type medium

# 使用微调模型
python project/predict.py image.jpg --model_type finetuned

# 保存标注结果
python project/predict.py image.jpg --save --output output_folder

# 调整置信度阈值
python project/predict.py image.jpg --conf 0.3

# 使用自定义模型
python project/predict.py image.jpg --model_path path/to/best.pt
```

### 方式2：Python API调用

```python
from predict import predict

# 最简调用
result = predict('image.jpg')

# 使用微调模型
result = predict('image.jpg', model_type='finetuned')

# 保存标注图
result = predict('image.jpg', save_output=True, output_dir='./results')

# 自定义参数
result = predict('image.jpg', 
                 model_path='custom_model.pt',
                 conf_threshold=0.3)
```

### 方式3：批量预测

```python
from predict import TobaccoPredictor

predictor = TobaccoPredictor(model_type='medium')

# 批量预测
results = predictor.predict_batch(['img1.jpg', 'img2.png', 'img3.tif'])

for r in results:
    if r['success']:
        print(f"{r['image_path']}: {r['total_detections']} 个目标")
```

---

## 三、输入输出说明

### 输入要求

| 项目 | 说明 |
|------|------|
| 格式 | 支持 PNG, JPG, JPEG, TIF, TIFF, BMP, WebP |
| 大小 | 任意尺寸（模型会自动resize到640×640） |
| 通道 | 3通道RGB（微调模型）或4通道RGB+NDVI（原始模型） |

### 输出内容

`predict()` 函数返回一个字典：

```python
{
    'success': True/False,          # 是否成功
    'image_path': 'image.jpg',      # 输入图片路径
    'image_size': (640, 640),       # 原图尺寸 (width, height)
    'total_detections': 39,         # 检测目标总数
    'detections': [                 # 每个检测目标的详细信息
        {
            'class_id': 1,
            'class_name': 'healthy',
            'confidence': 0.95,
            'bbox': [100, 200, 300, 400],  # [x1, y1, x2, y2]
            'center': [200, 300],           # 中心点坐标
            'area': 40000                   # 面积
        },
        ...
    ],
    'class_counts': {               # 各类别数量统计
        'healthy': 37,
        'severe_disease': 1,
        'moderate_disease': 1
    },
    'avg_confidence': 0.7378,       # 平均置信度
    'annotated_image': ndarray,     # 标注后的图片 (numpy数组)
    'output_path': 'results/image_detected.jpg'  # 保存的路径(可选)
}
```

---

## 四、微调训练说明

### 背景

原始中尺度模型在t数据集上迁移能力有限（F1=0.15），因此在t数据集上进行微调训练。

### 微调方法

```bash
python project/finetune_medium.py
```

### 训练结果

| 指标 | 值 |
|------|-----|
| mAP50 | 0.339 |
| mAP50-95 | 0.175 |
| Precision | 0.877 |
| Recall | 0.227 |
| 训练图片数 | 89 |
| 验证图片数 | 22 |

### 类别映射

t数据集(3类) → 微调数据集(4类)：
- grow_tobacco(0) → healthy(1)
- disease_tobacco(1) → light_disease(0)
- others(2) → 跳过

---

## 五、脚本说明

### 标注检查脚本

| 脚本 | 说明 |
|------|------|
| `check_annotations_large.py` | 检查large数据集标注 |
| `check_annotations_medium.py` | 检查medium数据集标注 |
| `check_annotations_small.py` | 检查small数据集标注 |

使用：
```bash
python project/check_annotations_medium.py --split valid --max_images 10
```

### 迁移能力检查脚本

| 脚本 | 说明 |
|------|------|
| `check_transfer_large.py` | 评估large模型在t数据集上的迁移精度 |
| `check_transfer_medium.py` | 评估medium模型在t数据集上的迁移精度 |
| `check_transfer_small.py` | 评估small模型在t数据集上的迁移精度 |

使用：
```bash
python project/check_transfer_medium.py --split valid
```

---

## 六、依赖库

```
ultralytics >= 8.0
opencv-python >= 4.0
numpy
Pillow (PIL)
```

安装：
```bash
pip install ultralytics opencv-python numpy Pillow
```
