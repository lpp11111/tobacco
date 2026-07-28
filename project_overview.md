# 烟草病害检测系统 - 项目文档

## 目录结构

```
tobacco/
├── predict.py                 ← 核心预测脚本 (统一入口)
├── models/                    ← 训练好的模型权重
│   ├── best_medium.pt         中尺度病害检测模型 (2类)
│   ├── best_small.pt          小尺度病害分割模型 (4类)
│   ├── best_large.pt          大尺度烟草分割模型 (1类)
│   ├── best_extra_small.pt    超高精度叶片识别模型
│   └── best_medium_finetuned.pt 微调后的中尺度模型
│
├── train/                     ← 模型训练相关
│   ├── train_medium.py        训练中尺度检测模型
│   ├── train_small.py         训练小尺度分割模型
│   ├── train_large.py         训练大尺度分割模型
│   ├── train_extra_small.py   训练超高精度模型
│   ├── data/                  训练数据集
│   │   ├── medium/            中尺度数据集 (healthy/disease)
│   │   │   ├── data.yaml      数据集配置
│   │   │   ├── train/         训练集 images/ + labels/
│   │   │   ├── valid/         验证集
│   │   │   └── test/          测试集
│   │   ├── large/             大尺度数据集 (tobacco 分割)
│   │   ├── small/             小尺度数据集 (4类病害分割)
│   │   └── extra_small/       超高精度数据集
│   ├── annotations/           标注检查脚本
│   │   ├── check_annotations_medium.py
│   │   ├── check_annotations_small.py
│   │   ├── check_annotations_large.py
│   │   └── check_annotations_t.py
│   └── statistics/            数据统计结果
│
├── check/                     ← 模型评估脚本
│   ├── check_transfer_t.py    在 t 数据集上评估所有模型
│   ├── check_transfer_medium.py 评估中尺度模型
│   ├── check_transfer_small.py 评估小尺度模型
│   └── check_transfer_large.py 评估大尺度模型
│
├── finetune/                  ← 模型微调
│   ├── finetune_medium.py     微调中尺度模型 (t数据集)
│   └── finetune_dataset/      微调数据集配置
│
├── t/                         ← 独立的 t 数据集
│   ├── data/                  t 数据集 (3类: 健康/病害/其他)
│   │   └── data.yaml
│   ├── TRAIN_transfer.py      迁移学习训练
│   ├── TRAIN_scratch.py       从零开始训练
│   ├── t_train_transfer/      迁移学习训练结果
│   ├── t_train_scratch/       从零训练结果
│   ├── yolov8m-seg.pt          预训练分割权重
│   └── yolo26n.pt              nano 模型权重
│
└── tobacco_project/           ← TIF 大图处理子项目
    ├── slice_tif.py           TIF 切片工具
    ├── tif_process.py         TIF 全流程处理
    ├── generate_comparison.py 结果对比分析
    └── chunks/                切片输出
        ├── rgb/               RGB 切片 (640×640)
        └── ndvi/              NDVI 切片 (640×640)
```

---

## predict.py — 核心预测脚本

### 功能说明

统一的预测入口，支持任意尺寸、任意格式的图片输入。自动对大图进行切片处理，小图直接推理。

### 输入要求

| 项目 | 说明 |
|------|------|
| **格式** | JPG / PNG / BMP / TIFF 等 OpenCV 支持的所有格式 |
| **通道** | 自动转换为 3 通道 RGB |
| **大小** | 任意尺寸 |
| **大图处理** | 长边 > 1024px：自动切片 (640×640，重叠 64px) |
| **小图处理** | 长边 ≤ 1024px：直接送入模型 (reshape 到 640×640) |
| **预处理** | 无需固定尺寸，无需 PNG 格式，无需预处理 |

### 模型选择

| model_type | 用途 | 模型架构 | 类别数 |
|------------|------|----------|--------|
| `'medium'` | 中尺度病害检测 | YOLOv8m | 2 (健康/病害) |
| `'small'` | 小尺度病害分割 | YOLOv8m-seg | 4 (4级病害) |
| `'large'` | 大尺度烟草分割 | YOLOv8m-seg | 1 (烟草) |
| `'extra_small'` | 超高精度叶片识别 | YOLOv8n | 2 (健康/病害) |

### 类别映射

- **medium**: `0=healthy`(健康), `1=disease`(病害)
- **small**: `0=light_disease`(轻度), `1=healthy`(健康), `2=severe_disease`(重度), `3=moderate_disease`(中度)
- **large**: `0=tobacco`(烟草)
- **extra_small**: `0=healthy`, `1=disease`

### 使用方法

**方式1：命令行**
```bash
python predict.py --source image.jpg --model_type medium --output results/
```

**方式2：代码调用**
```python
from predict import TobaccoPredictor

predictor = TobaccoPredictor(model_type='medium')
results = predictor.predict('image.jpg')
```

### predict() 返回值

```python
{
    'success': True/False,              # 是否成功
    'image_path': 'image.jpg',           # 输入图片路径
    'image_size': (1920, 1080),          # 原图尺寸 (宽, 高)
    'slicing_used': False,               # 是否启用了切片检测
    'total_detections': 15,              # 检测目标总数
    'class_counts': {                    # 各类别数量统计
        'healthy': 3,
        'disease': 12
    },
    'avg_confidence': 0.92,              # 平均置信度
    'detections': [                      # 检测框列表
        {
            'class_id': 1,
            'class_name': 'disease',
            'confidence': 0.95,
            'bbox': [100, 200, 300, 400],  # [x1, y1, x2, y2] 像素坐标
            'center': [200, 300],
            'area': 40000
        },
        ...
    ],
    'annotated_image': np.ndarray,        # 标注后的图像 (np.ndarray, BGR格式)
    'output_path': 'image_detected.jpg'   # 保存路径 (如果 save_output=True)
}
```

---

## 训练脚本 (train/)

### 1. train_medium.py — 中尺度病害检测训练

- **模型**：YOLOv8m (检测模型)
- **数据**：`train/data/medium/` (2类: healthy, disease)
- **输入尺寸**：640×640
- **输出**：`train/data_result/medium_train/weights/best.pt`
- **运行**：`python train/train_medium.py`

### 2. train_small.py — 小尺度病害分割训练

- **模型**：YOLOv8m-seg (分割模型)
- **数据**：`train/data/small/` (4类病害分级)
- **输入尺寸**：640×640
- **输出**：`train/data_result/small_train/weights/best.pt`
- **运行**：`python train/train_small.py`

### 3. train_large.py — 大尺度烟草分割训练

- **模型**：YOLOv8m-seg (分割模型)
- **数据**：`train/data/large/` (1类: tobacco)
- **输入尺寸**：640×640
- **输出**：`train/data_result/large_train/weights/best.pt`
- **运行**：`python train/train_large.py`

### 4. train_extra_small.py — 超高精度叶片识别

- **模型**：YOLOv8n (检测模型, nano)
- **数据**：`train/data/extra_small/` (2类)
- **输入尺寸**：640×640
- **输出**：`train/data_result/extra_small_train/weights/best.pt`
- **运行**：`python train/train_extra_small.py`

### 训练通用参数

| 参数 | 值 | 说明 |
|------|----|------|
| `epochs` | 100 | 最大训练轮数 |
| `patience` | 30 | 早停耐心值 |
| `imgsz` | 640 | 输入图像尺寸 |
| `batch` | 4 | 批大小 (GPU 6GB 适配) |
| `lr0` | 0.001 | 初始学习率 |
| `cos_lr` | True | 余弦退火 |
| `device` | 0 / 'cpu' | GPU 或 CPU 自动降级 |

---

## 评估脚本 (check/)

### 1. check_transfer_t.py — 全模型对比评估

- **功能**：在 t 数据集上同时评估所有 5 个模型
- **输出**：mAP50, mAP50-95, Precision, Recall, Mask mAP
- **运行**：`python check/check_transfer_t.py`

### 2. check_transfer_medium.py — 中尺度模型评估

- 评估 medium 模型在 medium 数据集上的表现
- **运行**：`python check/check_transfer_medium.py`

### 3. check_transfer_small.py — 小尺度模型评估

- 评估 small 分割模型，将 4 类映射为 2 类 (healthy/disease)
- **类别映射**：`0→1, 1→0, 2→1, 3→1`
- **运行**：`python check/check_transfer_small.py`

### 4. check_transfer_large.py — 大尺度模型评估

- 评估 large 分割模型，将类别映射为 t 数据集格式
- **运行**：`python check/check_transfer_large.py`

### 评估指标说明

| 指标 | 说明 |
|------|------|
| **mAP50** | IoU=0.5 时的平均精度均值 |
| **mAP50-95** | IoU=0.5~0.95 (步长0.05) 的平均精度均值 |
| **Precision** | 精确率：预测为正的样本中，真正为正的比例 |
| **Recall** | 召回率：真正为正的样本中，被正确预测的比例 |
| **Mask mAP** | 分割 mask 的平均精度 (仅分割模型) |

---

## 微调脚本 (finetune/)

### finetune_medium.py — 中尺度模型微调

- **功能**：将 t 数据集的分割标签转换为检测框，微调 medium 模型
- **输入**：`t/data/` (3类分割标签) + `models/best_medium.pt`
- **转换规则**：将 polygon 分割标签转为 xyxy 检测框
- **输出**：`models/best_medium_finetuned.pt`
- **运行**：`python finetune/finetune_medium.py`

---

## TIF 大图处理 (tobacco_project/)

### 1. slice_tif.py — 切片工具

- **功能**：将大尺寸 TIF 遥感图像切为 640×640 小块
- **输入**：`.tif` 文件 (任意大小，含坐标参考)
- **输出**：`chunks/rgb/` + `chunks/ndvi/` 目录下的 PNG 切片
- **参数**：`tile_size=640`, `overlap=64` (默认)
- **运行**：`python tobacco_project/slice_tif.py`

### 2. tif_process.py — 全流程处理

- **功能**：TIF 大图 → 切片 → 模型推理 → 结果汇总 → GeoJSON 输出
- **核心类**：`TIFProcessor`
- **输入**：`tif_path`, `model_path`
- **输出**：检测结果 CSV + GeoJSON (带地理坐标)
- **运行**：`python tobacco_project/tif_process.py`

### 3. generate_comparison.py — 结果对比

- 对比纯 RGB 检测 vs NDVI 辅助检测的结果差异
- **运行**：`python tobacco_project/generate_comparison.py`

---

## t 数据集训练 (t/)

### 1. TRAIN_transfer.py — 迁移学习训练

- 基于预训练的 yolov8m-seg 权重，在 t 数据集上微调
- **数据**：3 类 (`0=healthy`, `1=disease`, `2=other`)
- **输出**：`t/t_train_transfer/weights/best.pt`
- **运行**：`python t/TRAIN_transfer.py`

### 2. TRAIN_scratch.py — 从零训练

- 使用 yolov8m-seg 基础结构，在 t 数据集上从头训练
- **数据**：3 类 (`0=healthy`, `1=disease`, `2=other`)
- **输出**：`t/t_train_scratch/weights/best.pt`
- **运行**：`python t/TRAIN_scratch.py`

---

## 数据集格式说明

### 标注格式 (YOLO格式)

- **检测模型**：每行 1 行 → `class_id x_center y_center width height` (归一化)
- **分割模型**：每行 1 行 → `class_id x1 y1 x2 y2 ... xn yn` (多边形归一化)

### data.yaml 必须包含

```yaml
path: E:/tobacco/train/data/medium    # 数据集根目录 (绝对路径)
train: train/images                    # 训练集相对路径
val: valid/images                      # 验证集相对路径
test: test/images                      # 测试集相对路径
nc: 2                                  # 类别数
names: ['healthy', 'disease']          # 类别名称列表
```

---

## 依赖库

| 库 | 版本要求 | 用途 |
|----|----------|------|
| `ultralytics` | ≥8.0 | YOLOv8 框架 |
| `opencv-python` | ≥4.0 | 图像处理 |
| `numpy` | ≥1.24 | 数值计算 |
| `torch` | ≥2.0 | PyTorch 深度学习 |
| `rasterio` | ≥1.3 | TIF 图像处理 |
| `shapely` | ≥2.0 | 地理几何运算 |
| `geopandas` | ≥0.13 | 地理数据处理 |
| `matplotlib` | ≥3.7 | 结果可视化 |
| `pandas` | ≥2.0 | 数据分析 |

### 安装命令

```bash
pip install ultralytics opencv-python numpy torch rasterio shapely geopandas matplotlib pandas
```

---

## 快速开始

```bash
# 1. 直接预测单张图片
python predict.py --source test.jpg --model_type medium

# 2. 批量预测文件夹
python predict.py --source ./images/ --model_type small --output ./results/

# 3. 切片+TIF 全流程
python tobacco_project/tif_process.py --tif input.tif --model models/best_large.pt

# 4. 训练中尺度模型
python train/train_medium.py

# 5. 评估所有模型
python check/check_transfer_t.py
```