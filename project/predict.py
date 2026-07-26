import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import argparse


class TobaccoPredictor:
    """烟草检测预测器 - 支持任意格式、任意大小图片的检测
    
    使用方法:
        # 初始化（可选模型: large, medium, small, finetuned）
        predictor = TobaccoPredictor(model_type='medium')
        
        # 单张图片预测
        result = predictor.predict('image.jpg')
        
        # 批量预测
        results = predictor.predict_batch(['img1.jpg', 'img2.png'])
        
        # 使用自定义模型
        predictor = TobaccoPredictor(model_path='path/to/best.pt')
    """
    
    MODEL_MAP = {
        'large': 'models/best_large.pt',
        'medium': 'models/best_medium.pt',
        'small': 'models/best_small.pt',
        'finetuned': 'models/best_medium_finetuned.pt',
    }
    
    CLASS_NAMES = {
        'large': ['tobacco'],
        'medium': ['light_disease', 'healthy', 'severe_disease', 'moderate_disease'],
        'small': ['light_disease', 'healthy', 'severe_disease', 'moderate_disease'],
        'finetuned': ['light_disease', 'healthy', 'severe_disease', 'moderate_disease'],
    }
    
    SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'}
    
    def __init__(self, model_path: Optional[str] = None, model_type: str = 'medium',
                 conf_threshold: float = 0.25, imgsz: int = 640):
        """初始化预测器
        
        Args:
            model_path: 自定义模型路径（优先于model_type）
            model_type: 预设模型类型 ('large', 'medium', 'small', 'finetuned')
            conf_threshold: 置信度阈值，默认0.25
            imgsz: 输入图像大小，默认640
        """
        from ultralytics import YOLO
        
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        
        if model_path:
            self.model_path = model_path
            self.model_type = 'custom'
        else:
            if model_type not in self.MODEL_MAP:
                raise ValueError(f"未知模型类型: {model_type}, 可选: {list(self.MODEL_MAP.keys())}")
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.model_path = os.path.join(base_dir, self.MODEL_MAP[model_type])
            self.model_type = model_type
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        self.model = YOLO(self.model_path)
        
        if self.model_type in self.CLASS_NAMES:
            self.class_names = self.CLASS_NAMES[self.model_type]
        else:
            self.class_names = self.model.names
    
    def _load_image(self, img_path: str) -> np.ndarray:
        """加载图片，支持多种格式"""
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            # Try with PIL for tif/tiff support
            try:
                from PIL import Image
                pil_img = Image.open(img_path)
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except:
                raise ValueError(f"无法加载图片: {img_path}")
        return img
    
    def _draw_detections(self, img: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """在图片上绘制检测框"""
        annotated = img.copy()
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        for det in detections:
            cls_id = det['class_id']
            conf = det['confidence']
            class_name = det['class_name']
            x1, y1, x2, y2 = det['bbox']
            
            color = colors[cls_id % len(colors)]
            
            # Draw rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name} {conf:.2f}"
            font_scale = 0.6
            thickness = 2
            
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            
            cv2.rectangle(annotated, (x1, y1 - text_height - baseline - 4),
                         (x1 + text_width, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - baseline - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        return annotated
    
    def predict(self, img_path: str, save_output: bool = False,
                output_dir: Optional[str] = None,
                conf_threshold: Optional[float] = None) -> Dict[str, Any]:
        """单张图片预测
        
        Args:
            img_path: 图片路径（支持png, jpg, tif, bmp等）
            save_output: 是否保存标注后的图片
            output_dir: 输出目录（默认与图片同目录）
            conf_threshold: 覆盖默认置信度阈值
            
        Returns:
            dict: 预测结果，包含:
                - success: bool - 是否成功
                - image_path: str - 输入图片路径
                - image_size: tuple - 原图尺寸 (width, height)
                - total_detections: int - 检测目标总数
                - detections: list - 每个检测目标的详细信息
                - class_counts: dict - 各类别数量统计
                - avg_confidence: float - 平均置信度
                - annotated_image: np.ndarray - 标注后的图片
                - output_path: str - 保存的输出图片路径（如果save_output=True）
                
        Example:
            predictor = TobaccoPredictor(model_type='medium')
            result = predictor.predict('tobacco.jpg')
            
            print(f"检测到 {result['total_detections']} 个目标")
            print(f"类别统计: {result['class_counts']}")
            print(f"平均置信度: {result['avg_confidence']:.2f}")
        """
        result = {
            'success': False,
            'image_path': img_path,
            'image_size': (0, 0),
            'total_detections': 0,
            'detections': [],
            'class_counts': {},
            'avg_confidence': 0.0,
            'annotated_image': None,
            'output_path': None,
        }
        
        try:
            # Check file exists
            if not os.path.exists(img_path):
                result['error'] = f"文件不存在: {img_path}"
                return result
            
            # Check format
            ext = Path(img_path).suffix.lower()
            if ext not in self.SUPPORTED_FORMATS:
                result['error'] = f"不支持的图片格式: {ext}, 支持: {self.SUPPORTED_FORMATS}"
                return result
            
            # Load image
            img = self._load_image(img_path)
            h, w = img.shape[:2]
            result['image_size'] = (w, h)
            
            # Run prediction
            conf = conf_threshold if conf_threshold is not None else self.conf_threshold
            predictions = self.model.predict(
                img, 
                conf=conf, 
                imgsz=self.imgsz, 
                verbose=False
            )
            
            # Parse results
            detections = []
            for pred in predictions:
                if pred.boxes is not None:
                    for box in pred.boxes:
                        cls_id = int(box.cls[0])
                        conf_score = float(box.conf[0])
                        bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                        
                        if cls_id < len(self.class_names):
                            class_name = self.class_names[cls_id]
                        else:
                            class_name = f"class_{cls_id}"
                        
                        detection = {
                            'class_id': cls_id,
                            'class_name': class_name,
                            'confidence': conf_score,
                            'bbox': [int(b) for b in bbox],
                            'center': [int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)],
                            'area': int((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])),
                        }
                        detections.append(detection)
            
            # Update results
            result['detections'] = detections
            result['total_detections'] = len(detections)
            
            # Class counts
            class_counts = {}
            for det in detections:
                cls = det['class_name']
                class_counts[cls] = class_counts.get(cls, 0) + 1
            result['class_counts'] = class_counts
            
            # Average confidence
            if detections:
                result['avg_confidence'] = sum(d['confidence'] for d in detections) / len(detections)
            
            # Draw annotations
            result['annotated_image'] = self._draw_detections(img, detections)
            
            # Save output
            if save_output and result['annotated_image'] is not None:
                if output_dir is None:
                    output_dir = os.path.dirname(img_path)
                os.makedirs(output_dir, exist_ok=True)
                
                base_name = Path(img_path).stem
                output_path = os.path.join(output_dir, f"{base_name}_detected.jpg")
                cv2.imwrite(output_path, result['annotated_image'])
                result['output_path'] = output_path
            
            result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def predict_batch(self, img_paths: List[str], save_output: bool = False,
                      output_dir: Optional[str] = None,
                      conf_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """批量图片预测
        
        Args:
            img_paths: 图片路径列表
            save_output: 是否保存标注后的图片
            output_dir: 输出目录
            conf_threshold: 覆盖默认置信度阈值
            
        Returns:
            list: 每张图片的预测结果列表
        """
        results = []
        for img_path in img_paths:
            result = self.predict(img_path, save_output, output_dir, conf_threshold)
            results.append(result)
        return results


def predict(img_path: str, model_type: str = 'medium', 
            model_path: Optional[str] = None,
            conf_threshold: float = 0.25,
            save_output: bool = False,
            output_dir: Optional[str] = None) -> Dict[str, Any]:
    """便捷预测函数 - 直接调用
    
    Args:
        img_path: 图片路径
        model_type: 模型类型 ('large', 'medium', 'small', 'finetuned')
        model_path: 自定义模型路径（优先于model_type）
        conf_threshold: 置信度阈值
        save_output: 是否保存标注图
        output_dir: 输出目录
        
    Returns:
        dict: 预测结果
        
    Example:
        from predict import predict
        
        result = predict('tobacco.jpg')
        print(f"检测到 {result['total_detections']} 个目标")
        print(f"类别: {result['class_counts']}")
    """
    predictor = TobaccoPredictor(
        model_path=model_path,
        model_type=model_type,
        conf_threshold=conf_threshold
    )
    return predictor.predict(img_path, save_output, output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='烟草检测预测脚本')
    parser.add_argument('img', help='图片路径')
    parser.add_argument('--model_type', default='medium', 
                       choices=['large', 'medium', 'small', 'finetuned'],
                       help='模型类型')
    parser.add_argument('--model_path', help='自定义模型路径')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--save', action='store_true', help='保存标注图片')
    parser.add_argument('--output', help='输出目录')
    
    args = parser.parse_args()
    
    result = predict(
        img_path=args.img,
        model_type=args.model_type,
        model_path=args.model_path,
        conf_threshold=args.conf,
        save_output=args.save,
        output_dir=args.output
    )
    
    if result['success']:
        print(f"\n{'='*50}")
        print(f"检测成功!")
        print(f"{'='*50}")
        print(f"图片: {result['image_path']}")
        print(f"尺寸: {result['image_size']}")
        print(f"检测目标数: {result['total_detections']}")
        print(f"平均置信度: {result['avg_confidence']:.4f}")
        print(f"类别统计:")
        for cls_name, count in result['class_counts'].items():
            print(f"  - {cls_name}: {count}")
        if result['output_path']:
            print(f"标注图片: {result['output_path']}")
    else:
        print(f"\n检测失败: {result.get('error', '未知错误')}")
