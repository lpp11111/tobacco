import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import argparse


class TobaccoPredictor:
    """烟草检测预测器 - 支持任意格式、任意大小图片的检测
    
    使用方法:
        # 初始化（可选模型: large, medium, small, extra_small, finetuned, t_scratch, t_transfer, t_nano）
        predictor = TobaccoPredictor(model_type='medium')
        
        # 单张图片预测（自动处理大尺寸图片切片）
        result = predictor.predict('image.jpg')
        
        # 批量预测
        results = predictor.predict_batch(['img1.jpg', 'img2.png'])
        
        # 使用自定义模型
        predictor = TobaccoPredictor(model_path='path/to/best.pt')
    
    大尺寸图片自动切片:
        - 默认阈值: 1024px（超过则启用切片）
        - 切片大小: 640x640（与训练尺寸一致）
        - 重叠: 64px（防止边缘目标丢失）
        - 自动坐标映射 + NMS去重
    """
    
    MODEL_MAP = {
        'large': 'models/best_large.pt',
        'medium': 'models/best_medium.pt',
        'small': 'models/best_small.pt',
        'extra_small': 'models/best_extra_small.pt',
        'finetuned': 'models/best_medium_finetuned.pt',
        't_scratch': 'models/best_t_scratch.pt',
        't_transfer': 'models/best_t_transfer.pt',
        't_nano': 'models/best_t_nano.pt',
    }

    CLASS_NAMES = {
        'large': ['tobacco'],
        'medium': ['healthy', 'disease'],
        'small': ['light_disease', 'healthy', 'severe_disease', 'moderate_disease'],
        'extra_small': ['batang', 'bayam', 'daun bintik kuning', 'daun kecil', 'daun matang',
                        'daun sehat', 'daun_berlubang', 'daunberlubang', 'daunhama', 'penyakit k'],
        'finetuned': ['healthy', 'disease'],
        't_scratch': ['grow_tobacco', 'disease_tobacco', 'others'],
        't_transfer': ['grow_tobacco', 'disease_tobacco', 'others'],
        't_nano': ['grow_tobacco', 'disease_tobacco', 'others'],
    }
    
    SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'}
    
    def __init__(self, model_path: Optional[str] = None, model_type: str = 'medium',
                 conf_threshold: float = 0.25, imgsz: int = 640,
                 max_size_threshold: int = 1024):
        """初始化预测器
        
        Args:
            model_path: 自定义模型路径（优先于model_type）
            model_type: 预设模型类型 ('large', 'medium', 'small', 'extra_small', 'finetuned', 't_scratch', 't_transfer', 't_nano')
            conf_threshold: 置信度阈值，默认0.25
            imgsz: 模型输入图像大小，默认640
            max_size_threshold: 自动切片阈值（像素），图片边长超过此值则切片，默认1024
        """
        from ultralytics import YOLO
        
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.max_size_threshold = max_size_threshold
        
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
            try:
                from PIL import Image
                pil_img = Image.open(img_path)
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except:
                raise ValueError(f"无法加载图片: {img_path}")
        return img
    
    def _should_slice(self, img: np.ndarray) -> bool:
        """检查是否需要切片（图片边长超过阈值）"""
        h, w = img.shape[:2]
        return h > self.max_size_threshold or w > self.max_size_threshold
    
    def _slice_detect(self, img: np.ndarray) -> List[Dict]:
        """切片检测 - 用于大尺寸图片
        
        策略:
        1. 将图片切成640x640的重叠切片（overlap=64）
        2. 只保留中心点在切片"中心区域"（非重叠区）的检测结果
        3. 最后用NMS去除残留的重复框
        
        Args:
            img: 输入图片
            
        Returns:
            list: 检测结果列表（坐标已映射回原图）
        """
        h, w = img.shape[:2]
        tile_size = self.imgsz
        overlap = 64
        step = tile_size - overlap
        
        tiles_x = int(np.ceil((w - overlap) / step))
        tiles_y = int(np.ceil((h - overlap) / step))
        
        all_detections = []
        
        for i in range(tiles_x):
            for j in range(tiles_y):
                x_off = i * step
                y_off = j * step
                
                actual_w = min(tile_size, w - x_off)
                actual_h = min(tile_size, h - y_off)
                
                tile = img[y_off:y_off + actual_h, x_off:x_off + actual_w]
                
                if tile.size == 0 or tile.mean() == 0:
                    continue
                
                results = self.model.predict(
                    tile,
                    conf=self.conf_threshold,
                    imgsz=tile_size,
                    verbose=False
                )
                
                # 计算切片的"有效区域"（去除重叠边缘）
                # 只保留中心点在这个区域的检测
                margin = overlap // 2  # 32px
                valid_x1 = margin if i > 0 else 0
                valid_y1 = margin if j > 0 else 0
                valid_x2 = actual_w - margin if i < tiles_x - 1 else actual_w
                valid_y2 = actual_h - margin if j < tiles_y - 1 else actual_h
                
                for pred in results:
                    if pred.boxes is not None:
                        for box in pred.boxes:
                            cls_id = int(box.cls[0])
                            conf_score = float(box.conf[0])
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            
                            # 计算中心点（在切片坐标系内）
                            cx = (x1 + x2) / 2
                            cy = (y1 + y2) / 2
                            
                            # 只保留中心点在有效区域内的检测
                            if not (valid_x1 <= cx <= valid_x2 and valid_y1 <= cy <= valid_y2):
                                continue
                            
                            # 映射回原图坐标
                            abs_x1 = x_off + x1
                            abs_y1 = y_off + y1
                            abs_x2 = x_off + x2
                            abs_y2 = y_off + y2
                            
                            if cls_id < len(self.class_names):
                                class_name = self.class_names[cls_id]
                            else:
                                class_name = f"class_{cls_id}"
                            
                            detection = {
                                'class_id': cls_id,
                                'class_name': class_name,
                                'confidence': conf_score,
                                'bbox': [int(abs_x1), int(abs_y1), int(abs_x2), int(abs_y2)],
                                'center': [int((abs_x1 + abs_x2) / 2), int((abs_y1 + abs_y2) / 2)],
                                'area': int((abs_x2 - abs_x1) * (abs_y2 - abs_y1)),
                            }
                            all_detections.append(detection)
        
        # NMS去重（更严格的阈值）
        if len(all_detections) > 0:
            all_detections = self._nms(all_detections, iou_threshold=0.3)
        
        return all_detections
    
    def _nms(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """非极大值抑制（NMS）去重
        
        Args:
            detections: 检测框列表
            iou_threshold: IoU阈值
            
        Returns:
            list: 去重后的检测框
        """
        if len(detections) == 0:
            return []
        
        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['confidence'] for d in detections])
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return [detections[i] for i in keep]
    
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
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
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
        
        自动检测图片尺寸：
        - 小图片（边长<=阈值）：直接缩放预测
        - 大图片（边长>阈值）：自动切片检测 + NMS去重
        
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
                - slicing_used: bool - 是否启用了切片检测
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
            print(f"是否切片: {result['slicing_used']}")
            print(f"类别统计: {result['class_counts']}")
        """
        result = {
            'success': False,
            'image_path': img_path,
            'image_size': (0, 0),
            'slicing_used': False,
            'total_detections': 0,
            'detections': [],
            'class_counts': {},
            'avg_confidence': 0.0,
            'annotated_image': None,
            'output_path': None,
        }
        
        try:
            if not os.path.exists(img_path):
                result['error'] = f"文件不存在: {img_path}"
                return result
            
            ext = Path(img_path).suffix.lower()
            if ext not in self.SUPPORTED_FORMATS:
                result['error'] = f"不支持的图片格式: {ext}, 支持: {self.SUPPORTED_FORMATS}"
                return result
            
            img = self._load_image(img_path)
            h, w = img.shape[:2]
            result['image_size'] = (w, h)
            
            conf = conf_threshold if conf_threshold is not None else self.conf_threshold
            
            if self._should_slice(img):
                # 大尺寸图片：切片检测
                result['slicing_used'] = True
                detections = self._slice_detect(img)
            else:
                # 小尺寸图片：直接检测
                predictions = self.model.predict(
                    img,
                    conf=conf,
                    imgsz=self.imgsz,
                    verbose=False
                )
                
                detections = []
                for pred in predictions:
                    if pred.boxes is not None:
                        for box in pred.boxes:
                            cls_id = int(box.cls[0])
                            conf_score = float(box.conf[0])
                            bbox = box.xyxy[0].tolist()
                            
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
            
            result['detections'] = detections
            result['total_detections'] = len(detections)
            
            class_counts = {}
            for det in detections:
                cls = det['class_name']
                class_counts[cls] = class_counts.get(cls, 0) + 1
            result['class_counts'] = class_counts
            
            if detections:
                result['avg_confidence'] = sum(d['confidence'] for d in detections) / len(detections)
            
            result['annotated_image'] = self._draw_detections(img, detections)
            
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
            output_dir: Optional[str] = None,
            max_size_threshold: int = 1024) -> Dict[str, Any]:
    """便捷预测函数 - 直接调用
    
    Args:
        img_path: 图片路径
        model_type: 模型类型 ('large', 'medium', 'small', 'extra_small', 'finetuned', 't_scratch', 't_transfer')
        model_path: 自定义模型路径（优先于model_type）
        conf_threshold: 置信度阈值
        save_output: 是否保存标注图
        output_dir: 输出目录
        max_size_threshold: 自动切片阈值（像素），默认1024
        
    Returns:
        dict: 预测结果
        
    Example:
        from predict import predict
        
        result = predict('tobacco.jpg')
        print(f"检测到 {result['total_detections']} 个目标")
        print(f"是否切片: {result['slicing_used']}")
    """
    predictor = TobaccoPredictor(
        model_path=model_path,
        model_type=model_type,
        conf_threshold=conf_threshold,
        max_size_threshold=max_size_threshold
    )
    return predictor.predict(img_path, save_output, output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='烟草检测预测脚本')
    parser.add_argument('img', nargs='?', help='图片路径')
    parser.add_argument('--model_type', default='medium',
                       choices=['large', 'medium', 'small', 'extra_small', 'finetuned', 't_scratch', 't_transfer', 't_nano'],
                       help='模型类型')
    parser.add_argument('--model_path', help='自定义模型路径')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--save', action='store_true', help='保存标注图片')
    parser.add_argument('--output', help='输出目录')
    parser.add_argument('--max_size', type=int, default=1024,
                       help='自动切片阈值（像素），默认1024')
    parser.add_argument('--compare', action='store_true',
                       help='对比模式：用 medium(原始) 和 finetuned(微调后) 两个模型分别预测并生成两张对比图')
    parser.add_argument('--batch', help='批量模式：传文件夹路径，对文件夹内所有图片进行预测')

    args = parser.parse_args()

    # --compare 对比模式
    if args.compare:
        if not args.img:
            print("错误: --compare 模式需要指定图片路径")
            parser.print_help()
            exit(1)
        import shutil as _shutil
        out_dir = args.output or 'compare_before_after'
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # 用 medium 模型
        r1 = predict(args.img, model_type='medium', save_output=True, output_dir=out_dir,
                     max_size_threshold=args.max_size, conf_threshold=args.conf)
        if r1.get('output_path') and os.path.exists(r1['output_path']):
            dst1 = os.path.join(base_dir, out_dir, 'origin_median_for_finetune.jpg')
            _shutil.copy2(r1['output_path'], dst1)
            print(f"\n[原图] medium 模型: {dst1}")
            print(f"  目标数: {r1['total_detections']}, 置信度: {r1['avg_confidence']:.4f}")
            print(f"  类别: {r1['class_counts']}")

        # 用 finetuned 模型
        r2 = predict(args.img, model_type='finetuned', save_output=True, output_dir=out_dir,
                     max_size_threshold=args.max_size, conf_threshold=args.conf)
        if r2.get('output_path') and os.path.exists(r2['output_path']):
            dst2 = os.path.join(base_dir, out_dir, 'finetuned_median_for_finetune.jpg')
            _shutil.copy2(r2['output_path'], dst2)
            print(f"\n[微调] finetuned 模型: {dst2}")
            print(f"  目标数: {r2['total_detections']}, 置信度: {r2['avg_confidence']:.4f}")
            print(f"  类别: {r2['class_counts']}")
        print("\n对比图生成完成!")
        exit(0)

    # --batch 批量模式
    if args.batch:
        img_dir = args.batch
        exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'}
        imgs = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)
                        if Path(f).suffix.lower() in exts])
        if not imgs:
            print(f"错误: 目录 {img_dir} 中没有找到支持的图片")
            exit(1)
        print(f"批量模式: 找到 {len(imgs)} 张图片")
        out_dir = args.output or 'predict_batch_output'
        predictor = TobaccoPredictor(model_type=args.model_type, model_path=args.model_path,
                                      conf_threshold=args.conf, max_size_threshold=args.max_size)
        results = predictor.predict_batch(imgs, save_output=True, output_dir=out_dir)
        success = sum(1 for r in results if r['success'])
        print(f"\n批量预测完成: {success}/{len(results)} 张成功")
        for r in results:
            status = "✓" if r['success'] else "✗"
            print(f"  {status} {r['image_path']} -> {r['total_detections']} 目标")
        exit(0)

    # 单张模式（默认）
    if not args.img:
        parser.print_help()
        print("\n提示: 使用 --compare 生成对比图，使用 --batch <目录> 批量预测")
        exit(1)

    result = predict(
        img_path=args.img,
        model_type=args.model_type,
        model_path=args.model_path,
        conf_threshold=args.conf,
        save_output=args.save,
        output_dir=args.output,
        max_size_threshold=args.max_size
    )

    if result['success']:
        print(f"\n{'='*50}")
        print(f"检测成功!")
        print(f"{'='*50}")
        print(f"图片: {result['image_path']}")
        print(f"尺寸: {result['image_size']}")
        print(f"切片检测: {'是' if result['slicing_used'] else '否'}")
        print(f"检测目标数: {result['total_detections']}")
        print(f"平均置信度: {result['avg_confidence']:.4f}")
        print(f"类别统计:")
        for cls_name, count in result['class_counts'].items():
            print(f"  - {cls_name}: {count}")
        if result['output_path']:
            print(f"标注图片: {result['output_path']}")
    else:
        print(f"\n检测失败: {result.get('error', '未知错误')}")
