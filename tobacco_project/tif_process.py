import rasterio
import numpy as np
import os
import sys
import cv2
from ultralytics import YOLO
import geopandas as gpd
from shapely.geometry import Polygon
from rasterio.windows import Window


class TIFProcessor:
    def __init__(self, tif_path, model_path, output_dir="output", use_ndvi=False, ndvi_threshold=0.3):
        self.tif_path = tif_path
        self.model_path = model_path
        self.output_dir = output_dir
        self.use_ndvi = use_ndvi
        self.ndvi_threshold = ndvi_threshold
        self.ds = None
        self.model = None
        self.results = []
        
        os.makedirs(output_dir, exist_ok=True)
    
    def load_tif(self):
        self.ds = rasterio.open(self.tif_path)
        print(f"TIF文件信息:")
        print(f"  尺寸: {self.ds.width} x {self.ds.height}")
        print(f"  波段数: {self.ds.count}")
        print(f"  投影: {self.ds.crs}")
        print(f"  地理变换: {self.ds.transform}")
        print(f"  边界: {self.ds.bounds}")
        
        data_sample = self.ds.read(window=Window(10000, 8000, 100, 100))
        print(f"  数据示例 - 波段1范围: [{data_sample[0].min()}, {data_sample[0].max()}]")
        print(f"  数据示例 - 波段2范围: [{data_sample[1].min()}, {data_sample[1].max()}]")
        print(f"  数据示例 - 波段3范围: [{data_sample[2].min()}, {data_sample[2].max()}]")
        if self.ds.count >= 4:
            print(f"  数据示例 - 波段4范围: [{data_sample[3].min()}, {data_sample[3].max()}]")
            red = data_sample[2].astype(np.float32)
            nir = data_sample[3].astype(np.float32)
            ndvi = (nir - red) / (nir + red + 1e-8)
            print(f"  NDVI范围: [{ndvi.min():.4f}, {ndvi.max():.4f}]")
    
    def load_model(self):
        self.model = YOLO(self.model_path)
        print(f"模型加载完成: {self.model_path}")
    
    def is_valid_tile(self, data):
        if data.shape[0] < 3:
            return False
        
        for i in range(3):
            if data[i].max() == 0:
                return False
        
        return True
    
    def slice_and_detect(self, tile_size=640, overlap=64):
        width = self.ds.width
        height = self.ds.height
        
        tiles_x = int(np.ceil((width - overlap) / (tile_size - overlap)))
        tiles_y = int(np.ceil((height - overlap) / (tile_size - overlap)))
        
        print(f"\n开始切片检测:")
        print(f"  切片大小: {tile_size}x{tile_size}")
        print(f"  重叠: {overlap}")
        print(f"  切片数量: {tiles_x} x {tiles_y} = {tiles_x * tiles_y}")
        print(f"  NDVI辅助: {'启用' if self.use_ndvi else '禁用'} (阈值: {self.ndvi_threshold})")
        
        total_tiles = tiles_x * tiles_y
        processed_tiles = 0
        valid_tiles = 0
        
        for i in range(tiles_x):
            for j in range(tiles_y):
                x_off = i * (tile_size - overlap)
                y_off = j * (tile_size - overlap)
                
                actual_width = min(tile_size, width - x_off)
                actual_height = min(tile_size, height - y_off)
                
                window = Window(x_off, y_off, actual_width, actual_height)
                
                try:
                    data = self.ds.read(window=window)
                    
                    if not self.is_valid_tile(data):
                        processed_tiles += 1
                        continue
                    
                    valid_tiles += 1
                    
                    rgb = data[:3].transpose(1, 2, 0)
                    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    
                    if rgb.dtype != np.uint8:
                        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8) * 255
                        rgb = rgb.astype(np.uint8)
                    
                    results = self.model(rgb, imgsz=tile_size, conf=0.25)
                    
                    ndvi_tile = None
                    if self.use_ndvi and self.ds.count >= 4:
                        red = data[2].astype(np.float32)
                        nir = data[3].astype(np.float32)
                        ndvi_tile = (nir - red) / (nir + red + 1e-8)
                    
                    for result in results:
                        if result.boxes is not None:
                            for box in result.boxes:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                conf = float(box.conf[0])
                                class_id = int(box.cls[0])
                                
                                pixel_x1 = x_off + x1
                                pixel_y1 = y_off + y1
                                pixel_x2 = x_off + x2
                                pixel_y2 = y_off + y2
                                
                                geo_x1, geo_y1 = self.ds.transform * (pixel_x1, pixel_y1)
                                geo_x2, geo_y2 = self.ds.transform * (pixel_x2, pixel_y2)
                                
                                ndvi_value = None
                                if ndvi_tile is not None:
                                    cx = int((x1 + x2) / 2)
                                    cy = int((y1 + y2) / 2)
                                    if 0 <= cx < ndvi_tile.shape[1] and 0 <= cy < ndvi_tile.shape[0]:
                                        ndvi_value = float(ndvi_tile[cy, cx])
                                
                                result_item = {
                                    'tile': f'tile_{i}_{j}',
                                    'pixel_x1': pixel_x1, 'pixel_y1': pixel_y1,
                                    'pixel_x2': pixel_x2, 'pixel_y2': pixel_y2,
                                    'geo_x1': geo_x1, 'geo_y1': geo_y1,
                                    'geo_x2': geo_x2, 'geo_y2': geo_y2,
                                    'conf': conf,
                                    'class_id': class_id,
                                    'ndvi': ndvi_value
                                }
                                
                                if self.use_ndvi and ndvi_value is not None:
                                    if ndvi_value >= self.ndvi_threshold:
                                        self.results.append(result_item)
                                else:
                                    self.results.append(result_item)
                
                except Exception as e:
                    print(f"  切片 ({i},{j}) 处理失败: {e}")
                
                processed_tiles += 1
                if processed_tiles % 50 == 0:
                    print(f"  进度: {processed_tiles}/{total_tiles}, 有效切片: {valid_tiles}, 检测目标: {len(self.results)}")
        
        print(f"检测完成，共处理 {processed_tiles} 个切片，有效切片 {valid_tiles}，检测到 {len(self.results)} 个目标")
    
    def remove_overlaps(self, iou_threshold=0.5):
        if len(self.results) == 0:
            return
        
        print(f"\n去重处理: {len(self.results)} 个检测框")
        
        boxes = []
        indices = []
        for idx, r in enumerate(self.results):
            boxes.append([r['geo_x1'], r['geo_y1'], r['geo_x2'], r['geo_y2'], r['conf']])
            indices.append(idx)
        
        boxes = np.array(boxes)
        indices = np.array(indices)
        
        sorted_indices = np.argsort(boxes[:, 4])[::-1]
        boxes = boxes[sorted_indices]
        indices = indices[sorted_indices]
        
        keep_indices = []
        while len(boxes) > 0:
            keep_indices.append(indices[0])
            current = boxes[0]
            boxes = boxes[1:]
            indices = indices[1:]
            
            ious = self._calculate_iou(current[:4], boxes[:, :4])
            mask = ious < iou_threshold
            boxes = boxes[mask]
            indices = indices[mask]
        
        original_results = self.results.copy()
        self.results = [original_results[i] for i in keep_indices]
        
        print(f"去重完成，保留 {len(self.results)} 个检测框")
    
    def _calculate_iou(self, box1, boxes2):
        x1 = np.maximum(box1[0], boxes2[:, 0])
        y1 = np.maximum(box1[1], boxes2[:, 1])
        x2 = np.minimum(box1[2], boxes2[:, 2])
        y2 = np.minimum(box1[3], boxes2[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        iou = intersection / (area1 + area2 - intersection + 1e-8)
        return iou
    
    def generate_vector(self):
        if len(self.results) == 0:
            print("没有检测结果，跳过矢量生成")
            return
        
        print("\n生成矢量文件...")
        
        geometries = []
        attributes = []
        
        for i, r in enumerate(self.results):
            polygon = Polygon([
                (r['geo_x1'], r['geo_y1']),
                (r['geo_x2'], r['geo_y1']),
                (r['geo_x2'], r['geo_y2']),
                (r['geo_x1'], r['geo_y2']),
                (r['geo_x1'], r['geo_y1'])
            ])
            
            geometries.append(polygon)
            attrs = {
                'id': i + 1,
                'confidence': r['conf'],
                'class_id': r['class_id']
            }
            if 'ndvi' in r and r['ndvi'] is not None:
                attrs['ndvi'] = r['ndvi']
            attributes.append(attrs)
        
        gdf = gpd.GeoDataFrame(attributes, geometry=geometries, crs=self.ds.crs)
        
        ndvi_suffix = "_ndvi" if self.use_ndvi else ""
        output_path = os.path.join(self.output_dir, f"tobacco_plants{ndvi_suffix}.shp")
        gdf.to_file(output_path)
        
        print(f"矢量文件已保存: {output_path}")
        print(f"烟草苗总数: {len(gdf)}")
    
    def save_results_csv(self):
        import csv
        
        ndvi_suffix = "_ndvi" if self.use_ndvi else ""
        csv_path = os.path.join(self.output_dir, f"detection_results{ndvi_suffix}.csv")
        
        headers = ['id', 'pixel_x1', 'pixel_y1', 'pixel_x2', 'pixel_y2',
                   'geo_x1', 'geo_y1', 'geo_x2', 'geo_y2', 'confidence', 'class_id']
        if self.use_ndvi:
            headers.append('ndvi')
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
            for i, r in enumerate(self.results):
                row = [
                    i + 1,
                    r['pixel_x1'], r['pixel_y1'], r['pixel_x2'], r['pixel_y2'],
                    r['geo_x1'], r['geo_y1'], r['geo_x2'], r['geo_y2'],
                    r['conf'], r['class_id']
                ]
                if 'ndvi' in r and r['ndvi'] is not None:
                    row.append(r['ndvi'])
                writer.writerow(row)
        
        print(f"检测结果CSV已保存: {csv_path}")
    
    def run(self):
        self.load_tif()
        self.load_model()
        self.slice_and_detect()
        self.remove_overlaps()
        self.generate_vector()
        self.save_results_csv()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="TIF影像烟草苗检测")
    parser.add_argument("--tif", default="rs_data/result.tif", help="TIF文件路径")
    parser.add_argument("--model", default="../train/small/small_train/weights/best.pt", help="模型权重路径")
    parser.add_argument("--output", default="output", help="输出目录")
    parser.add_argument("--tile_size", type=int, default=640, help="切片大小")
    parser.add_argument("--overlap", type=int, default=64, help="重叠像素数")
    parser.add_argument("--ndvi", action="store_true", help="启用NDVI辅助检测")
    parser.add_argument("--ndvi_threshold", type=float, default=0.3, help="NDVI过滤阈值")
    
    args = parser.parse_args()
    
    processor = TIFProcessor(args.tif, args.model, args.output, args.ndvi, args.ndvi_threshold)
    processor.run()


if __name__ == "__main__":
    main()
