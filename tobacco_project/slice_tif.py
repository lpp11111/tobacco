import rasterio
import numpy as np
import os
import cv2
from rasterio.windows import Window


def slice_tif(tif_path, output_dir="chunks", tile_size=640, overlap=64, save_rgb=True, save_ndvi=False):
    os.makedirs(output_dir, exist_ok=True)
    if save_rgb:
        os.makedirs(os.path.join(output_dir, "rgb"), exist_ok=True)
    if save_ndvi:
        os.makedirs(os.path.join(output_dir, "ndvi"), exist_ok=True)
    
    with rasterio.open(tif_path) as ds:
        width = ds.width
        height = ds.height
        
        tiles_x = int(np.ceil((width - overlap) / (tile_size - overlap)))
        tiles_y = int(np.ceil((height - overlap) / (tile_size - overlap)))
        
        print(f"TIF文件: {width}x{height}")
        print(f"切片数量: {tiles_x} x {tiles_y} = {tiles_x * tiles_y}")
        print(f"切片大小: {tile_size}x{tile_size}, 重叠: {overlap}")
        
        total_tiles = tiles_x * tiles_y
        saved_tiles = 0
        
        for i in range(tiles_x):
            for j in range(tiles_y):
                x_off = i * (tile_size - overlap)
                y_off = j * (tile_size - overlap)
                
                actual_width = min(tile_size, width - x_off)
                actual_height = min(tile_size, height - y_off)
                
                window = Window(x_off, y_off, actual_width, actual_height)
                data = ds.read(window=window)
                
                if data.shape[0] < 3:
                    continue
                
                is_valid = True
                for b in range(3):
                    if data[b].max() == 0:
                        is_valid = False
                        break
                
                if not is_valid:
                    continue
                
                rgb = data[:3].transpose(1, 2, 0)
                rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                
                if rgb.dtype != np.uint8:
                    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8) * 255
                    rgb = rgb.astype(np.uint8)
                
                if save_rgb:
                    rgb_path = os.path.join(output_dir, "rgb", f"tile_{i:03d}_{j:03d}.png")
                    cv2.imwrite(rgb_path, rgb)
                
                if save_ndvi and ds.count >= 4:
                    red = data[2].astype(np.float32)
                    nir = data[3].astype(np.float32)
                    ndvi = (nir - red) / (nir + red + 1e-8)
                    ndvi = np.clip(ndvi, -1, 1)
                    ndvi = ((ndvi + 1) / 2 * 255).astype(np.uint8)
                    ndvi_colored = cv2.applyColorMap(ndvi, cv2.COLORMAP_JET)
                    
                    ndvi_path = os.path.join(output_dir, "ndvi", f"tile_{i:03d}_{j:03d}.png")
                    cv2.imwrite(ndvi_path, ndvi_colored)
                
                saved_tiles += 1
                if saved_tiles % 50 == 0:
                    print(f"  已保存 {saved_tiles}/{total_tiles} 个切片")
        
        print(f"\n切片完成! 共保存 {saved_tiles} 个有效切片")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="TIF影像切片工具")
    parser.add_argument("--tif", default="rs_data/result.tif", help="TIF文件路径")
    parser.add_argument("--output", default="chunks", help="输出目录")
    parser.add_argument("--tile_size", type=int, default=640, help="切片大小")
    parser.add_argument("--overlap", type=int, default=64, help="重叠像素数")
    parser.add_argument("--no_rgb", action="store_true", help="不保存RGB切片")
    parser.add_argument("--ndvi", action="store_true", help="保存NDVI切片")
    
    args = parser.parse_args()
    
    slice_tif(args.tif, args.output, args.tile_size, args.overlap, 
              save_rgb=not args.no_rgb, save_ndvi=args.ndvi)


if __name__ == "__main__":
    main()
