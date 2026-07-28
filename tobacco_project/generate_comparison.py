import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_data():
    rgb_df = pd.read_csv('output/detection_results.csv')
    ndvi_df = pd.read_csv('output/detection_results_ndvi.csv')
    return rgb_df, ndvi_df


def plot_count_comparison(rgb_df, ndvi_df, output_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    
    methods = ['纯RGB检测', 'NDVI辅助检测']
    counts = [len(rgb_df), len(ndvi_df)]
    removed = len(rgb_df) - len(ndvi_df)
    
    bars = ax.bar(methods, counts, color=['#1f77b4', '#2ca02c'])
    
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom')
    
    ax.set_ylabel('检测目标数量')
    ax.set_title('检测方法目标数量对比')
    ax.set_ylim(0, max(counts) + 10)
    
    fig.text(0.98, 0.02, f'NDVI过滤移除: {removed}个', ha='right', fontsize=10, color='gray')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'count_comparison.png'), dpi=150)
    plt.close()


def plot_confidence_distribution(rgb_df, ndvi_df, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    rgb_confs = rgb_df['confidence']
    ndvi_confs = ndvi_df['confidence']
    
    bins = np.linspace(0.2, 0.8, 20)
    
    ax1.hist(rgb_confs, bins=bins, color='#1f77b4', alpha=0.7, edgecolor='black')
    ax1.set_title('纯RGB检测置信度分布')
    ax1.set_xlabel('置信度')
    ax1.set_ylabel('数量')
    ax1.set_xlim(0.2, 0.8)
    
    ax2.hist(ndvi_confs, bins=bins, color='#2ca02c', alpha=0.7, edgecolor='black')
    ax2.set_title('NDVI辅助检测置信度分布')
    ax2.set_xlabel('置信度')
    ax2.set_ylabel('数量')
    ax2.set_xlim(0.2, 0.8)
    
    fig.suptitle('检测置信度分布对比', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'), dpi=150)
    plt.close()


def plot_ndvi_distribution(ndvi_df, output_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    
    ndvi_values = ndvi_df['ndvi']
    
    bins = np.linspace(0, 1, 25)
    
    ax.hist(ndvi_values, bins=bins, color='#17becf', alpha=0.7, edgecolor='black')
    ax.axvline(x=0.3, color='red', linestyle='--', label='NDVI阈值(0.3)')
    ax.axvline(x=0.5, color='orange', linestyle='--', label='NDVI=0.5')
    
    ax.set_title('NDVI辅助检测目标NDVI值分布')
    ax.set_xlabel('NDVI值')
    ax.set_ylabel('数量')
    ax.set_xlim(0, 1)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ndvi_distribution.png'), dpi=150)
    plt.close()


def plot_spatial_distribution(rgb_df, ndvi_df, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    rgb_x = (rgb_df['pixel_x1'] + rgb_df['pixel_x2']) / 2
    rgb_y = (rgb_df['pixel_y1'] + rgb_df['pixel_y2']) / 2
    
    ndvi_x = (ndvi_df['pixel_x1'] + ndvi_df['pixel_x2']) / 2
    ndvi_y = (ndvi_df['pixel_y1'] + ndvi_df['pixel_y2']) / 2
    
    im = ax1.scatter(rgb_x, rgb_y, c=rgb_df['confidence'], cmap='viridis', s=50, alpha=0.7)
    ax1.set_title('纯RGB检测目标空间分布')
    ax1.set_xlabel('像素X坐标')
    ax1.set_ylabel('像素Y坐标')
    ax1.invert_yaxis()
    plt.colorbar(im, ax=ax1, label='置信度')
    
    im2 = ax2.scatter(ndvi_x, ndvi_y, c=ndvi_df['ndvi'], cmap='RdYlGn', s=50, alpha=0.7)
    ax2.set_title('NDVI辅助检测目标空间分布')
    ax2.set_xlabel('像素X坐标')
    ax2.set_ylabel('像素Y坐标')
    ax2.invert_yaxis()
    plt.colorbar(im2, ax=ax2, label='NDVI值')
    
    fig.suptitle('检测目标空间分布对比', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spatial_distribution.png'), dpi=150)
    plt.close()


def plot_comparison_table(rgb_df, ndvi_df, output_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('tight')
    ax.axis('off')
    
    data = [
        ['指标', '纯RGB检测', 'NDVI辅助检测', '差异'],
        ['检测目标数', len(rgb_df), len(ndvi_df), f'-{len(rgb_df)-len(ndvi_df)}'],
        ['平均置信度', f'{rgb_df["confidence"].mean():.4f}', f'{ndvi_df["confidence"].mean():.4f}', 
         f'+{(ndvi_df["confidence"].mean()-rgb_df["confidence"].mean()):.4f}'],
        ['最高置信度', f'{rgb_df["confidence"].max():.4f}', f'{ndvi_df["confidence"].max():.4f}', '-'],
        ['最低置信度', f'{rgb_df["confidence"].min():.4f}', f'{ndvi_df["confidence"].min():.4f}', '-'],
        ['NDVI范围', '-', f'[{ndvi_df["ndvi"].min():.2f}, {ndvi_df["ndvi"].max():.2f}]', '-'],
        ['NDVI均值', '-', f'{ndvi_df["ndvi"].mean():.4f}', '-']
    ]
    
    table = ax.table(cellText=data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    
    for i in range(len(data)):
        for j in range(len(data[0])):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor('#4a4a4a')
                cell.set_text_props(color='white')
            else:
                cell.set_facecolor('#f5f5f5')
    
    plt.title('检测方法对比表', fontsize=14, pad=20)
    plt.savefig(os.path.join(output_dir, 'comparison_table.png'), dpi=150, bbox_inches='tight')
    plt.close()


def main():
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    print("加载检测结果数据...")
    rgb_df, ndvi_df = load_data()
    
    print(f"纯RGB检测: {len(rgb_df)} 个目标")
    print(f"NDVI辅助检测: {len(ndvi_df)} 个目标")
    
    print("\n生成对比图...")
    
    plot_count_comparison(rgb_df, ndvi_df, output_dir)
    print("  - count_comparison.png")
    
    plot_confidence_distribution(rgb_df, ndvi_df, output_dir)
    print("  - confidence_distribution.png")
    
    plot_ndvi_distribution(ndvi_df, output_dir)
    print("  - ndvi_distribution.png")
    
    plot_spatial_distribution(rgb_df, ndvi_df, output_dir)
    print("  - spatial_distribution.png")
    
    plot_comparison_table(rgb_df, ndvi_df, output_dir)
    print("  - comparison_table.png")
    
    print("\n所有对比图已生成!")


if __name__ == "__main__":
    main()
