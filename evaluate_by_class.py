import sys
import os
import json
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ultralytics_main'))

from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

def evaluate_by_class(model_path, data_yaml="grating_data.yaml", imgsz=640):
    """
    按类别评估模型，特别关注颜色类别的区分能力
    """
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)
    
    results = model.val(
        data=data_yaml,
        imgsz=imgsz,
        verbose=True,
        save_json=True
    )
    
    # 类别名称
    class_names = list(results.names.values())
    
    # 按类别提取指标
    class_metrics = {}
    
    # 从results中按类别提取AP50
    if hasattr(results, 'ap50') and results.ap50 is not None:
        ap50_per_class = results.ap50
        for i, name in enumerate(class_names):
            if i < len(ap50_per_class):
                class_metrics[name] = {
                    'AP50': float(ap50_per_class[i]),
                    'AP50-95': float(results.ap[i]) if hasattr(results, 'ap') and results.ap is not None else 0.0
                }
    
    # 打印结果
    print("\n" + "="*60)
    print("按类别评估结果")
    print("="*60)
    
    for name, metrics in class_metrics.items():
        print(f"{name:>10}: AP50={metrics['AP50']:.4f}, AP50-95={metrics['AP50-95']:.4f}")
    
    # 计算颜色区分度指标
    if len(class_metrics) >= 3:
        ap50_values = [m['AP50'] for m in class_metrics.values()]
        ap50_mean = np.mean(ap50_values)
        ap50_std = np.std(ap50_values)
        ap50_range = max(ap50_values) - min(ap50_values)
        
        print(f"\n颜色区分度指标:")
        print(f"  AP50均值: {ap50_mean:.4f}")
        print(f"  AP50标准差: {ap50_std:.4f} (越小表示各类别性能越均衡)")
        print(f"  AP50极差: {ap50_range:.4f} (越小表示各类别性能差异越小)")
        print(f"  变异系数: {ap50_std/ap50_mean:.4f}")
    
    return {
        'class_metrics': class_metrics,
        'overall_map50': results.results_dict.get('metrics/mAP50(B)', 0.0),
        'overall_map50_95': results.results_dict.get('metrics/mAP50-95(B)', 0.0)
    }

def compare_models_for_color():
    """
    对比完整模型和移除ECA的模型，分析ECA对颜色区分的影响
    """
    models = {
        '完整模型(ECA+CoordAtt+P2)': 'runs/detect/ablation_full_model/weights/best.pt',
        '移除ECA(CoordAtt+P2)': 'runs/detect/ablation_no_eca/weights/best.pt',
        '移除CoordAtt(ECA+P2)': 'runs/detect/ablation_no_coordatt/weights/best.pt',
        '基础模型': 'runs/detect/ablation_baseline/weights/best.pt'
    }
    
    all_results = {}
    
    for name, path in models.items():
        if os.path.exists(path):
            print(f"\n{'='*70}")
            print(f"评估: {name}")
            print(f"{'='*70}")
            try:
                results = evaluate_by_class(path)
                all_results[name] = results
            except Exception as e:
                print(f"评估失败: {e}")
        else:
            print(f"模型不存在: {path}")
    
    # 对比分析
    print("\n" + "="*70)
    print("ECA对颜色区分的影响分析")
    print("="*70)
    
    if '完整模型(ECA+CoordAtt+P2)' in all_results and '移除ECA(CoordAtt+P2)' in all_results:
        full = all_results['完整模型(ECA+CoordAtt+P2)']['class_metrics']
        no_eca = all_results['移除ECA(CoordAtt+P2)']['class_metrics']
        
        print("\n【ECA对各类别AP50的影响】")
        print(f"{'类别':>10} {'完整模型':>12} {'移除ECA':>12} {'变化':>10} {'趋势':>6}")
        print("-"*60)
        
        for cls_name in full.keys():
            if cls_name in no_eca:
                full_ap50 = full[cls_name]['AP50']
                no_eca_ap50 = no_eca[cls_name]['AP50']
                change = no_eca_ap50 - full_ap50
                trend = "↑" if change > 0.01 else ("↓" if change < -0.01 else "→")
                print(f"{cls_name:>10} {full_ap50:>12.4f} {no_eca_ap50:>12.4f} {change:>+10.4f} {trend:>6}")
        
        # 分析颜色区分均衡性
        full_values = [m['AP50'] for m in full.values()]
        no_eca_values = [m['AP50'] for m in no_eca.values()]
        
        full_std = np.std(full_values)
        no_eca_std = np.std(no_eca_values)
        
        print(f"\n【颜色区分均衡性】")
        print(f"完整模型 AP50标准差: {full_std:.4f}")
        print(f"移除ECA  AP50标准差: {no_eca_std:.4f}")
        
        if no_eca_std < full_std:
            print(f"移除ECA后各类别性能更均衡 (标准差降低 {full_std - no_eca_std:.4f})")
        else:
            print(f"完整模型各类别性能更均衡 (标准差低 {no_eca_std - full_std:.4f})")
    
    # 可视化对比
    plot_class_comparison(all_results)
    
    return all_results

def plot_class_comparison(all_results):
    """绘制各类别AP50对比图"""
    try:
        import matplotlib
        matplotlib.use('Agg')  # 无GUI环境
        
        models_to_plot = ['完整模型(ECA+CoordAtt+P2)', '移除ECA(CoordAtt+P2)', 
                         '移除CoordAtt(ECA+P2)', '基础模型']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(3)  # 3个类别
        width = 0.2
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        for i, model_name in enumerate(models_to_plot):
            if model_name in all_results:
                metrics = all_results[model_name]['class_metrics']
                values = [metrics[name]['AP50'] for name in ['purple', 'green', 'yellow']]
                ax.bar(x + i*width, values, width, label=model_name, color=colors[i])
        
        ax.set_xlabel('颜色类别')
        ax.set_ylabel('AP50')
        ax.set_title('各模型在不同颜色类别上的AP50对比')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(['Purple', 'Green', 'Yellow'])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('class_ap50_comparison.png', dpi=150)
        print(f"\n对比图已保存: class_ap50_comparison.png")
    except Exception as e:
        print(f"绘图失败: {e}")

def main():
    if len(sys.argv) > 1:
        # 评估单个模型
        model_path = sys.argv[1]
        evaluate_by_class(model_path)
    else:
        # 对比所有模型
        compare_models_for_color()

if __name__ == "__main__":
    main()