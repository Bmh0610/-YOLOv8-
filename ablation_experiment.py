import sys
import os
import yaml
from ultralytics import YOLO
from datetime import datetime

# Add local ultralytics-main directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ultralytics-main'))

# 消融实验配置
ABLATION_CONFIGS = {
    "full_model": {
        "name": "完整模型",
        "description": "所有组件都启用",
        "eca": True,
        "coordatt": True,
        "p2_head": True
    },
    "no_eca": {
        "name": "移除ECA",
        "description": "移除ECA注意力机制",
        "eca": False,
        "coordatt": True,
        "p2_head": True
    },
    "no_coordatt": {
        "name": "移除CoordAtt",
        "description": "移除CoordAtt坐标注意力",
        "eca": True,
        "coordatt": False,
        "p2_head": True
    },
    "no_p2": {
        "name": "移除P2头",
        "description": "移除P2小目标检测头",
        "eca": True,
        "coordatt": True,
        "p2_head": False
    },
    "no_eca_coordatt": {
        "name": "移除ECA+CoordAtt",
        "description": "同时移除两种注意力机制",
        "eca": False,
        "coordatt": False,
        "p2_head": True
    },
    "baseline": {
        "name": "基础模型",
        "description": "移除所有自定义组件，使用标准YOLOv8n",
        "eca": False,
        "coordatt": False,
        "p2_head": False
    }
}

def generate_model_config(experiment_name, config):
    """根据消融配置生成模型YAML文件"""
    base_config = {
        "nc": 3,
        "scales": {"n": [0.33, 0.25, 1024]}
    }
    
    # Backbone
    backbone = [
        [-1, 1, "Conv", [64, 3, 2]],    # 0-P1/2
        [-1, 1, "Conv", [128, 3, 2]],   # 1-P2/4
    ]
    
    if config["eca"]:
        backbone.append([-1, 1, "ECA", [3]])  # 🌟 ECA注意力
    else:
        backbone.append([-1, 1, "Conv", [128, 1, 1]])  # 用1x1卷积替代
    
    backbone.extend([
        [-1, 3, "C2f", [128, True]],    # 3
        [-1, 1, "Conv", [256, 3, 2]],   # 4-P3/8
        [-1, 6, "C2f", [256, True]],    # 5
    ])
    
    if config["coordatt"]:
        backbone.append([-1, 1, "CoordAtt", [256, 256]])  # 🌟 CoordAtt
    else:
        backbone.append([-1, 1, "Conv", [256, 1, 1]])  # 用1x1卷积替代
    
    backbone.extend([
        [-1, 1, "Conv", [512, 3, 2]],   # 7-P4/16
        [-1, 6, "C2f", [512, True]],    # 8
        [-1, 1, "Conv", [1024, 3, 2]],  # 9-P5/32
        [-1, 3, "C2f", [1024, True]],   # 10
        [-1, 1, "SPPF", [1024, 5]],     # 11
    ])
    
    base_config["backbone"] = backbone
    
    # Head
    if config["p2_head"]:
        # 带P2头的结构
        head = [
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
            [[-1, 8], 1, "Concat", [1]],   # cat P4
            [-1, 3, "C2f", [512]],         # 14
            
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
            [[-1, 5], 1, "Concat", [1]],   # cat P3
            [-1, 3, "C2f", [256]],         # 17
            
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
            [[-1, 1], 1, "Concat", [1]],   # cat P2
            [-1, 3, "C2f", [128]],         # 20 (P2层)
            
            [-1, 1, "Conv", [128, 3, 2]],
            [[-1, 17], 1, "Concat", [1]],  # cat P3
            [-1, 3, "C2f", [256]],         # 23
            
            [-1, 1, "Conv", [256, 3, 2]],
            [[-1, 14], 1, "Concat", [1]],  # cat P4
            [-1, 3, "C2f", [512]],         # 26
            
            [-1, 1, "Conv", [512, 3, 2]],
            [[-1, 11], 1, "Concat", [1]],  # cat P5
            [-1, 3, "C2f", [1024]],        # 29
            
            [[20, 23, 26, 29], 1, "Detect", ["nc"]],  # 4个检测头
        ]
    else:
        # 标准YOLOv8结构（无P2头）
        head = [
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
            [[-1, 8], 1, "Concat", [1]],   # cat P4
            [-1, 3, "C2f", [512]],         # 14
            
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
            [[-1, 5], 1, "Concat", [1]],   # cat P3
            [-1, 3, "C2f", [256]],         # 17
            
            [-1, 1, "Conv", [256, 3, 2]],
            [[-1, 14], 1, "Concat", [1]],  # cat P4
            [-1, 3, "C2f", [512]],         # 20
            
            [-1, 1, "Conv", [512, 3, 2]],
            [[-1, 11], 1, "Concat", [1]],  # cat P5
            [-1, 3, "C2f", [1024]],        # 23
            
            [[17, 20, 23], 1, "Detect", ["nc"]],  # 3个检测头
        ]
    
    base_config["head"] = head
    
    # 保存配置文件
    config_path = f"yolov8_ablation_{experiment_name}.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(base_config, f, default_flow_style=False, sort_keys=False)
    
    return config_path

def run_ablation_experiment(experiment_name, config):
    """运行单个消融实验"""
    print(f"\n{'='*60}")
    print(f"🚀 开始消融实验: {config['name']}")
    print(f"📋 描述: {config['description']}")
    print(f"{'='*60}")
    
    # 生成模型配置
    model_config = generate_model_config(experiment_name, config)
    
    # 构建模型
    model = YOLO(model_config)
    model.load("yolov8n.pt")  # 加载预训练权重
    
    # 训练参数（与主训练脚本保持一致）
    train_params = {
        "data": "grating_data.yaml",
        "epochs": 100,
        "imgsz": 640,
        "batch": 16,
        # ── 优化器 ──
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "weight_decay": 0.0005,
        "warmup_epochs": 3,
        "cos_lr": True,
        # ── 正则化 ──
        "dropout": 0.1,
        "cls": 1.8,
        "box": 7.5,
        "dfl": 1.5,
        # ── 数据增强：保留谱线物理特征 ──
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.3,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "erasing": 0.0,
        # ── 早停 ──
        "patience": 30,
        # ── 其他 ──
        "amp": True,
        "seed": 42,
        "name": f"ablation_{experiment_name}",
        "verbose": True,
        "plots": True,
    }
    
    # 开始训练
    results = model.train(**train_params)
    
    # 获取mAP@0.5结果
    metrics = results.results_dict
    map50 = metrics.get('metrics/mAP50(B)', 0.0)
    map50_95 = metrics.get('metrics/mAP50-95(B)', 0.0)
    
    print(f"\n📊 实验结果:")
    print(f"  mAP@0.5: {map50:.4f}")
    print(f"  mAP@0.5:0.95: {map50_95:.4f}")
    
    return {
        "experiment": experiment_name,
        "name": config["name"],
        "description": config["description"],
        "mAP50": map50,
        "mAP50_95": map50_95,
        "eca": config["eca"],
        "coordatt": config["coordatt"],
        "p2_head": config["p2_head"],
        "timestamp": datetime.now().isoformat()
    }

def main():
    print("🎯 消融实验框架 - 光栅衍射检测模型")
    print("="*60)
    
    results = []
    
    # 运行所有消融实验
    for exp_name, config in ABLATION_CONFIGS.items():
        try:
            result = run_ablation_experiment(exp_name, config)
            results.append(result)
        except Exception as e:
            print(f"❌ 实验 {exp_name} 失败: {str(e)}")
            results.append({
                "experiment": exp_name,
                "name": config["name"],
                "description": config["description"],
                "mAP50": None,
                "mAP50_95": None,
                "error": str(e)
            })
    
    # 保存结果
    results_file = "ablation_results.yaml"
    with open(results_file, 'w') as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)
    
    # 打印汇总表格
    print("\n" + "="*80)
    print("📈 消融实验结果汇总")
    print("="*80)
    print(f"{'实验名称':<15} {'mAP@0.5':<10} {'mAP@0.5:0.95':<12} {'ECA':<5} {'CoordAtt':<10} {'P2 Head':<8}")
    print("-"*80)
    
    for res in results:
        if res["mAP50"] is not None:
            print(f"{res['name']:<15} {res['mAP50']:<10.4f} {res['mAP50_95']:<12.4f} {str(res.get('eca', '-')):<5} {str(res.get('coordatt', '-')):<10} {str(res.get('p2_head', '-')):<8}")
        else:
            print(f"{res['name']:<15} {'失败':<10} {'-':<12} {'-':<5} {'-':<10} {'-':<8}")
    
    print("\n📁 结果已保存到: ablation_results.yaml")
    print("📂 训练日志在: runs/detect/")

if __name__ == "__main__":
    main()