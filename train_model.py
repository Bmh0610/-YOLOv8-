#!/usr/bin/env python3
"""
光栅衍射三色谱线检测 — YOLOv8 改进模型训练脚本
======================================================
架构: CoordAtt + P2 Head (ECA 已移除，消融实验证实无正向贡献)

用法:
    python train_model.py                  # 默认配置训练
    python train_model.py --resume         # 从上次中断恢复
    python train_model.py --epochs 200     # 自定义 epoch 数
    python train_model.py --batch 8        # 自定义 batch size
    python train_model.py --device cpu     # 指定设备
"""

import sys
import os
import argparse
from pathlib import Path

# ── 路径设置 ────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ultralytics-main"))
from ultralytics import YOLO

# ── 训练常量 ────────────────────────────────────────────
MODEL_YAML = "yolov8_spectrometer.yaml.yaml"
DATA_YAML = "grating_data.yaml"
PRETRAINED = "yolov8n.pt"
EXPERIMENT_NAME = "grating_ca_p2_v3"


def build_training_args(args: argparse.Namespace) -> dict:
    """组装训练参数。"""

    # ── 基础参数 ──
    params = {
        "data": DATA_YAML,
        "epochs": args.epochs,
        "imgsz": 640,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "name": EXPERIMENT_NAME,
        "exist_ok": args.exist_ok,
        "resume": args.resume,
        "pretrained": True if not args.no_pretrain else False,
        "verbose": True,
        "plots": True,
        "save": True,
        "save_period": 10,  # 每 10 个 epoch 保存一次 checkpoint
        "val": True,
        "amp": not args.no_amp,  # 混合精度训练（显存不足时关闭）
        "seed": 42,
        "deterministic": True,
        "single_cls": False,
    }

    # ── 优化器 & 学习率 ──
    params.update({
        "optimizer": "AdamW",       # AdamW 在小数据集上通常比 SGD 更稳定
        "lr0": 0.001,               # 初始学习率 (AdamW 推荐 1e-3)
        "lrf": 0.01,                # 最终 lr = lr0 * lrf
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3,         # 前 3 个 epoch 线性预热
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "cos_lr": True,             # 余弦退火调度
    })

    # ── 正则化 ──
    params.update({
        "dropout": 0.1,             # 减轻小数据集上的过拟合
        "cls": args.cls_weight,     # 分类损失权重
        "box": 7.5,                 # 边界框损失权重
        "dfl": 1.5,                 # DFL 损失权重
    })

    # ── 数据增强：保留谱线物理特征 ──
    params.update({
        "hsv_h": 0.0,               # 禁止色调变换（保持颜色纯正）
        "hsv_s": 0.0,               # 禁止饱和度变换
        "hsv_v": 0.4,               # 允许亮度变化（模拟曝光差异）
        "degrees": 0.0,             # 禁止旋转（谱线必须是垂直的）
        "translate": 0.1,           # 小幅平移（谱线左右偏移）
        "scale": 0.3,               # 缩放（模拟不同拍摄距离）
        "shear": 0.0,               # 禁止错切
        "perspective": 0.0,         # 禁止透视变换
        "flipud": 0.0,              # 禁止上下翻转
        "fliplr": 0.5,              # 左右翻转（谱线对称，不影响）
        "mosaic": 0.0,              # 禁用 Mosaic（会把细谱线切碎）
        "mixup": 0.0,               # 禁用 MixUp（颜色会混合失真）
        "copy_paste": 0.0,          # 禁用复制粘贴
        "erasing": 0.0,             # 禁用随机擦除（谱线可能被擦掉）
    })

    # ── 早停 ──
    params.update({
        "patience": 50,             # 50 个 epoch 无提升则提前停止
    })

    return params


def print_config(args: argparse.Namespace, params: dict) -> None:
    """打印训练配置摘要。"""
    print("=" * 60)
    print("  光栅衍射三色谱线 YOLOv8 改进模型训练")
    print("=" * 60)
    print(f"  模型配置 : {MODEL_YAML}")
    print(f"  预训练权重: {PRETRAINED}")
    print(f"  数据集   : {DATA_YAML}")
    print(f"  实验名称 : {EXPERIMENT_NAME}")
    print(f"  Epochs  : {params['epochs']}")
    print(f"  Batch   : {params['batch']}")
    print(f"  ImgSz   : {params['imgsz']}")
    print(f"  设备     : {params['device']}")
    print(f"  优化器   : {params['optimizer']}")
    print(f"  初始 LR  : {params['lr0']}")
    print(f"  分类权重 : {params['cls']}")
    print(f"  Dropout  : {params['dropout']}")
    print(f"  AMP      : {'✅' if params['amp'] else '❌'}")
    print(f"  Mosaic   : {'✅' if params['mosaic'] > 0 else '❌ (禁用)'}")
    print(f"  早停耐心 : {params['patience']} epochs")
    print("=" * 60)


def main():
    # ── 命令行参数 ──
    parser = argparse.ArgumentParser(
        description="光栅衍射 YOLOv8 改进模型训练"
    )
    parser.add_argument("--epochs", type=int, default=300,
                        help="训练轮数 (默认 300)")
    parser.add_argument("--batch", type=int, default=16,
                        help="batch size (默认 16)")
    parser.add_argument("--device", type=str, default=None,
                        help="设备: 0, 1, cpu (默认自动)")
    parser.add_argument("--workers", type=int, default=8,
                        help="数据加载线程数 (默认 8)")
    parser.add_argument("--cls-weight", type=float, default=1.8,
                        help="分类损失权重 (默认 1.8)")
    parser.add_argument("--resume", action="store_true",
                        help="从上次中断恢复训练")
    parser.add_argument("--exist-ok", action="store_true",
                        help="覆盖已存在的实验目录")
    parser.add_argument("--no-pretrain", action="store_true",
                        help="不使用预训练权重")
    parser.add_argument("--no-amp", action="store_true",
                        help="禁用混合精度训练")

    args = parser.parse_args()

    # ── 构建参数 ──
    params = build_training_args(args)
    print_config(args, params)

    # ── 构建模型 ──
    print("\n🔧 正在构建网络...")
    model = YOLO(MODEL_YAML)

    if not args.no_pretrain and Path(PRETRAINED).exists():
        print(f"📥 加载预训练权重: {PRETRAINED}")
        model.load(PRETRAINED)
    else:
        print("⚠️  未使用预训练权重，从头训练")

    print("✅ 网络构建成功！\n")

    # ── 训练 ──
    try:
        results = model.train(**params)
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断。权重已自动保存到 last.pt。")
        print(f"💡 恢复训练: python train_model.py --resume")
        return
    except Exception as e:
        print(f"\n❌ 训练异常: {e}")
        raise

    # ── 最终评估 ──
    print("\n" + "=" * 60)
    print("📊 训练完成 — 最终评估结果")
    print("=" * 60)

    metrics = results.results_dict
    for key, val in metrics.items():
        print(f"  {key}: {val:.4f}")

    best_path = f"runs/detect/{EXPERIMENT_NAME}/weights/best.pt"
    print(f"\n🏆 最佳模型: {best_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
