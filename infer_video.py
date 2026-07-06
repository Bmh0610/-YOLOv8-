#!/usr/bin/env python3
"""
光栅衍射三色谱线 — 视频推理与锁定播报
========================================
功能:
  - 读取视频文件，逐帧 YOLOv8 推理
  - 检测紫/绿/黄三条谱线是否与画面中心线重合
  - 重合时语音播报 + 画面冻结（模拟锁定效果）
  - 输出带标注的 MP4 视频（默认保存到 results 目录）

用法:
    python infer_video.py [video.mp4] [选项]

    # 默认处理 E:/grating_yolo/videos/2.mp4，输出到 results 目录
    python infer_video.py

    # 指定输入视频（输出默认保存到 results 目录）
    python infer_video.py E:/grating_yolo/videos/test.mp4

    # 指定输出路径
    python infer_video.py E:/grating_yolo/videos/test.mp4 -o custom/output.mp4

    # 自定义中心线位置（默认帧宽/2）
    python infer_video.py E:/grating_yolo/videos/test.mp4 --center 960

    # 禁用语音（仅视觉标注）
    python infer_video.py E:/grating_yolo/videos/test.mp4 --no-voice

    # 禁用冻结效果
    python infer_video.py E:/grating_yolo/videos/test.mp4 --no-freeze

    # 开启图像预处理（CLAHE + 去噪 + Gamma + 锐化）
    python infer_video.py E:/grating_yolo/videos/test.mp4 --preproc
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from utils.cv2_chinese import get_text_size_cn, put_text_cn

# ── 可选依赖 ──
try:
    import pyttsx3
    import threading
    import queue as tts_queue

    _voice_available = True
except ImportError:
    _voice_available = False


# ══════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════

WAVELENGTHS = {
    0: {
        "name": "Purple",
        "cn_name": "紫色谱线",
        "lambda_nm": 435.8,
        "color_bgr": (255, 0, 255),
    },
    1: {
        "name": "Green",
        "cn_name": "绿色谱线",
        "lambda_nm": 546.1,
        "color_bgr": (0, 255, 0),
    },
    2: {
        "name": "Yellow",
        "cn_name": "黄色谱线",
        "lambda_nm": 578.0,
        "color_bgr": (0, 255, 255),
    },
}

MODEL_CANDIDATES = [
    "runs/detect/grating_ca_p2_v3/weights/best.pt",
]

# ── 锁定参数 ──
LOCK_PIXEL_THRESHOLD = 4.0    # 谱线中心与画面中心线的像素偏差阈值
LOCK_REQUIRE_FRAMES = 5       # 连续命中帧数才触发锁定
UNLOCK_FRAMES = 3             # 连续偏移超标帧数才解锁
LOCK_HOLD_SECONDS = 2.5       # 锁定后冻结帧的持续秒数
VOICE_COOLDOWN = 3.0          # 同色重复播报冷却（秒）
CONF_THRESHOLD = 0.15         # YOLO 置信度下限（与 realtime_physics 一致）

# ── 颜色 ──
COLOR_CENTER = (255, 255, 255)
COLOR_LOCKED = (0, 255, 0)
COLOR_UNLOCKED = (0, 0, 255)
COLOR_INFO = (200, 200, 200)


# ══════════════════════════════════════════════════════════
# 语音播报
# ══════════════════════════════════════════════════════════

if _voice_available:
    _voice_queue: "tts_queue.Queue[Optional[str]]" = tts_queue.Queue()

    def _voice_worker() -> None:
        while True:
            text = _voice_queue.get()
            if text is None:
                break
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", 160)
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"⚠️ 语音异常: {e}")

    threading.Thread(target=_voice_worker, daemon=True).start()

    def speak(text: str) -> None:
        _voice_queue.put(text)
else:
    def speak(text: str) -> None:
        pass


# ══════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════

def find_model() -> str:
    """按优先级搜索可用模型权重。"""
    for path in MODEL_CANDIDATES:
        if Path(path).exists():
            return path
    print("❌ 未找到模型权重！请先运行 train_model.py 训练。")
    sys.exit(1)


_PREPROC_CACHE: dict[str, object] = {}


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    图像预处理管线（与 realtime_physics.py 一致）：
      ① bilateralFilter — 保边去噪
      ② CLAHE (LAB L通道) — 局部对比度增强
      ③ Gamma 校正 (γ=0.8) — 提亮暗部
      ④ Unsharp Mask — 轻微锐化
    """
    # ① 保边去噪
    frame = cv2.bilateralFilter(frame, d=5, sigmaColor=30, sigmaSpace=30)

    # ② CLAHE
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = _PREPROC_CACHE.get("clahe")
    if clahe is None:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        _PREPROC_CACHE["clahe"] = clahe
    l_ch = clahe.apply(l_ch)
    frame = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

    # ③ Gamma
    lut = _PREPROC_CACHE.get("gamma_lut")
    if lut is None:
        inv_gamma = 1.0 / 0.8
        lut = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8
        )
        _PREPROC_CACHE["gamma_lut"] = lut
    frame = cv2.LUT(frame, lut)

    # ④ 锐化
    blur = cv2.GaussianBlur(frame, (0, 0), sigmaX=1.0)
    frame = cv2.addWeighted(frame, 1.5, blur, -0.5, 0)

    return frame


def annotate_frame(
    frame: np.ndarray,
    detections: list[dict],
    center_x: int,
    lock_states: dict,
    *,
    freeze_overlay: Optional[dict] = None,
) -> np.ndarray:
    """
    在帧上绘制：中心线 / 检测框 / 偏移指示 / 锁定状态 / 冻结叠加文字。
    直接在 frame 上绘制并返回。
    """
    h, w = frame.shape[:2]

    # ── 中心线 ──
    cv2.line(frame, (center_x, 0), (center_x, h), COLOR_CENTER, 1)
    cv2.drawMarker(
        frame, (center_x, h // 2), COLOR_CENTER, cv2.MARKER_CROSS, 20, 1
    )

    # ── 检测框 + 偏移 ──
    for det in detections:
        cls_id = det["cls_id"]
        info = WAVELENGTHS[cls_id]
        color = info["color_bgr"]

        cv2.rectangle(frame, (det["x1"], det["y1"]), (det["x2"], det["y2"]), color, 2)
        cv2.putText(
            frame,
            f"{info['name']} {det['conf']:.2f}",
            (det["x1"], det["y1"] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

        offset = det["gx"] - center_x
        direction = "◀◀" if offset < 0 else "▶▶"
        cv2.putText(
            frame,
            f"{direction} {abs(offset):.1f}px",
            (int(det["gx"]) - 30, det["y1"] - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            COLOR_UNLOCKED,
            1,
        )

    # ── 锁定状态指示器（右上角） ──
    indicator_y = 60
    for cid, info in WAVELENGTHS.items():
        state = lock_states.get(cid, {})
        locked = state.get("locked", False)
        progress = state.get("progress", 0.0)

        if locked:
            status_text = "● LOCKED"
            status_color = (0, 255, 0)
        elif progress > 0:
            status_text = f"◐ {progress:.0%}"
            status_color = (
                0,
                int(255 * progress),
                int(255 * (1 - progress)),
            )
        else:
            status_text = "○ "
            status_color = COLOR_INFO

        put_text_cn(
            frame,
            f"{info['cn_name']}: {status_text}",
            (w - 210, indicator_y),
            0.5,
            status_color,
            1,
        )
        indicator_y += 22

    # ── 冻结叠加 ──
    if freeze_overlay is not None:
        overlay_text = freeze_overlay["text"]
        overlay_color = freeze_overlay["color"]
        # 半透明背景条
        (tw, th), _ = get_text_size_cn(
            overlay_text, 1.2, 3
        )
        bx1 = (w - tw) // 2 - 20
        bx2 = bx1 + tw + 40
        by1 = h // 2 - th - 30
        by2 = h // 2 + 10
        overlay = frame.copy()
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
        put_text_cn(
            frame,
            overlay_text,
            ((w - tw) // 2, h // 2 - 10),
            1.2,
            overlay_color,
            3,
        )

    return frame


# ══════════════════════════════════════════════════════════
# 主逻辑
# ══════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="光栅视频推理 — 谱线锁定播报"
    )
    parser.add_argument(
        "video", nargs="?", default="E:/grating_yolo/videos/2.mp4", help="输入视频路径（默认 E:/grating_yolo/videos/2.mp4）"
    )
    parser.add_argument(
        "--output", "-o", default=None, help="输出视频路径（默认覆盖输入文件）"
    )
    parser.add_argument(
        "--center", type=int, default=None, help="画面中心 X 坐标（默认帧宽/2）"
    )
    parser.add_argument(
        "--no-voice", action="store_true", help="禁用语音播报"
    )
    parser.add_argument(
        "--no-freeze", action="store_true", help="禁用锁定冻结效果"
    )
    parser.add_argument(
        "--preproc", action="store_true", help="开启图像预处理（CLAHE+去噪+Gamma+锐化）"
    )
    parser.add_argument(
        "--lock-threshold",
        type=float,
        default=LOCK_PIXEL_THRESHOLD,
        help=f"锁定像素阈值（默认 {LOCK_PIXEL_THRESHOLD}）",
    )
    parser.add_argument(
        "--freeze-sec",
        type=float,
        default=LOCK_HOLD_SECONDS,
        help=f"冻结持续秒数（默认 {LOCK_HOLD_SECONDS}）",
    )
    parser.add_argument(
        "--conf", type=float, default=CONF_THRESHOLD, help=f"置信度下限（默认 {CONF_THRESHOLD}）"
    )
    args = parser.parse_args()

    # ── 检查输入 ──
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ 视频不存在: {video_path}")
        sys.exit(1)

    # ── 输出路径 ──
    if args.output:
        output_path = Path(args.output)
    else:
        # 默认输出到 results 目录，保持原文件名
        results_dir = Path("E:/grating_yolo/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = results_dir / video_path.name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 如果输出 == 输入，先写到临时文件，处理完再替换（避免同时读写同一文件）
    overwrite_input = (output_path.resolve() == video_path.resolve())
    if overwrite_input:
        actual_output = output_path.with_suffix(".tmp.mp4")
    else:
        actual_output = output_path

    # ── 加载模型 ──
    model_path = find_model()
    print(f"📦 模型: {model_path}")
    model = YOLO(model_path)

    print("🔥 模型预热中...")
    _ = model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
    print("✅ 预热完成")

    # ── 打开视频 ──
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    center_x = args.center if args.center is not None else width // 2

    print(f"📹 输入: {width}x{height} @ {fps:.1f} FPS, ~{total_frames} 帧")
    print(f"🎯 中心线: x={center_x}")
    if overwrite_input:
        print(f"📼 输出: {output_path} (覆盖原文件)")
    else:
        print(f"📼 输出: {output_path}")
    if args.preproc:
        print("🖼️  预处理: 开启 (CLAHE + 去噪 + Gamma + 锐化)")
    if args.no_voice:
        print("🔇 语音: 关闭")
    if args.no_freeze:
        print("🧊 冻结: 关闭")
    print(f"🔒 锁定阈值: {args.lock_threshold} px  |  冻结: {args.freeze_sec}s  |  置信度: ≥{args.conf}")

    # ── 视频写入器 ──
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(actual_output), fourcc, fps, (width, height))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(actual_output), fourcc, fps, (width, height))
    if not writer.isOpened():
        print("❌ 无法创建输出视频（尝试了 mp4v / avc1 编码）")
        sys.exit(1)

    # ── 状态变量 ──
    lock_counters = {cid: 0 for cid in WAVELENGTHS}
    unlock_counters = {cid: 0 for cid in WAVELENGTHS}
    last_spoken: dict[int, float] = {cid: 0.0 for cid in WAVELENGTHS}

    frozen = False
    frozen_frame: Optional[np.ndarray] = None
    freeze_until = 0.0
    active_lock_id: Optional[int] = None

    frame_idx = 0
    written_frames = 0
    lock_event_count = 0
    t_start = time.time()
    last_diag_frame = 0  # 上次诊断输出的帧号

    if not _voice_available:
        print("⚠️  pyttsx3 未安装，语音播报不可用（仅视觉标注）")

    print("=" * 60)
    print("  开始推理...")
    print("=" * 60)

    while True:
        now = time.time()

        if not frozen:
            ret, frame = cap.read()
            if not ret:
                break  # 视频结束

            # ── 预处理（可选） ──
            if args.preproc:
                frame = preprocess_frame(frame)

            # ── YOLO 推理 ──
            results = model(frame, verbose=False, conf=args.conf, iou=0.5)
            detections_all: list[dict] = []

            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls_id not in WAVELENGTHS:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    gx = (x1 + x2) / 2.0
                    detections_all.append(
                        {
                            "cls_id": cls_id,
                            "conf": conf,
                            "gx": gx,
                            "x1": int(x1),
                            "y1": int(y1),
                            "x2": int(x2),
                            "y2": int(y2),
                        }
                    )

            # ── 每种颜色只保留置信度最高的那一条 ──
            detections: list[dict] = []
            for cid in WAVELENGTHS:
                cls_dets = [d for d in detections_all if d["cls_id"] == cid]
                if cls_dets:
                    detections.append(max(cls_dets, key=lambda d: d["conf"]))

            # ── 锁定判定（带迟滞） ──
            new_lock_triggered = False

            for cid in WAVELENGTHS:
                # 检查该类是否在过滤后的检测中
                cls_dets = [d for d in detections if d["cls_id"] == cid]
                if cls_dets:
                    avg_gx = cls_dets[0]["gx"]  # 每种颜色只有一条
                    offset = abs(avg_gx - center_x)

                    if offset <= args.lock_threshold:
                        lock_counters[cid] = min(
                            lock_counters[cid] + 1, LOCK_REQUIRE_FRAMES + 10
                        )
                        unlock_counters[cid] = 0
                    else:
                        unlock_counters[cid] += 1
                        if unlock_counters[cid] >= UNLOCK_FRAMES:
                            lock_counters[cid] = max(0, lock_counters[cid] - 1)

                    # 刚达到锁定阈值
                    if lock_counters[cid] == LOCK_REQUIRE_FRAMES:
                        if not args.no_voice and (
                            now - last_spoken[cid] > VOICE_COOLDOWN
                        ):
                            name = WAVELENGTHS[cid]["cn_name"]
                            speak(f"{name}已重合")
                            last_spoken[cid] = now
                        if not args.no_freeze:
                            new_lock_triggered = True
                            active_lock_id = cid
                else:
                    # 当前帧未检测到该类 → 衰减
                    unlock_counters[cid] += 1
                    if unlock_counters[cid] >= UNLOCK_FRAMES:
                        lock_counters[cid] = max(0, lock_counters[cid] - 1)

            # ── 构建锁定状态字典 ──
            lock_states: dict[int, dict] = {}
            for cid in WAVELENGTHS:
                progress = min(lock_counters[cid] / LOCK_REQUIRE_FRAMES, 1.0)
                lock_states[cid] = {
                    "locked": progress >= 1.0,
                    "progress": progress,
                }

            # ── 标注帧 ──
            annotated = annotate_frame(frame, detections, center_x, lock_states)

            # ── 如果触发锁定，准备冻结 ──
            if new_lock_triggered:
                frozen = True
                info = WAVELENGTHS[active_lock_id]
                # 先标注再叠加冻结文字
                frozen_frame = annotate_frame(
                    annotated.copy(), detections, center_x, lock_states,
                    freeze_overlay={
                        "text": f"🔒 {info['cn_name']} 已锁定!",
                        "color": info["color_bgr"],
                    },
                )
                freeze_until = now + args.freeze_sec
                lock_event_count += 1
                print(
                    f"  🔒 第 {frame_idx} 帧: {info['cn_name']} 锁定 "
                    f"(冻结 {args.freeze_sec}s, 事件 #{lock_event_count})"
                )

            writer.write(annotated)
            written_frames += 1

        else:
            # ── 冻结状态：重复写入冻结帧 ──
            writer.write(frozen_frame)
            written_frames += 1

            if now >= freeze_until:
                frozen = False
                active_lock_id = None
                frozen_frame = None
                # 重置所有锁定计数器，避免解冻后立即重新触发
                for cid in WAVELENGTHS:
                    lock_counters[cid] = 0
                    unlock_counters[cid] = 0

        frame_idx += 1

        # ── 诊断输出（每 30 帧打印一次检测摘要） ──
        if frame_idx - last_diag_frame >= 30:
            last_diag_frame = frame_idx
            elapsed = now - t_start
            fps_actual = frame_idx / elapsed if elapsed > 0 else 0

            # 汇总当前检测状态
            diag_parts = []
            for cid, info in WAVELENGTHS.items():
                lc = lock_counters[cid]
                progress = min(lc / LOCK_REQUIRE_FRAMES, 1.0)
                if lc > 0:
                    diag_parts.append(
                        f"{info['cn_name']}:cnt={lc}/{LOCK_REQUIRE_FRAMES} prog={progress:.0%}"
                    )
            det_summary = " | ".join(diag_parts) if diag_parts else "无检测"
            print(
                f"  [{frame_idx:5d}] {fps_actual:5.1f}fps  "
                f"检测={len(detections)}个  {det_summary}"
            )

    # ── 清理 ──
    cap.release()
    writer.release()
    if _voice_available:
        _voice_queue.put(None)

    # ── 如果覆盖模式，用临时文件替换原文件 ──
    if overwrite_input:
        print("  正在替换原文件...")
        actual_output.replace(output_path)

    elapsed = time.time() - t_start
    print("=" * 60)
    print("✅ 推理完成！")
    print(f"   输入帧数: {frame_idx}")
    print(f"   输出帧数: {written_frames}")
    print(f"   锁定事件: {lock_event_count} 次")
    print(f"   耗时:     {elapsed:.1f}s  ({frame_idx / elapsed:.1f} fps)" if elapsed > 0 else "")
    print(f"   输出:     {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
