#!/usr/bin/env python3
"""
光栅衍射三色谱线 — 实时检测与锁定系统
========================================
功能:
  - YOLOv8 实时检测紫/绿/黄三条谱线
  - 滑动窗口平滑 + 防抖锁定 + 语音播报
  - 光栅常数实时计算
  - 可调曝光 / 亮度 / 转盘角度

键盘操作:
  Q / ESC    退出
  C          校准准星到当前谱线位置
  R          重置准星到画面中心
  ↑ ↓        微调转盘角度 (±0.1°)
  ← →        粗调转盘角度 (±1.0°)
  S          截屏保存
  H          显示/隐藏帮助
  Space      手动触发语音播报
"""

import cv2
import math
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from utils.cv2_chinese import put_text_cn

# ── 可选依赖 ──
try:
    import pyttsx3
    import threading
    import queue as tts_queue

    _voice_available = True
except ImportError:
    _voice_available = False

try:
    import pythoncom  # noqa: F401 — Windows COM 支持
except ImportError:
    pass


# ══════════════════════════════════════════════════════════
# 1. 配置区
# ══════════════════════════════════════════════════════════

# ── 谱线物理参数 ──
WAVELENGTHS = {
    0: {
        "name": "Purple",
        "cn_name": "紫色谱线",
        "lambda_nm": 435.8,
        "color_bgr": (255, 0, 255),
        "threshold": 3.5,  # 锁定像素阈值
        "conf_limit": 0.50,  # 置信度下限
    },
    1: {
        "name": "Green",
        "cn_name": "绿色谱线",
        "lambda_nm": 546.1,
        "color_bgr": (0, 255, 0),
        "threshold": 3.0,
        "conf_limit": 0.40,
    },
    2: {
        "name": "Yellow",
        "cn_name": "黄色谱线",
        "lambda_nm": 578.0,
        "color_bgr": (0, 255, 255),
        "threshold": 4.5,
        "conf_limit": 0.20,
    },
}

# ── 摄像头 ──
CAMERA_INDEX = 1  # 外接 USB 摄像头通常为 1；内置为 0
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

# ── 模型路径（优先级从高到低） ──
MODEL_CANDIDATES = [
    "runs/detect/grating_ca_p2_v3/weights/best.pt",  # 无 ECA 新版
]

# ── 锁定与平滑 ──
HISTORY_LEN = 10  # 滑动窗口长度（越大越平滑，但响应越慢）
LOCK_REQUIREMENT = 5  # 连续命中帧数才触发锁定
UNLOCK_FRAMES = 3  # 连续偏移超标帧数才解锁
VOICE_COOLDOWN = 3.0  # 同色重复播报冷却 (秒)

# ── 颜色 ──
COLOR_CENTER = (255, 255, 255)  # 准星线
COLOR_LOCKED = (0, 255, 0)  # 已锁定文字
COLOR_UNLOCKED = (0, 0, 255)  # 未锁定偏移指示
COLOR_INFO = (200, 200, 200)  # 信息文字
COLOR_HELP = (180, 180, 180)  # 帮助文字


# ══════════════════════════════════════════════════════════
# 2. 语音播报系统
# ══════════════════════════════════════════════════════════

if _voice_available:
    _voice_queue: "tts_queue.Queue[str | None]" = tts_queue.Queue()

    def _voice_worker() -> None:
        """后台语音线程：Windows COM 兼容。"""
        try:
            import pythoncom
            pythoncom.CoInitialize()  # type: ignore[union-attr]
        except ImportError:
            pass
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
                print(f"⚠️ 语音播报异常: {e}")

    threading.Thread(target=_voice_worker, daemon=True).start()

    def speak(text: str) -> None:
        _voice_queue.put(text)
else:
    def speak(text: str) -> None:
        pass  # 静默降级


# ══════════════════════════════════════════════════════════
# 3. 工具函数
# ══════════════════════════════════════════════════════════

def find_model() -> str:
    """按优先级搜索可用模型权重。"""
    for path in MODEL_CANDIDATES:
        if Path(path).exists():
            return path
    print("❌ 未找到任何模型权重文件！请先运行 train_model.py 训练。")
    sys.exit(1)


def calc_grating_constant(wavelength_nm: float, theta_deg: float, k: int = 1) -> float:
    """光栅常数 d = kλ / sin(θ)"""
    if theta_deg <= 0:
        return 0.0
    return (k * wavelength_nm) / math.sin(math.radians(theta_deg))


def draw_centered_text(img, text: str, y: int, *, color=COLOR_INFO, scale=0.7, thickness=2):
    """在图像顶部水平居中绘制文字。"""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = (img.shape[1] - tw) // 2
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


# ══════════════════════════════════════════════════════════
# 3.5 图像预处理
# ══════════════════════════════════════════════════════════

_PREPROC_CACHE: dict[str, object] = {}  # 缓存 CLAHE 对象等，避免每帧重建


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    对摄像头帧做预处理，增强三色谱线的可分辨性。

    管线（按顺序）：
      ① bilateralFilter — 保边去噪，不模糊细谱线
      ② CLAHE (LAB L通道) — 局部对比度增强，让暗弱谱线凸显
      ③ Gamma 校正 (γ=0.8) — 非线性提亮暗部
      ④ Unsharp Mask — 轻微锐化，强化细线边缘

    注意：不修改色调和饱和度，与训练时的 hsv_h=0 / hsv_s=0 约束一致。
    """
    # ── ① 保边去噪 ──
    frame = cv2.bilateralFilter(frame, d=5, sigmaColor=30, sigmaSpace=30)

    # ── ② CLAHE 局部对比度增强（LAB 的 L 通道） ──
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    clahe = _PREPROC_CACHE.get("clahe")
    if clahe is None:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        _PREPROC_CACHE["clahe"] = clahe
    l_ch = clahe.apply(l_ch)

    frame = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

    # ── ③ Gamma 校正 (γ=0.8，提亮暗部) ──
    gamma = _PREPROC_CACHE.get("gamma_lut")
    if gamma is None:
        inv_gamma = 1.0 / 0.8
        gamma = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8
        )
        _PREPROC_CACHE["gamma_lut"] = gamma
    frame = cv2.LUT(frame, gamma)

    # ── ④ 轻微锐化（Unsharp Mask） ──
    blur = cv2.GaussianBlur(frame, (0, 0), sigmaX=1.0)
    frame = cv2.addWeighted(frame, 1.5, blur, -0.5, 0)

    return frame


# ══════════════════════════════════════════════════════════
# 4. 主循环
# ══════════════════════════════════════════════════════════

def main() -> None:
    # ── 加载模型 ──
    model_path = find_model()
    print(f"📦 模型: {model_path}")
    model = YOLO(model_path)

    # ── 模型预热（避免首帧卡顿） ──
    print("🔥 模型预热中...")
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    _ = model(dummy, verbose=False)
    print("✅ 预热完成")

    # ── 打开摄像头 ──
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"❌ 无法打开摄像头 (索引 {CAMERA_INDEX})！尝试索引 0...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ 所有摄像头均无法打开！")
            return

    # 摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 手动曝光模式
    cap.set(cv2.CAP_PROP_EXPOSURE, -5)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"📷 分辨率: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")

    # ── 窗口与 UI ──
    WINDOW_NAME = "Spectrometer Tracker"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    def _noop(_val):
        pass

    cv2.createTrackbar("Exposure", WINDOW_NAME, 5, 10, lambda v: cap.set(cv2.CAP_PROP_EXPOSURE, v - 10))
    cv2.createTrackbar("Brightness", WINDOW_NAME, 100, 200, _noop)
    cv2.createTrackbar("Conf Boost", WINDOW_NAME, 0, 30, _noop)  # 0~30 → 置信度补偿 -0.15~+0.15
    cv2.createTrackbar("Preproc", WINDOW_NAME, 1, 1, _noop)  # 0=关 1=开，图像预处理

    # ── 状态变量 ──
    calibrated_center = actual_w // 2
    dial_angle = 15.5  # 转盘角度（度）

    # 平滑历史
    history_x = {cid: deque(maxlen=HISTORY_LEN) for cid in WAVELENGTHS}
    # 锁定计数
    lock_counters = {cid: 0 for cid in WAVELENGTHS}
    unlock_counters = {cid: 0 for cid in WAVELENGTHS}
    locked_id: int | None = None

    # 语音状态
    last_spoken_id: int | None = None
    last_spoken_time = 0.0

    # FPS 统计
    fps_deque: deque[float] = deque(maxlen=30)
    frame_t0 = time.perf_counter()

    # 帮助开关
    show_help = False

    # 截图目录
    screenshot_dir = Path("screenshots")
    screenshot_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("  实时检测已启动")
    print(f"  准星位置: x={calibrated_center}")
    print(f"  转盘角度: {dial_angle}°")
    print("  按 H 显示键盘帮助")
    print("=" * 60)

    # ══════════════════════════════════════════════════════
    # 主循环
    # ══════════════════════════════════════════════════════
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 无法读取画面")
            break

        # ── 软件亮度 ──
        bright_val = cv2.getTrackbarPos("Brightness", WINDOW_NAME)
        if bright_val != 100:
            frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=bright_val - 100)

        # ── 置信度补偿 ──
        conf_boost = cv2.getTrackbarPos("Conf Boost", WINDOW_NAME) / 100.0 - 0.15

        # ── 图像预处理（增强谱线可分辨性） ──
        if cv2.getTrackbarPos("Preproc", WINDOW_NAME):
            frame = preprocess_frame(frame)

        # ── 画准星 ──
        h, w = frame.shape[:2]
        cv2.line(frame, (calibrated_center, 0), (calibrated_center, h), COLOR_CENTER, 1)
        # 准星标记（十字）
        cv2.drawMarker(
            frame, (calibrated_center, h // 2), COLOR_CENTER,
            cv2.MARKER_CROSS, 20, 1,
        )

        # ── YOLO 推理 ──
        results = model(frame, verbose=False, conf=0.15, iou=0.5)
        detections_all: list[dict] = []  # 当前帧的所有有效检测

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if cls_id not in WAVELENGTHS:
                    continue
                info = WAVELENGTHS[cls_id]
                if conf + conf_boost < info["conf_limit"]:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                gx = (x1 + x2) / 2.0
                bw = x2 - x1  # 谱线宽度

                detections_all.append({
                    "cls_id": cls_id,
                    "conf": conf,
                    "gx": gx,
                    "bw": bw,
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                })

        # ── 每种颜色只保留置信度最高的那一条 ──
        detections: list[dict] = []
        for cls_id in WAVELENGTHS:
            cls_dets = [d for d in detections_all if d["cls_id"] == cls_id]
            if cls_dets:
                detections.append(max(cls_dets, key=lambda d: d["conf"]))

        # ── 更新平滑历史 ──
        for det in detections:
            history_x[det["cls_id"]].append(det["gx"])

        # ── 绘制检测框 & 锁定判定 ──
        current_aligned_id: int | None = None

        for det in detections:
            cls_id = det["cls_id"]
            info = WAVELENGTHS[cls_id]

            # 平滑位置
            if len(history_x[cls_id]) > 0:
                stable_x = sum(history_x[cls_id]) / len(history_x[cls_id])
            else:
                stable_x = det["gx"]

            offset = stable_x - calibrated_center
            color = info["color_bgr"]

            # ── 画框 ──
            cv2.rectangle(frame, (det["x1"], det["y1"]), (det["x2"], det["y2"]), color, 2)
            cv2.putText(
                frame, f"{info['name']} {det['conf']:.2f}",
                (det["x1"], det["y1"] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
            )

            # ── 锁定判定（带迟滞） ──
            if abs(offset) <= info["threshold"]:
                lock_counters[cls_id] = min(lock_counters[cls_id] + 1, LOCK_REQUIREMENT + 10)
                unlock_counters[cls_id] = 0
            else:
                unlock_counters[cls_id] += 1
                if unlock_counters[cls_id] >= UNLOCK_FRAMES:
                    lock_counters[cls_id] = max(0, lock_counters[cls_id] - 1)

            # ── 锁定状态绘制 ──
            if lock_counters[cls_id] >= LOCK_REQUIREMENT:
                current_aligned_id = cls_id
                lock_counters[cls_id] = LOCK_REQUIREMENT  # 封顶

                # 锁定图标 (在谱线旁边画 ⬤)
                cv2.circle(frame, (calibrated_center, det["y1"] - 35), 8, (0, 255, 0), -1)
                put_text_cn(
                    frame, f"LOCKED: {info['cn_name']}",
                    (calibrated_center + 15, det["y1"] - 25),
                    0.8, (0, 255, 0), 2,
                )
            else:
                # 偏移指示
                direction = "◀◀" if offset < 0 else "▶▶"
                progress = min(lock_counters[cls_id] / LOCK_REQUIREMENT, 1.0)
                bar_len = int(60 * progress)
                bar_color = (
                    int(255 * (1 - progress)),
                    int(255 * progress),
                    0,
                )

                # 进度条（在谱线上方）
                bar_y = det["y1"] - 35
                cv2.rectangle(frame, (det["x1"], bar_y), (det["x2"], bar_y + 4), bar_color, -1)

                # 偏移文字
                cv2.putText(
                    frame, f"{direction} {abs(offset):.1f}px",
                    (int(stable_x) - 30, det["y1"] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_UNLOCKED, 1,
                )

        # ── 无检测到任何谱线时，衰减所有锁 ──
        if not detections:
            for cid in WAVELENGTHS:
                lock_counters[cid] = max(0, lock_counters[cid] - 1)

        # ── 语音播报 ──
        now = time.time()
        if current_aligned_id is not None:
            should_speak = (
                current_aligned_id != last_spoken_id
                or (now - last_spoken_time > VOICE_COOLDOWN)
            )
            if should_speak:
                name = WAVELENGTHS[current_aligned_id]["cn_name"]
                speak(f"{name}已重合")
                last_spoken_id = current_aligned_id
                last_spoken_time = now
        else:
            if last_spoken_id is not None and now - last_spoken_time > 2.0:
                last_spoken_id = None

        # ── 信息叠加 ──
        # FPS
        frame_t1 = time.perf_counter()
        fps_deque.append(1.0 / max(frame_t1 - frame_t0, 0.001))
        frame_t0 = frame_t1
        avg_fps = sum(fps_deque) / len(fps_deque) if fps_deque else 0

        cv2.putText(
            frame, f"FPS: {avg_fps:.0f}",
            (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_INFO, 1,
        )

        # 角度 + 光栅常数
        cv2.putText(
            frame, f"Angle: {dial_angle:.3f} deg",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_INFO, 2,
        )
        if locked_id is not None and current_aligned_id is not None:
            locked_id = current_aligned_id  # 更新
        elif current_aligned_id is not None:
            locked_id = current_aligned_id
        elif not detections:
            locked_id = None

        if locked_id is not None and current_aligned_id == locked_id:
            d_val = calc_grating_constant(
                WAVELENGTHS[locked_id]["lambda_nm"], dial_angle
            )
            cv2.putText(
                frame, f"d = {d_val:.1f} nm",
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                WAVELENGTHS[locked_id]["color_bgr"], 2,
            )

        # 各谱线锁定状态指示器（右上角）
        indicator_y = 60
        for cid, info in WAVELENGTHS.items():
            progress = min(lock_counters[cid] / LOCK_REQUIREMENT, 1.0)
            if progress >= 1.0:
                status = "● LOCKED"
                status_color = (0, 255, 0)
            elif progress > 0:
                status = f"◐ {progress:.0%}"
                status_color = (0, int(255 * progress), int(255 * (1 - progress)))
            else:
                status = "○ "
                status_color = COLOR_INFO

            put_text_cn(
                frame, f"{info['cn_name']}: {status}",
                (w - 200, indicator_y), 0.5, status_color, 1,
            )
            indicator_y += 22

        # ── 帮助叠加 ──
        if show_help:
            help_lines = [
                "Q/ESC: 退出    H: 隐藏帮助",
                "C: 校准准星    R: 重置准星",
                "↑↓: 角度±0.1   ←→: 角度±1.0",
                "S: 截屏       Space: 播报",
            ]
            help_y = h - 110
            cv2.rectangle(frame, (5, help_y - 5), (350, h - 5), (0, 0, 0), -1)
            cv2.rectangle(frame, (5, help_y - 5), (350, h - 5), (80, 80, 80), 1)
            for i, line in enumerate(help_lines):
                cv2.putText(
                    frame, line, (10, help_y + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_HELP, 1,
                )

        # ── 显示 ──
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF

        # ── 键盘处理 ──
        if key == ord("q") or key == 27:  # Q 或 ESC
            break
        elif key == ord("c"):
            # 校准：将当前谱线位置设为准星
            if detections:
                # 取置信度最高的检测
                best = max(detections, key=lambda d: d["conf"])
                calibrated_center = int(best["gx"])
                print(f"🎯 准星已校准到: x={calibrated_center}")
        elif key == ord("r"):
            calibrated_center = w // 2
            print(f"🔄 准星已重置到中心: x={calibrated_center}")
        elif key == ord("h"):
            show_help = not show_help
        elif key == ord("s"):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = screenshot_dir / f"grating_{ts}.png"
            cv2.imwrite(str(fname), frame)
            print(f"📸 截图已保存: {fname}")
        elif key == ord(" "):
            if locked_id is not None:
                name = WAVELENGTHS[locked_id]["cn_name"]
                speak(f"{name}已重合")
        elif key == 0:  # 上箭头
            dial_angle = round(dial_angle + 0.1, 1)
        elif key == 1:  # 下箭头
            dial_angle = round(dial_angle - 0.1, 1)
        elif key == 2:  # 左箭头
            dial_angle = round(dial_angle - 1.0, 1)
        elif key == 3:  # 右箭头
            dial_angle = round(dial_angle + 1.0, 1)
        # 兼容某些系统的方向键映射
        elif key == 82:  # ↑ (某些平台)
            dial_angle = round(dial_angle + 0.1, 1)
        elif key == 84:  # ↓
            dial_angle = round(dial_angle - 0.1, 1)
        elif key == 81:  # ←
            dial_angle = round(dial_angle - 1.0, 1)
        elif key == 83:  # →
            dial_angle = round(dial_angle + 1.0, 1)

    # ── 清理 ──
    cap.release()
    cv2.destroyAllWindows()
    if _voice_available:
        _voice_queue.put(None)

    print("👋 实时检测已退出")


if __name__ == "__main__":
    main()
