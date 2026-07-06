#!/usr/bin/env python3
"""
光栅衍射实验 — 桌面控制台
==========================
Tkinter UI，集成摄像头预览与 YOLO 实时检测。

用法:
    python grating_ui.py
"""

from __future__ import annotations

import math
import sys
import time
import threading
import queue as tts_queue
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk

from utils.cv2_chinese import put_text_cn

# ── Tkinter ──
import tkinter as tk
from tkinter import ttk, messagebox

# ── YOLO ──
from ultralytics import YOLO

# ── 语音 ──
try:
    import pyttsx3
    _VOICE_OK = True
except ImportError:
    _VOICE_OK = False


# ══════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════

WAVELENGTHS = {
    0: {"name": "Purple", "cn_name": "紫色谱线", "lambda_nm": 435.8, "color_hex": "#FF00FF"},
    1: {"name": "Green",  "cn_name": "绿色谱线", "lambda_nm": 546.1, "color_hex": "#00FF00"},
    2: {"name": "Yellow", "cn_name": "黄色谱线", "lambda_nm": 578.0, "color_hex": "#FFFF00"},
}

MODEL_CANDIDATES = [
    "runs/detect/grating_ca_p2_v3/weights/best.pt",
]

CAMERA_INDEX = 1
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
PREVIEW_WIDTH = 960
PREVIEW_HEIGHT = 540

LOCK_REQUIRE_FRAMES = 8
UNLOCK_FRAMES = 12
LOCK_PIXEL_THRESHOLD = 2.0
VOICE_COOLDOWN = 3.0

# ── 预处理缓存 ──
_PREPROC_CACHE: dict = {}


# ══════════════════════════════════════════════════════════
# 语音播报
# ══════════════════════════════════════════════════════════

if _VOICE_OK:
    _voice_queue: "tts_queue.Queue" = tts_queue.Queue()

    def _voice_worker():
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

    def speak(text: str):
        _voice_queue.put(text)
else:
    def speak(text: str):
        pass


# ══════════════════════════════════════════════════════════
# 预处理
# ══════════════════════════════════════════════════════════

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """CLAHE + 去噪 + Gamma + 锐化"""
    frame = cv2.bilateralFilter(frame, d=5, sigmaColor=30, sigmaSpace=30)

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = _PREPROC_CACHE.get("clahe")
    if clahe is None:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        _PREPROC_CACHE["clahe"] = clahe
    l_ch = clahe.apply(l_ch)
    frame = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

    lut = _PREPROC_CACHE.get("gamma_lut")
    if lut is None:
        inv_gamma = 1.0 / 0.8
        lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
        _PREPROC_CACHE["gamma_lut"] = lut
    frame = cv2.LUT(frame, lut)

    blur = cv2.GaussianBlur(frame, (0, 0), sigmaX=1.0)
    frame = cv2.addWeighted(frame, 1.5, blur, -0.5, 0)
    return frame


# ══════════════════════════════════════════════════════════
# 主窗口
# ══════════════════════════════════════════════════════════

class GratingApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("光栅衍射实验")
        self.root.geometry("1400x750")
        self.root.minsize(1200, 650)
        self.root.configure(bg="#1a1a2e")

        # ── 状态 ──
        self.cap: cv2.VideoCapture | None = None
        self.model: YOLO | None = None
        self.detecting = False
        self.running = True
        self.frame_count = 0
        self.fps_deque: deque[float] = deque(maxlen=30)
        self.t_frame = time.perf_counter()
        self._last_infer_ms: float = 0.0

        # 锁定状态
        self.lock_counters = {cid: 0 for cid in WAVELENGTHS}
        self.unlock_counters = {cid: 0 for cid in WAVELENGTHS}
        self.last_spoken = {cid: 0.0 for cid in WAVELENGTHS}
        self.locked_id: int | None = None

        # 准星
        self.center_x = PREVIEW_WIDTH // 2

        # 截图
        self.screenshot_dir = Path("screenshots")
        self.screenshot_dir.mkdir(exist_ok=True)

        # ── 样式 ──
        self._setup_style()

        # ── 构建 UI ──
        self._build_ui()

        # ── 打开摄像头 ──
        self._open_camera()

        # ── 帧刷新循环 ──
        self._update_frame()

        # ── 关闭处理 ──
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ══════════════════════════════════════════════════════
    # 样式
    # ══════════════════════════════════════════════════════

    def _setup_style(self):
        style = ttk.Style()
        style.theme_use("clam")

        bg = "#1a1a2e"
        fg = "#e0e0e0"
        accent = "#e94560"
        panel_bg = "#16213e"
        btn_bg = "#0f3460"

        style.configure("TFrame", background=bg)
        style.configure("Panel.TFrame", background=panel_bg)
        style.configure("TLabelframe.Label", background=panel_bg, foreground=fg,
                        font=("Microsoft YaHei UI", 12, "bold"))
        style.configure("TLabel", background=bg, foreground=fg, font=("Microsoft YaHei UI", 12))
        style.configure("Panel.TLabel", background=panel_bg, foreground=fg, font=("Microsoft YaHei UI", 12))
        style.configure("Title.TLabel", background=bg, foreground="#e94560",
                        font=("Microsoft YaHei UI", 20, "bold"))
        style.configure("Status.TLabel", background=panel_bg, foreground="#a0a0a0",
                        font=("Microsoft YaHei UI", 11))

        style.configure("Start.TButton", font=("Microsoft YaHei UI", 14, "bold"),
                        background="#00b894", foreground="white", padding=(30, 12))
        style.configure("Stop.TButton", font=("Microsoft YaHei UI", 14, "bold"),
                        background="#d63031", foreground="white", padding=(30, 12))
        style.configure("Action.TButton", font=("Microsoft YaHei UI", 11),
                        background=btn_bg, foreground=fg, padding=(15, 8))

        style.configure("Green.TLabel", background=panel_bg, foreground="#00b894",
                        font=("Microsoft YaHei UI", 12, "bold"))
        style.configure("Red.TLabel", background=panel_bg, foreground="#e94560",
                        font=("Microsoft YaHei UI", 12))
        style.configure("Yellow.TLabel", background=panel_bg, foreground="#fdcb6e",
                        font=("Microsoft YaHei UI", 12))
        style.configure("Data.TLabel", background=panel_bg, foreground="#c0c0d0",
                        font=("Microsoft YaHei UI", 11))
        style.configure("Calc.TButton", font=("Microsoft YaHei UI", 11, "bold"),
                        background="#6c5ce7", foreground="white", padding=(16, 8))

        style.map("Start.TButton", background=[("active", "#00a381")])
        style.map("Stop.TButton", background=[("active", "#c0262c")])
        style.map("Action.TButton", background=[("active", "#1a4a8a")])
        style.map("Calc.TButton", background=[("active", "#5a4bd1")])

    # ══════════════════════════════════════════════════════
    # 界面构建
    # ══════════════════════════════════════════════════════

    def _build_ui(self):
        # ── 顶栏 ──
        top_bar = ttk.Frame(self.root, style="TFrame")
        top_bar.pack(fill=tk.X, padx=0, pady=(0, 0))
        ttk.Label(top_bar, text="光栅衍射实验", style="Title.TLabel").pack(
            side=tk.LEFT, padx=20, pady=10
        )

        # ── 主体 ──
        main_frame = ttk.Frame(self.root, style="TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # 左侧：视频预览
        left_frame = ttk.Frame(main_frame, style="Panel.TFrame")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.canvas = tk.Canvas(
            left_frame, width=PREVIEW_WIDTH, height=PREVIEW_HEIGHT,
            bg="#000000", highlightthickness=1, highlightbackground="#333355",
        )
        self.canvas.pack(padx=5, pady=5)
        self.canvas_image = self.canvas.create_image(
            PREVIEW_WIDTH // 2, PREVIEW_HEIGHT // 2, anchor=tk.CENTER
        )

        # 截图按钮（视频下方）
        canvas_btns = ttk.Frame(left_frame, style="Panel.TFrame")
        canvas_btns.pack(fill=tk.X, padx=5, pady=(0, 5))
        ttk.Button(canvas_btns, text="📸 截屏 (S)", style="Action.TButton",
                   command=self._screenshot).pack(side=tk.LEFT, padx=5)
        ttk.Button(canvas_btns, text="🔄 重置准星 (R)", style="Action.TButton",
                   command=self._reset_center).pack(side=tk.LEFT, padx=5)

        # ── 右侧：控制面板 ──
        right_frame = ttk.Frame(main_frame, style="Panel.TFrame", width=460)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        right_frame.pack_propagate(False)

        # 滚动容器
        canvas_right = tk.Canvas(right_frame, bg="#16213e", highlightthickness=0, width=440)
        scrollbar = ttk.Scrollbar(right_frame, orient=tk.VERTICAL, command=canvas_right.yview)
        scrollable = ttk.Frame(canvas_right, style="Panel.TFrame")

        scrollable.bind("<Configure>", lambda e: canvas_right.configure(scrollregion=canvas_right.bbox("all")))
        canvas_right.create_window((0, 0), window=scrollable, anchor=tk.NW)
        canvas_right.configure(yscrollcommand=scrollbar.set)

        canvas_right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 鼠标滚轮
        def _on_mousewheel(event):
            canvas_right.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas_right.bind_all("<MouseWheel>", _on_mousewheel)

        # ── 摄像头信息 ──
        cam_group = ttk.LabelFrame(scrollable, text="📷 摄像头", style="Panel.TFrame")
        cam_group.pack(fill=tk.X, padx=8, pady=(8, 4))
        self.lbl_cam_info = ttk.Label(cam_group, text="正在连接...", style="Panel.TLabel")
        self.lbl_cam_info.pack(anchor=tk.W, padx=8, pady=4)

        # ── 模型信息 ──
        model_group = ttk.LabelFrame(scrollable, text="📦 模型", style="Panel.TFrame")
        model_group.pack(fill=tk.X, padx=8, pady=4)
        self.lbl_model_info = ttk.Label(model_group, text="未加载", style="Panel.TLabel")
        self.lbl_model_info.pack(anchor=tk.W, padx=8, pady=4)

        # ── 谱线状态 ──
        status_group = ttk.LabelFrame(scrollable, text="🎯 谱线状态", style="Panel.TFrame")
        status_group.pack(fill=tk.X, padx=8, pady=4)

        self.color_status_labels = {}
        self.color_offset_labels = {}
        for cid, info in WAVELENGTHS.items():
            row = ttk.Frame(status_group, style="Panel.TFrame")
            row.pack(fill=tk.X, padx=8, pady=2)

            # 颜色圆点
            dot = tk.Canvas(row, width=16, height=16, bg="#16213e", highlightthickness=0)
            dot.create_oval(2, 2, 14, 14, fill=info["color_hex"], outline="")
            dot.pack(side=tk.LEFT, padx=(0, 6))

            ttk.Label(row, text=f"{info['cn_name']} ({info['lambda_nm']}nm)",
                      style="Panel.TLabel").pack(side=tk.LEFT)

            lbl_status = ttk.Label(row, text="○ 待检测", style="Status.TLabel")
            lbl_status.pack(side=tk.RIGHT, padx=(0, 4))
            self.color_status_labels[cid] = lbl_status

            lbl_offset = ttk.Label(row, text="", style="Status.TLabel")
            lbl_offset.pack(side=tk.RIGHT, padx=(0, 8))
            self.color_offset_labels[cid] = lbl_offset

        # ── 测量结果 ──
        measure_group = ttk.LabelFrame(scrollable, text="测量", style="Panel.TFrame")
        measure_group.pack(fill=tk.X, padx=8, pady=4)
        self.lbl_grating = ttk.Label(measure_group, text="光栅常数: --", style="Panel.TLabel")
        self.lbl_grating.pack(anchor=tk.W, padx=8, pady=2)
        self.lbl_angle = ttk.Label(measure_group, text="转盘角度: 15.5°", style="Panel.TLabel")
        self.lbl_angle.pack(anchor=tk.W, padx=8, pady=2)

        # ── 第一组：紫色谱线数据（光栅垂直检查）──
        group1 = ttk.LabelFrame(scrollable, text=" 紫色谱线 — 光栅垂直检查", style="Panel.TFrame")
        group1.pack(fill=tk.X, padx=8, pady=4)

        # 帮助：输入行
        g1_help = ttk.Label(group1,
                            text="锁定紫色谱线时，记录分光计刻度盘示数：",
                            style="Data.TLabel")
        g1_help.pack(anchor=tk.W, padx=8, pady=(6, 2))

        self.entry_white_deg, self.entry_white_min = self._make_dms_row(group1, "中心白线")
        self.entry_pl_deg, self.entry_pl_min = self._make_dms_row(group1, "左侧紫色")
        self.entry_pr_deg, self.entry_pr_min = self._make_dms_row(group1, "右侧紫色")

        # 检查按钮
        g1_btn_row = ttk.Frame(group1, style="Panel.TFrame")
        g1_btn_row.pack(fill=tk.X, padx=8, pady=(8, 4))
        ttk.Button(g1_btn_row, text="检查垂直", style="Calc.TButton",
                   command=self._check_perpendicular).pack(side=tk.LEFT)
        self.lbl_perp_result = ttk.Label(g1_btn_row, text="", style="Data.TLabel")
        self.lbl_perp_result.pack(side=tk.LEFT, padx=(12, 0))

        # ── 第二组：绿色谱线数据（光栅常数计算）──
        group2 = ttk.LabelFrame(scrollable, text=" 绿色谱线 — 光栅常数计算", style="Panel.TFrame")
        group2.pack(fill=tk.X, padx=8, pady=4)

        g2_help = ttk.Label(group2,
                            text="锁定绿色谱线时，记录双游标刻度盘示数：",
                            style="Data.TLabel")
        g2_help.pack(anchor=tk.W, padx=8, pady=(6, 2))

        self.entry_gl1_deg, self.entry_gl1_min = self._make_dms_row(group2, "左侧游标1")
        self.entry_gl2_deg, self.entry_gl2_min = self._make_dms_row(group2, "左侧游标2")
        self.entry_gr1_deg, self.entry_gr1_min = self._make_dms_row(group2, "右侧游标1")
        self.entry_gr2_deg, self.entry_gr2_min = self._make_dms_row(group2, "右侧游标2")

        # 计算按钮
        g2_btn_row = ttk.Frame(group2, style="Panel.TFrame")
        g2_btn_row.pack(fill=tk.X, padx=8, pady=(8, 4))
        ttk.Button(g2_btn_row, text="计算光栅常数", style="Calc.TButton",
                   command=self._calc_grating).pack(side=tk.LEFT)

        # 结果区
        self.lbl_d_result = ttk.Label(group2, text="", style="Data.TLabel")
        self.lbl_d_result.pack(anchor=tk.W, padx=8, pady=(4, 0))
        self.lbl_theta_result = ttk.Label(group2, text="", style="Data.TLabel")
        self.lbl_theta_result.pack(anchor=tk.W, padx=8, pady=(0, 4))

        # ── 性能 ──
        perf_group = ttk.LabelFrame(scrollable, text="⏱ 性能", style="Panel.TFrame")
        perf_group.pack(fill=tk.X, padx=8, pady=4)
        self.lbl_fps = ttk.Label(perf_group, text="FPS: --", style="Panel.TLabel")
        self.lbl_fps.pack(anchor=tk.W, padx=8, pady=2)
        self.lbl_infer = ttk.Label(perf_group, text="推理: -- ms", style="Panel.TLabel")
        self.lbl_infer.pack(anchor=tk.W, padx=8, pady=2)

        # ── 控制按钮 ──
        btn_group = ttk.Frame(scrollable, style="Panel.TFrame")
        btn_group.pack(fill=tk.X, padx=8, pady=(12, 8))

        self.btn_start = ttk.Button(btn_group, text="▶  开始检测", style="Start.TButton",
                                    command=self._start_detection)
        self.btn_start.pack(fill=tk.X, pady=3)

        self.btn_stop = ttk.Button(btn_group, text="⏹  停止检测", style="Stop.TButton",
                                   command=self._stop_detection, state=tk.DISABLED)
        self.btn_stop.pack(fill=tk.X, pady=3)

        # ── 准星微调 ──
        cross_group = ttk.Frame(scrollable, style="Panel.TFrame")
        cross_group.pack(fill=tk.X, padx=8, pady=(4, 8))
        ttk.Label(cross_group, text="准星微调", style="Panel.TLabel").pack(anchor=tk.W, padx=8)
        arrows = ttk.Frame(cross_group, style="Panel.TFrame")
        arrows.pack(fill=tk.X, padx=8, pady=4)
        ttk.Button(arrows, text="◀◀", style="Action.TButton",
                   command=lambda: self._nudge_center(-5)).pack(side=tk.LEFT, padx=2)
        ttk.Button(arrows, text="◀", style="Action.TButton",
                   command=lambda: self._nudge_center(-1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(arrows, text="▶", style="Action.TButton",
                   command=lambda: self._nudge_center(1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(arrows, text="▶▶", style="Action.TButton",
                   command=lambda: self._nudge_center(5)).pack(side=tk.LEFT, padx=2)

        self.lbl_center = ttk.Label(cross_group, text=f"准星 X: {self.center_x}", style="Status.TLabel")
        self.lbl_center.pack(anchor=tk.W, padx=8, pady=(4, 0))

        # ── 底部状态栏 ──
        self.status_bar = ttk.Label(self.root, text="就绪 — 点击「开始检测」启动 YOLO 推理",
                                    style="Status.TLabel", background="#0f0f23")
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM, ipady=4)

    # ══════════════════════════════════════════════════════
    # 摄像头
    # ══════════════════════════════════════════════════════

    def _open_camera(self):
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.lbl_cam_info.config(text="[X] 无法打开摄像头")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -5)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.center_x = PREVIEW_WIDTH // 2
        self.lbl_cam_info.config(text=f"分辨率: {actual_w}×{actual_h}  |  设备 #{CAMERA_INDEX}")
        self.lbl_center.config(text=f"准星 X: {self.center_x}")

    # ══════════════════════════════════════════════════════
    # 帧刷新
    # ══════════════════════════════════════════════════════

    def _update_frame(self):
        if not self.running:
            return

        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                t0 = time.perf_counter()

                # 缩放
                display = cv2.resize(frame, (PREVIEW_WIDTH, PREVIEW_HEIGHT))

                # 预处理 + 检测
                if self.detecting and self.model is not None:
                    display = self._run_detection(display)

                # 准星线
                self._draw_crosshair(display)

                # FPS
                t1 = time.perf_counter()
                self.fps_deque.append(1.0 / max(t1 - self.t_frame, 0.001))
                self.t_frame = t1

                # 转为 PhotoImage
                rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.canvas.itemconfig(self.canvas_image, image=imgtk)
                self._current_imgtk = imgtk  # 保持引用

                self.frame_count += 1

        # 更新右侧面板
        if self.frame_count % 5 == 0:
            self._update_panel()

        self.root.after(33, self._update_frame)  # ≈30fps

    # ══════════════════════════════════════════════════════
    # 检测
    # ══════════════════════════════════════════════════════

    def _run_detection(self, frame: np.ndarray) -> np.ndarray:
        """在帧上运行 YOLO 推理并绘制标注。"""
        # 预处理
        proc = preprocess_frame(frame)

        # 推理
        t_infer = time.perf_counter()
        results = self.model(proc, verbose=False, conf=0.15, iou=0.5)
        infer_ms = (time.perf_counter() - t_infer) * 1000
        self._last_infer_ms = infer_ms

        detections_all = []
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
                detections_all.append({
                    "cls_id": cls_id, "conf": conf, "gx": gx,
                    "x1": int(x1), "y1": int(y1),
                    "x2": int(x2), "y2": int(y2),
                })

        # 每种颜色只保留置信度最高的那一条
        detections = []
        for cid in WAVELENGTHS:
            cls_dets = [d for d in detections_all if d["cls_id"] == cid]
            if cls_dets:
                best = max(cls_dets, key=lambda d: d["conf"])
                detections.append(best)

        # 锁定判定
        seen_ids = {d["cls_id"] for d in detections}
        now = time.time()
        self.locked_id = None

        for cid in WAVELENGTHS:
            if cid in seen_ids:
                cls_dets = [d for d in detections if d["cls_id"] == cid]
                avg_gx = sum(d["gx"] for d in cls_dets) / len(cls_dets)
                offset = abs(avg_gx - self.center_x)

                if offset <= LOCK_PIXEL_THRESHOLD:
                    self.lock_counters[cid] = min(self.lock_counters[cid] + 1, LOCK_REQUIRE_FRAMES + 10)
                    self.unlock_counters[cid] = 0
                else:
                    self.unlock_counters[cid] += 1
                    if self.unlock_counters[cid] >= UNLOCK_FRAMES:
                        self.lock_counters[cid] = max(0, self.lock_counters[cid] - 1)

                if self.lock_counters[cid] == LOCK_REQUIRE_FRAMES:
                    if now - self.last_spoken[cid] > VOICE_COOLDOWN:
                        speak(f"{WAVELENGTHS[cid]['cn_name']}已重合")
                        self.last_spoken[cid] = now
                    self.locked_id = cid
            else:
                self.unlock_counters[cid] += 1
                if self.unlock_counters[cid] >= UNLOCK_FRAMES:
                    self.lock_counters[cid] = max(0, self.lock_counters[cid] - 1)

        # 绘制检测框
        for det in detections:
            info = WAVELENGTHS[det["cls_id"]]
            # BGR color
            hex_c = info["color_hex"].lstrip("#")
            bgr = (int(hex_c[4:6], 16), int(hex_c[2:4], 16), int(hex_c[0:2], 16))

            cv2.rectangle(frame, (det["x1"], det["y1"]), (det["x2"], det["y2"]), bgr, 2)
            cv2.putText(frame, f"{info['name']} {det['conf']:.2f}",
                        (det["x1"], det["y1"] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, bgr, 1)

            offset = det["gx"] - self.center_x
            direction = "◀◀" if offset < 0 else "▶▶"
            cv2.putText(frame, f"{direction} {abs(offset):.1f}px",
                        (int(det["gx"]) - 25, det["y1"] - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 255), 1)

        # 锁定叠加
        if self.locked_id is not None:
            info = WAVELENGTHS[self.locked_id]
            hex_c = info["color_hex"].lstrip("#")
            bgr = (int(hex_c[4:6], 16), int(hex_c[2:4], 16), int(hex_c[0:2], 16))
            put_text_cn(frame, f"🔒 {info['cn_name']} LOCKED",
                        (PREVIEW_WIDTH // 2 - 130, 50),
                        0.9, bgr, 2)

        return frame

    def _draw_crosshair(self, frame: np.ndarray):
        """绘制准星线。"""
        h = frame.shape[0]
        cv2.line(frame, (self.center_x, 0), (self.center_x, h), (255, 255, 255), 1)
        cv2.drawMarker(frame, (self.center_x, h // 2), (255, 255, 255),
                       cv2.MARKER_CROSS, 15, 1)

    # ══════════════════════════════════════════════════════
    # 面板更新
    # ══════════════════════════════════════════════════════

    def _update_panel(self):
        # FPS
        avg_fps = sum(self.fps_deque) / len(self.fps_deque) if self.fps_deque else 0
        self.lbl_fps.config(text=f"FPS: {avg_fps:.1f}")

        self.lbl_infer.config(text=f"推理: {self._last_infer_ms:.0f} ms")

        # 谱线状态
        for cid, info in WAVELENGTHS.items():
            lc = self.lock_counters[cid]
            progress = min(lc / LOCK_REQUIRE_FRAMES, 1.0)

            if progress >= 1.0:
                self.color_status_labels[cid].config(
                    text="● LOCKED", style="Green.TLabel"
                )
            elif progress > 0:
                self.color_status_labels[cid].config(
                    text=f"◐ {progress:.0%}", style="Yellow.TLabel"
                )
            else:
                self.color_status_labels[cid].config(
                    text="○ 待检测", style="Status.TLabel"
                )

        # 模型状态
        if self.model is not None:
            if self.detecting:
                self.lbl_model_info.config(text="状态: 推理中")
            else:
                self.lbl_model_info.config(text="状态: 已加载 (待启动)")
        else:
            self.lbl_model_info.config(text="状态: 未加载")

    # ══════════════════════════════════════════════════════
    # 控制
    # ══════════════════════════════════════════════════════

    def _start_detection(self):
        if self.model is None:
            self.status_bar.config(text="正在加载模型...")
            self.root.update()
            try:
                model_path = self._find_model()
                self.model = YOLO(model_path)
                _ = self.model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
                self.lbl_model_info.config(text=f"模型: {Path(model_path).parent.name}")
            except Exception as e:
                messagebox.showerror("模型加载失败", str(e))
                self.status_bar.config(text="模型加载失败")
                return

        self.detecting = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.status_bar.config(text="实时检测运行中...")

    def _stop_detection(self):
        self.detecting = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.status_bar.config(text="检测已暂停 — 点击「开始检测」恢复")

        # 重置锁定状态
        for cid in WAVELENGTHS:
            self.lock_counters[cid] = 0
            self.unlock_counters[cid] = 0

    def _find_model(self) -> str:
        for p in MODEL_CANDIDATES:
            if Path(p).exists():
                return p
        raise FileNotFoundError("未找到模型权重，请先运行 train_model.py 训练")

    def _screenshot(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                from datetime import datetime
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = self.screenshot_dir / f"grating_{ts}.png"
                cv2.imwrite(str(fname), frame)
                self.status_bar.config(text=f"📸 截图已保存: {fname}")

    def _reset_center(self):
        self.center_x = PREVIEW_WIDTH // 2
        self.lbl_center.config(text=f"准星 X: {self.center_x}")
        self.status_bar.config(text="准星已重置到画面中心")

    def _nudge_center(self, dx: int):
        self.center_x = max(10, min(PREVIEW_WIDTH - 10, self.center_x + dx))
        self.lbl_center.config(text=f"准星 X: {self.center_x}")
        self.status_bar.config(text=f"准星已偏移到 X={self.center_x}")

    def _on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        if _VOICE_OK:
            _voice_queue.put(None)
        self.root.destroy()

    # ══════════════════════════════════════════════════════
    # 数据录入辅助
    # ══════════════════════════════════════════════════════

    def _make_dms_row(self, parent: ttk.Frame, label_text: str):
        """创建一个度/分输入行，返回 (deg_entry, min_entry)。"""
        row = ttk.Frame(parent, style="Panel.TFrame")
        row.pack(fill=tk.X, padx=8, pady=2)

        lbl = ttk.Label(row, text=label_text, style="Data.TLabel", width=10, anchor=tk.E)
        lbl.pack(side=tk.LEFT, padx=(0, 4))

        deg_entry = tk.Entry(row, width=4, justify=tk.RIGHT,
                             bg="#0f0f23", fg="#e0e0e0",
                             insertbackground="#e0e0e0",
                             relief=tk.FLAT, bd=2,
                             font=("Consolas", 12))
        deg_entry.pack(side=tk.LEFT)

        ttk.Label(row, text="°", style="Data.TLabel").pack(side=tk.LEFT)

        min_entry = tk.Entry(row, width=3, justify=tk.RIGHT,
                             bg="#0f0f23", fg="#e0e0e0",
                             insertbackground="#e0e0e0",
                             relief=tk.FLAT, bd=2,
                             font=("Consolas", 12))
        min_entry.pack(side=tk.LEFT)

        ttk.Label(row, text="'", style="Data.TLabel").pack(side=tk.LEFT)

        return deg_entry, min_entry

    @staticmethod
    def _dms_to_minutes(deg_str: str, min_str: str) -> float | None:
        """将 度/分 字符串转为总角分；解析失败返回 None。"""
        try:
            d = float(deg_str.strip())
            m = float(min_str.strip())
        except ValueError:
            return None
        if m < 0 or m >= 60:
            return None
        return abs(d) * 60.0 + m

    @staticmethod
    def _minutes_to_dms_str(total_minutes: float) -> str:
        """总角分转为 ° ' 显示字符串。"""
        d = int(total_minutes // 60)
        m = total_minutes - d * 60
        return f"{d}°{m:.1f}'"

    # ══════════════════════════════════════════════════════
    # 第一组：光栅垂直检查
    # ══════════════════════════════════════════════════════

    def _check_perpendicular(self):
        """检查光栅是否垂直：| |左紫-白| - |右紫-白| | ≤ 5'。"""
        w_m = self._dms_to_minutes(
            self.entry_white_deg.get(), self.entry_white_min.get())
        pl_m = self._dms_to_minutes(
            self.entry_pl_deg.get(), self.entry_pl_min.get())
        pr_m = self._dms_to_minutes(
            self.entry_pr_deg.get(), self.entry_pr_min.get())

        if w_m is None or pl_m is None or pr_m is None:
            self.lbl_perp_result.config(
                text="[!] 请输入完整的度/分数据", foreground="#fdcb6e")
            return

        # 计算 | |左-白| - |右-白| |
        diff_left = abs(pl_m - w_m)
        diff_right = abs(pr_m - w_m)
        delta = abs(diff_left - diff_right)

        if delta <= 5.0:
            self.lbl_perp_result.config(
                text=f"[OK] 光栅已垂直  (差值 {delta:.1f}' ≤ 5')",
                foreground="#00b894")
        else:
            self.lbl_perp_result.config(
                text=f"[X] 光栅未垂直，请重新调节  (差值 {delta:.1f}' > 5')",
                foreground="#e94560")

    # ══════════════════════════════════════════════════════
    # 第二组：光栅常数计算
    # ══════════════════════════════════════════════════════

    def _calc_grating(self):
        """
        计算光栅常数 d 及不确定度 u_d。

        左右各两组游标读数，先取平均再计算：
            θ_L = (游标1_L + 游标2_L) / 2
            θ_R = (游标1_R + 游标2_R) / 2
            θ   = |θ_R − θ_L| / 2              （衍射角）
            d   = λ / sin(θ)                  （光栅常数，λ = 546.1 nm）
            u_d = d · √[ (u_λ/λ)² + (cot(θ) · u_θ)² ]

        其中 u_λ = 0.1 nm,  u_θ = 1' = π/10800 rad
        """
        LAMBDA = 546.1          # nm, 绿光波长
        U_LAMBDA = 0.1           # nm, 波长不确定度
        U_THETA_RAD = math.pi / (180.0 * 60.0)  # 1' → rad

        # 读取四组读数
        gl1_m = self._dms_to_minutes(
            self.entry_gl1_deg.get(), self.entry_gl1_min.get())
        gl2_m = self._dms_to_minutes(
            self.entry_gl2_deg.get(), self.entry_gl2_min.get())
        gr1_m = self._dms_to_minutes(
            self.entry_gr1_deg.get(), self.entry_gr1_min.get())
        gr2_m = self._dms_to_minutes(
            self.entry_gr2_deg.get(), self.entry_gr2_min.get())

        if gl1_m is None or gl2_m is None or gr1_m is None or gr2_m is None:
            self.lbl_d_result.config(
                text="[!] 请输入完整的度/分数据", foreground="#fdcb6e")
            self.lbl_theta_result.config(text="")
            return

        # 左右各取双游标平均
        gl_avg = (gl1_m + gl2_m) / 2.0
        gr_avg = (gr1_m + gr2_m) / 2.0

        if abs(gr_avg - gl_avg) < 0.01:
            self.lbl_d_result.config(
                text="[!] 左右示数不能相同", foreground="#fdcb6e")
            self.lbl_theta_result.config(text="")
            return

        # 衍射角
        theta_min = abs(gr_avg - gl_avg) / 2.0
        theta_deg = theta_min / 60.0
        theta_rad = math.radians(theta_deg)

        sin_theta = math.sin(theta_rad)
        if sin_theta < 1e-6:
            self.lbl_d_result.config(
                text="[!] 衍射角过小，无法计算", foreground="#fdcb6e")
            self.lbl_theta_result.config(text="")
            return

        cot_theta = math.cos(theta_rad) / sin_theta

        # 光栅常数 d
        d_nm = LAMBDA / sin_theta

        # 不确定度 u_d
        u_d_nm = d_nm * math.sqrt(
            (U_LAMBDA / LAMBDA) ** 2 +
            (cot_theta * U_THETA_RAD) ** 2
        )

        self.lbl_d_result.config(
            text=f"光栅常数 d = {d_nm:.1f} ± {u_d_nm:.1f} nm",
            foreground="#00b894")
        self.lbl_theta_result.config(
            text=f"衍射角 θ = {theta_deg:.4f}° ({theta_min:.2f}')",
            foreground="#c0c0d0")

    # ══════════════════════════════════════════════════════
    # 主循环
    # ══════════════════════════════════════════════════════

    def run(self):
        self.root.mainloop()


# ══════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════

def main():
    app = GratingApp()
    app.run()


if __name__ == "__main__":
    main()
