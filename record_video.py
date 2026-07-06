import cv2
import os
import datetime

def record_usb_camera(camera_index=1):
    # 1. 获取当前目录，并设定专门存放视频的文件夹 (dataset/videos)
    current_dir = os.getcwd()
    video_dir = os.path.join(current_dir, "data", "videos")
    
    # 如果文件夹不存在，则自动创建
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
        print(f"📁 已自动创建视频存放文件夹: {video_dir}")

    # 2. 利用当前时间自动生成唯一的文件名，绝对不会覆盖之前的视频
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"grating_{current_time}.mp4"
    output_filename = os.path.join(video_dir, video_filename)

    print(f"🔄 正在连接摄像头 (编号: {camera_index})...")
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"❌ 错误：无法连接到编号为 {camera_index} 的摄像头。请检查连接。")
        return

    # 强制设置 1K (1080p) 分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # ================= 曝光控制代码 =================
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) 
    cap.set(cv2.CAP_PROP_EXPOSURE, -5) 
    # ================================================
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if fps == 0 or fps is None:
        fps = 30.0

    print(f"✅ 摄像头连接成功！当前分辨率: {width}x{height}, 帧率: {fps} FPS")

    # 设置视频编码器并创建写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    print("--------------------------------------------------")
    print(f"🎬 开始录制！")
    print(f"📁 当前视频将保存在: {output_filename}")
    print("👉 操作指南：调节好光栅和曝光后，在监控窗口上按英文 'q' 键结束录制。")
    print("💡 提示：录完这一段后，你可以再次运行本代码录制下一段，视频会自动按时间命名。")
    print("--------------------------------------------------")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 无法读取画面。")
            break

        out.write(frame)
        cv2.imshow('Recording Grating Diffraction (Press Q to stop)', frame)

        # 监听键盘，按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 收到停止指令...")
            break

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"🎉 录制完成！视频已安全保存至: {output_filename}")

if __name__ == "__main__":
    record_usb_camera()