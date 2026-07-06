"""
简单的OpenCV打开USB摄像头1
"""

import cv2

def open_usb_camera():
    """打开USB摄像头1并显示画面"""
    
    # 打开USB摄像头1
    cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("无法打开USB摄像头1，尝试其他索引...")
        # 尝试其他摄像头索引
        for idx in range(4):
            if idx != 1:
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    print(f"成功打开摄像头 {idx}")
                    break
        else:
            print("无法打开任何摄像头！")
            return
    
    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 手动曝光模式
    cap.set(cv2.CAP_PROP_EXPOSURE, -5)  # 降低曝光值（数值越小，曝光越低）
    
    # 获取实际参数
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    print(f"USB摄像头已打开: {width}x{height} @ {fps:.1f} FPS")
    print("按 'Q' 或 'ESC' 退出")
    
    while True:
        # 读取帧
        ret, frame = cap.read()
        
        if not ret:
            print("无法读取摄像头画面！")
            break
        
        # 显示画面
        cv2.imshow("USB Camera 1", frame)
        
        # 按Q或ESC退出
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # Q 或 ESC
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("程序已退出")

if __name__ == "__main__":
    open_usb_camera()