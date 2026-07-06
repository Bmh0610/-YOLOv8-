import cv2
import os
import glob

def extract_frames_from_all_videos(interval=15):
    # 1. 设置好路径
    current_dir = os.getcwd()
    video_dir = os.path.join(current_dir, "data", "videos")
    output_folder = os.path.join(current_dir, "data", "images")
    
    # 2. 检查视频文件夹是否存在
    if not os.path.exists(video_dir):
        print(f"❌ 找不到视频文件夹: {video_dir}")
        print("💡 请确认你是否已经运行过录制程序并生成了视频。")
        return
    
    # 3. 创建保存照片的文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"📁 已创建照片保存文件夹: {output_folder}")
    
    # 4. 自动搜寻文件夹里所有的 mp4 视频
    video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
    if len(video_files) == 0:
        print(f"⚠️ 视频文件夹 {video_dir} 中没有任何 .mp4 文件！")
        return
    
    print(f"🔍 找到了 {len(video_files)} 个视频文件，准备开始批量抽帧...")
    
    # 5. 计算当前 images 文件夹里已经有多少张照片了，防止名字覆盖
    existing_images = glob.glob(os.path.join(output_folder, "*.jpg"))
    saved_count = len(existing_images)
    total_extracted_this_time = 0
    
    # 6. 挨个处理每一个视频
    for video_path in video_files:
        video_name = os.path.basename(video_path)
        # 创建一个标记文件路径，用于记录已处理的视频
        processed_marker = os.path.join(video_dir, os.path.splitext(video_name)[0] + ".processed")
        
        # 检查是否已经处理过这个视频
        if os.path.exists(processed_marker):
            print(f"⏭️ 视频 {video_name} 已经被处理过，跳过。")
            continue
        
        print(f"\n▶️ 正在处理视频: {video_name}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 无法打开视频 {video_name}，已跳过。")
            continue
        
        frame_count = 0
        video_frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # 这个视频读完了
            
            # 达到间隔帧数，保存一张照片
            if frame_count % interval == 0:
                # 名字会一直累加，比如 grating_00001.jpg, grating_00002.jpg
                filename = os.path.join(output_folder, f"grating_{saved_count:05d}.jpg")
                cv2.imwrite(filename, frame)
                saved_count += 1
                total_extracted_this_time += 1
                video_frame_count += 1
            
            frame_count += 1
        
        cap.release()
        
        # 标记这个视频已经处理完成
        with open(processed_marker, 'w') as f:
            f.write(f"Processed at {os.path.basename(__file__)}\n")
            f.write(f"Total frames extracted: {video_frame_count}\n")
            f.write(f"Frame interval: {interval}\n")
        
        print(f"✅ 视频 {video_name} 处理完毕！共提取 {video_frame_count} 张帧。")
    
    print("\n======================================")
    print(f"🎉 批量抽帧全部完成！")
    print(f"📸 本次共提取了 {total_extracted_this_time} 张照片。")
    print(f"📂 目前总计有 {saved_count} 张照片存放在: {output_folder}")
    print("======================================")

if __name__ == "__main__":
    # 抽帧间隔：对于 30fps 视频，15 意味着每秒抽 2 张
    # 如果你录视频时移动得很慢，可以把 15 改成 30（每秒1张），减少废片
    extract_frames_from_all_videos(interval=15)