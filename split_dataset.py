import os
import random
import shutil

def split_yolo_dataset(source_dir, split_ratio=0.8):
    """
    将混杂着图片和标签的文件夹，一键划分为 YOLO 标准格式的 train 和 val 集。
    
    参数:
        source_dir: 包含所有 .jpg 和 .txt 的原始文件夹路径
        split_ratio: 训练集所占比例 (默认 0.8，即 80% 训练，20% 验证)
    """
    # 1. 定义 YOLO 标准输出目录结构
    base_dir = os.getcwd()
    yolo_dir = os.path.join(base_dir, "dataset")
    
    dirs_to_make = [
        os.path.join(yolo_dir, "images", "train"),
        os.path.join(yolo_dir, "images", "val"),
        os.path.join(yolo_dir, "labels", "train"),
        os.path.join(yolo_dir, "labels", "val")
    ]
    
    # 创建所有需要的文件夹
    for d in dirs_to_make:
        os.makedirs(d, exist_ok=True)
    
    # 2. 收集所有成对的图片和标签
    image_files = [f for f in os.listdir(source_dir) if f.endswith('.jpg') or f.endswith('.png')]
    
    # 尝试在source_dir的同级目录labels中查找标签文件
    labels_dir = os.path.join(os.path.dirname(source_dir), "labels")
    if not os.path.exists(labels_dir):
        print(f"❌ 错误：找不到标签文件目录 {labels_dir}，请检查路径！")
        return
    
    valid_pairs = []
    for img_name in image_files:
        # 寻找同名的 .txt 标签文件
        base_name = os.path.splitext(img_name)[0]
        txt_name = base_name + ".txt"
        txt_path = os.path.join(labels_dir, txt_name)
        
        if os.path.exists(txt_path):
            valid_pairs.append(base_name)
        else:
            print(f"⚠️ 警告: 图片 {img_name} 没有找到对应的标签文件，已跳过。")

    total_valid = len(valid_pairs)
    if total_valid == 0:
        print("❌ 错误：没有找到任何匹配的图片和标签，请检查源文件夹路径！")
        return

    print(f"🔍 成功找到 {total_valid} 对匹配的图片和标签。")

    # 3. 随机打乱数据 (保证模型学到的不是固定顺序)
    random.seed(42) # 设置随机种子，保证每次划分结果一致，方便实验复现
    random.shuffle(valid_pairs)

    # 4. 计算划分界限
    train_size = int(total_valid * split_ratio)
    train_pairs = valid_pairs[:train_size]
    val_pairs = valid_pairs[train_size:]

    print(f"⚙️ 准备划分：训练集 {len(train_pairs)} 张，验证集 {len(val_pairs)} 张...")

    # 5. 开始复制文件到对应的文件夹
    def copy_files(file_pairs, subset_name):
        for base_name in file_pairs:
            # 复制图片
            img_src = os.path.join(source_dir, base_name + ".jpg") # 假设是 jpg
            img_dst = os.path.join(yolo_dir, "images", subset_name, base_name + ".jpg")
            if os.path.exists(img_src):
                shutil.copy(img_src, img_dst)
            
            # 复制标签
            txt_src = os.path.join(labels_dir, base_name + ".txt")
            txt_dst = os.path.join(yolo_dir, "labels", subset_name, base_name + ".txt")
            if os.path.exists(txt_src):
                shutil.copy(txt_src, txt_dst)

    copy_files(train_pairs, "train")
    copy_files(val_pairs, "val")

    print("\n======================================")
    print("🎉 数据集划分完成！")
    print(f"📂 标准 YOLO 数据集已生成在: {yolo_dir}")
    print("======================================")

if __name__ == "__main__":
    # ==========================================
    # 请在这里修改为你存放图片和标签的文件夹路径
    # 如果你的图片和打好的标签都在 dataset/images 里，就填 "dataset/images"
    # ==========================================
    SOURCE_FOLDER = "data/images" 
    
    split_yolo_dataset(SOURCE_FOLDER, split_ratio=0.8)