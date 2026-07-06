"""
生成三条谱线锁定的语音提示音频
使用pyttsx3库生成中文语音
"""

import pyttsx3
import os
from pathlib import Path

def generate_single_audio(engine, filename, text, output_dir):
    """生成单个音频文件，每次重新初始化引擎"""
    try:
        # 重新初始化引擎，避免状态污染
        new_engine = pyttsx3.init()
        
        # 设置中文语音
        voices = new_engine.getProperty('voices')
        chinese_voice = None
        for voice in voices:
            if 'chinese' in voice.name.lower() or 'zh' in voice.id.lower():
                chinese_voice = voice.id
                break
        
        if chinese_voice:
            new_engine.setProperty('voice', chinese_voice)
        
        # 设置参数
        new_engine.setProperty('rate', 150)
        new_engine.setProperty('volume', 0.9)
        
        output_path = output_dir / filename
        print(f"🔊 生成: {text}")
        print(f"   保存到: {output_path}")
        
        # 删除已存在的文件
        if output_path.exists():
            output_path.unlink()
        
        new_engine.save_to_file(text, str(output_path))
        new_engine.runAndWait()
        
        # 验证文件是否生成
        if output_path.exists():
            size_kb = output_path.stat().st_size / 1024
            print(f"   ✅ 生成成功 ({size_kb:.1f} KB)")
            return True
        else:
            print(f"   ❌ 生成失败 - 文件未创建")
            return False
            
    except Exception as e:
        print(f"   ❌ 生成失败 - 错误: {e}")
        return False
    finally:
        try:
            new_engine.stop()
        except:
            pass

def generate_voice_audios():
    """生成三句语音锁定提示音频"""
    
    # 输出目录
    output_dir = Path("E:/grating_yolo/audio1")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 三句语音内容
    voice_texts = {
        "purple_locked.wav": "紫色谱线已锁定",
        "green_locked.wav": "绿色谱线已锁定",
        "yellow_locked.wav": "黄色谱线已锁定",
    }
    
    print("=" * 50)
    print("🎙️  开始生成语音音频文件...")
    print("=" * 50)
    
    # 初始化引擎一次，用于获取语音信息
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    chinese_voice = None
    for voice in voices:
        if 'chinese' in voice.name.lower() or 'zh' in voice.id.lower():
            chinese_voice = voice.id
            break
    
    if chinese_voice:
        print(f"✅ 使用中文语音: {chinese_voice}")
    else:
        print("⚠️ 未找到中文语音，使用默认语音")
    
    engine.stop()
    
    print()
    
    # 逐个生成音频文件，每次使用新引擎
    success_count = 0
    for filename, text in voice_texts.items():
        if generate_single_audio(engine, filename, text, output_dir):
            success_count += 1
        print()
    
    print("=" * 50)
    if success_count == len(voice_texts):
        print(f"🎉 全部完成！音频文件保存在: {output_dir}")
    else:
        print(f"⚠️  部分完成！成功 {success_count}/{len(voice_texts)} 个文件")
    print("=" * 50)
    
    # 列出生成的文件
    print("\n📁 生成的文件列表:")
    for filename in voice_texts.keys():
        filepath = output_dir / filename
        if filepath.exists():
            size_kb = filepath.stat().st_size / 1024
            print(f"   ✓ {filename} ({size_kb:.1f} KB)")
        else:
            print(f"   ✗ {filename} (未生成)")

if __name__ == "__main__":
    try:
        generate_voice_audios()
    except Exception as e:
        print(f"❌ 错误: {e}")
        print("\n💡 提示:")
        print("   1. 请确保已安装 pyttsx3: pip install pyttsx3")
        print("   2. Windows 系统自带 TTS 引擎，无需额外安装")
        print("   3. 如果需要更自然的语音，可安装第三方 TTS 引擎")