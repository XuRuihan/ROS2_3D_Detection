import os
import fnmatch
from PIL import Image


def create_gif_from_images():
    # 输入目录
    input_dir = "data/result"
    
    # 确保输入目录存在
    if not os.path.exists(input_dir):
        print(f"错误：目录 {input_dir} 不存在")
        return
    
    # 获取所有以"scene_"开头的文件
    scene_files = fnmatch.filter(os.listdir(input_dir), "scene_*")
    
    # 如果没有找到文件，返回
    if not scene_files:
        print(f"错误：在 {input_dir} 中没有找到以'scene_'开头的文件")
        return
    
    # 按字典序排序文件
    scene_files.sort()
    
    # 创建图像列表
    images = []
    for filename in scene_files:
        file_path = os.path.join(input_dir, filename)
        try:
            # 打开图像并转换为RGB模式（避免PNG透明问题）
            with Image.open(file_path) as img:
                img = img.convert('RGB')
                images.append(img)
        except Exception as e:
            print(f"警告：无法处理文件 {file_path} - {str(e)}")
    
    # 检查是否至少有一张有效图像
    if not images:
        print("错误：没有有效的图像可以用于创建GIF")
        return
    
    # 保存为GIF
    output_path = "scene_animation.gif"
    try:
        # 保存为GIF，第一张图作为初始帧，后续的作为动画帧
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=100,  # 每帧显示100毫秒（根据需要调整）
            loop=0         # 无限循环
        )
        print(f"GIF已成功创建：{output_path}")
    except Exception as e:
        print(f"错误：无法创建GIF - {str(e)}")


if __name__ == "__main__":
    create_gif_from_images()
