import os
import cv2
from PIL import Image
from pathlib import Path

# ===================== 配置参数（根据实际路径修改） =====================
# 原始图片目录（lol_dataset/our485/low）
INPUT_DIR = r"D:\fyp_project\supporting_document\lol_dataset\eval15\low"
# 输出目录（resize后的图片保存路径，自动创建）
OUTPUT_DIR = r"D:\fyp_project\supporting_document\lol_dataset\eval15\low_256"
# 目标尺寸
TARGET_SIZE = (256, 256)
# 是否保留原图片的比例（False=强制拉伸为256×256，True=等比例缩放+补黑边）
KEEP_ASPECT_RATIO = False

# ===================== 核心处理函数 =====================
def resize_image(image_path, output_path, target_size=(256, 256), keep_aspect_ratio=False):
    """
    单张图片resize
    :param image_path: 输入图片路径
    :param output_path: 输出图片路径
    :param target_size: 目标尺寸 (w, h)
    :param keep_aspect_ratio: 是否保持宽高比
    """
    try:
        # 打开图片（自动处理JPG/PNG等格式）
        img = Image.open(image_path).convert("RGB")
        
        if keep_aspect_ratio:
            # 等比例缩放 + 补黑边到256×256
            img.thumbnail(target_size, Image.Resampling.LANCZOS)
            new_img = Image.new("RGB", target_size, (0, 0, 0))
            # 居中放置
            offset = ((target_size[0] - img.width) // 2, (target_size[1] - img.height) // 2)
            new_img.paste(img, offset)
            img = new_img
        else:
            # 强制拉伸为256×256
            img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # 保存图片（创建父目录）
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path, quality=95)  # 高质量保存
        print(f"✅ 处理完成: {image_path} -> {output_path}")
    except Exception as e:
        print(f"❌ 处理失败 {image_path}: {str(e)}")

# ===================== 批量执行 =====================
if __name__ == "__main__":
    # 遍历所有图片文件
    input_path = Path(INPUT_DIR)
    image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    
    for img_file in input_path.rglob("*"):
        if img_file.suffix.lower() in image_extensions:
            # 构建输出路径（保留原目录结构）
            relative_path = img_file.relative_to(input_path)
            output_file = Path(OUTPUT_DIR) / relative_path
            
            # 执行resize
            resize_image(
                str(img_file), 
                str(output_file), 
                target_size=TARGET_SIZE,
                keep_aspect_ratio=KEEP_ASPECT_RATIO
            )
    
    print("\n🎉 所有图片处理完成！输出目录：", OUTPUT_DIR)