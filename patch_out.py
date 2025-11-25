import os
import cv2
import numpy as np
import argparse
from PIL import Image

def merge_images_by_name(input_dir, output_image_path, tile_width, tile_height, full_width, full_height, start_tile, end_tile):
    """
    从指定范围的小图（文件命名格式为 pred_X）重新拼接成大图。
    使用 OpenCV 读取和拼接图像。
    """
    # 构造文件名列表
    selected_files = []
    for i in range(start_tile, end_tile + 1):
        file_name = f"pred_{i}.png"  # 文件命名格式
        file_path = os.path.join(input_dir, file_name)
        if os.path.exists(file_path):
            selected_files.append(file_path)
        else:
            print(f"文件 {file_name} 不存在，跳过。")

    # 创建一个空白的大图，初始化为零（黑色）
    output_image = np.zeros((full_height, full_width, 3), dtype=np.uint8)

    count = 0
    for y in range(full_height // tile_height):
        for x in range(full_width // tile_width):
            if count >= len(selected_files):
                break

            tile_img = cv2.imread(selected_files[count])  # 使用 OpenCV 读取小图
            if tile_img is None:
                print(f"无法读取图像文件：{selected_files[count]}")
                continue

            # 计算拼接位置
            left = x * tile_width
            upper = y * tile_height
            output_image[upper:upper + tile_height, left:left + tile_width] = tile_img  # 拼接小图
            count += 1

    # 保存大图（使用 OpenCV 保存）
    cv2.imwrite(output_image_path, output_image)
    print(f"拼接完成，大图保存为 {output_image_path}")

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Merge images by name.")
    parser.add_argument('--image_name', type=str, default='', help="The image name (e.g. 'kodim28').")
    parser.add_argument('--start_tile', type=int, default='0', help="The start tile index.")
    parser.add_argument('--end_tile', type=int, default='83', help="The end tile index.")
    args = parser.parse_args()
    return args

# 主程序
if __name__ == "__main__":
    args = parse_args()

    # 拼接参数
    tile_width = 1280
    tile_height = 720
    # 大图尺寸
    full_width = 15360
    full_height = 5120

    # 根据传入的 image_name 确定输入目录
    input_dir = f""

    # 输出大图的路径
    output_image_path = f"{args.image_name}_merged_32.png"

    # 重新拼接图像
    merge_images_by_name(input_dir, output_image_path, tile_width, tile_height, full_width, full_height, start_tile=args.start_tile, end_tile=args.end_tile)

    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(output_image_path)
    image.show()

