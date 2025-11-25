import os
import cv2
import numpy as np
from PIL import Image

def split_image(input_image_path, output_dir, tile_width, tile_height, start_tile=85):
    """
    将大图切分为小图并保存到指定目录。
    使用 OpenCV 来读取图像。
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 打开原始图像（使用 OpenCV）
    image = cv2.imread(input_image_path)
    img_height, img_width, _ = image.shape  # 获取图像的宽高

    # 计算切分后的图块总数
    x_tiles = img_width // tile_width
    y_tiles = img_height // tile_height

    count = start_tile  # 设置切分图像的起始编号
    for y in range(y_tiles):
        for x in range(x_tiles):
            # 切割图像
            left = x * tile_width
            upper = y * tile_height
            right = left + tile_width
            lower = upper + tile_height

            cropped_img = image[upper:lower, left:right]  # 使用 NumPy 数组切割图像

            # 保存小图像（使用 OpenCV 保存）
            tile_path = os.path.join(output_dir, f"tile_{count:04d}.png")  # 使用从 start_tile 开始的编号
            cv2.imwrite(tile_path, cropped_img)  # 保存为 PNG 格式
            print(f"Saved tile {count:04d}")

            count += 1  # 更新文件编号

    print(f"切分完成，总共生成 {count - start_tile} 张小图。")






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




# 示例用法
if __name__ == "__main__":
    # 输入大图路径
    input_image_path = "./DIV2k/0799.png"
    # 切分输出目录
    output_dir = "./tiles_2k_2"

    # 切分参数
    tile_width = 1020
    tile_height = 678
    # 大图尺寸
    full_width = 2040
    full_height = 1356

    # 切分图像
    split_image(input_image_path, output_dir, tile_width, tile_height, start_tile=1)

