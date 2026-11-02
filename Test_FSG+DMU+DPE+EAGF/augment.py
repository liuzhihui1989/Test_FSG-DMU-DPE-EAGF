import os
import random
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms.functional as tf
from torchvision import transforms

# ====== 自定义 CutMix（只处理图片） ======
def cut_mix(img1, img2):
    """
    简化版 CutMix，不处理标签
    :param img1: PIL.Image
    :param img2: PIL.Image
    :return: PIL.Image
    """
    random_gen = random.Random()

    img1_np = np.array(img1)
    img2_np = np.array(img2)

    h1, w1, _ = img1_np.shape
    h2, w2, _ = img2_np.shape

    # 随机矩形位置和大小
    rand_x = random_gen.random() * 0.75
    rand_y = random_gen.random() * 0.75
    rand_w = random_gen.random() * 0.5
    rand_h = random_gen.random() * 0.5

    cx_1 = int(rand_x * w1)
    cy_1 = int(rand_y * h1)
    cw_1 = int((rand_w + 0.25) * w1)
    ch_1 = int((rand_h + 0.25) * h1)

    cx_2 = int(rand_x * w2)
    cy_2 = int(rand_y * h2)
    cw_2 = int((rand_w + 0.25) * w2)
    ch_2 = int((rand_h + 0.25) * h2)

    if cy_1 + ch_1 > h1: ch_1 = h1 - cy_1
    if cx_1 + cw_1 > w1: cw_1 = w1 - cx_1

    # 裁剪参考图区域
    cutout_img = img2_np[cy_2:cy_2 + ch_2, cx_2:cx_2 + cw_2]

    # 调整到目标大小
    cutout_img = cv2.resize(cutout_img, (cw_1, ch_1))

    # 替换
    img1_np[cy_1:cy_1 + ch_1, cx_1:cx_1 + cw_1] = cutout_img

    return Image.fromarray(img1_np.astype(np.uint8))

# ===== 配置 =====
img_path_1 = "test1.tif"  # 主图
img_path_2 = "test2.tif"  # 参考图（如果没有可以和主图相同）
save_dir = "aug_steps"    # 保存结果目录
os.makedirs(save_dir, exist_ok=True)

# 读取原图
image = Image.open(img_path_1).convert('RGB')
image.save(os.path.join(save_dir, "00_original.jpg"))

# Step 1: CutMix
image_refer = Image.open(img_path_2).convert('RGB')
image = cut_mix(image, image_refer)
image.save(os.path.join(save_dir, "01_cutmix.jpg"))

# Step 2: 随机缩放（固定成系数1.2）
rand_h = 1.5
rand_w = 1.5
resize_h = int(image.height * rand_h)
resize_w = int(image.width * rand_w)
image = tf.resize(image, [resize_h, resize_w])
image.save(os.path.join(save_dir, "02_resize.jpg"))

# Step 3: 随机裁剪（固定成288×288）
i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(288, 288))
image = tf.crop(image, i, j, h, w)
image.save(os.path.join(save_dir, "03_crop.jpg"))

# Step 4: 水平翻转
image = tf.hflip(image)
image.save(os.path.join(save_dir, "04_hflip.jpg"))

# Step 5: Color Jitter
jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
image = jitter(image)
image.save(os.path.join(save_dir, "05_color_jitter.jpg"))

# Step 6: Gaussian Blur（kernel=5）
blur = transforms.GaussianBlur(kernel_size=5)
image = blur(image)
image.save(os.path.join(save_dir, "06_blur.jpg"))

# Step 7: Perspective 变换
start_p, end_p = transforms.RandomPerspective.get_params(image.width, image.height, distortion_scale=0.5)
image = tf.perspective(image, start_p, end_p)
image.save(os.path.join(save_dir, "07_perspective.jpg"))

print(f"所有增强完成，结果已保存到：{save_dir}")
