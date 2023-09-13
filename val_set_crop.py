#!/usr/bin/env python
# coding: utf-8


# Resize the dataset
import cv2
import os

def crop_and_resize():
    """
    对图片进行裁剪和上采样

    Args:
        image_path: 图片路径
        output_path: 输出路径
    """

    input_path = "./data/img_align_celeba/img_align_celeba"
    output_root = "./data/celeba_202599"

    # 获取所有图片的路径
    image_paths = os.listdir(input_path)

    # 遍历所有图片，并进行裁剪和上采样
    for image_path in image_paths:
        output_path = os.path.join(output_root, image_path)
        # crop_and_resize(os.path.join(input_path, image_path), output_path)

        image = cv2.imread(os.path.join(input_path, image_path))
        height, width = image.shape[:2]   # 218,178

        # 裁剪图片
        crop_size = int((height - width) / 2)
        crop_image = image[crop_size:height-crop_size, :]

        # 上采样图片
        resized_image = cv2.resize(crop_image, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)

        cv2.imwrite(output_path, resized_image)


if __name__ == "__main__":
    crop_and_resize()

