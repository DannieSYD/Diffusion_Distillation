import cv2
import os


# 获取指定目录下的所有图片路径
def get_image_paths(dir_path):
    image_paths = []
    for root, dirnames, filenames in os.walk(dir_path):
        for filename in filenames:
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_paths.append(os.path.join(root, filename))
    return image_paths


# 裁剪图片
def crop_image(image_path, output_path, y0, y1, x0, x1):
    image = cv2.imread(image_path)
    crop_image = image[y0:y1, x0:x1]
    cv2.imwrite(output_path, crop_image)


if __name__ == "__main__":
    dir_path = "./images/celeba/base_0/full"
    output_dir = "./images/celeba/base_0/full"
    # 获取指定目录下的所有图片路径
    image_paths = get_image_paths(dir_path)
    # 循环遍历所有图片路径，并对每个图片进行裁剪
    for image_path in image_paths:
        # 裁剪图片
        crop_image(image_path, os.path.join(output_dir, os.path.basename(image_path)), 256*4, 256*5, 0, 256)