import os
import cv2
from tqdm import tqdm
def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def merge_yuv_channels(gray_img_path, color_img_path, output_path=None, resize=True):
    gray_img = cv2.imread(gray_img_path, cv2.IMREAD_GRAYSCALE)
    color_img = cv2.imread(color_img_path, cv2.IMREAD_COLOR)

    if gray_img is None:
        raise ValueError(f"无法读取灰度图像: {gray_img_path}")
    if color_img is None:
        raise ValueError(f"无法读取彩色图像: {color_img_path}")


    if resize:
        gray_h, gray_w = gray_img.shape[:2]
        color_img = cv2.resize(color_img, (gray_w, gray_h))

    #Convert the color image to the YUV color space.
    yuv_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv_img)

    #Replace the Y channel with a grayscale image.
    merged_yuv = cv2.merge([gray_img, u, v])

    # to RGB
    merged_bgr = cv2.cvtColor(merged_yuv, cv2.COLOR_YUV2BGR)

    # result
    if output_path:
        cv2.imwrite(output_path, merged_bgr)
        return output_path
    else:
        return merged_bgr


def process_image_folders(gray_dir, color_dir, output_dir, resize=True, suffix="_merged"):
    ensure_dir(output_dir)
    gray_files = {os.path.splitext(f)[0]: f for f in os.listdir(gray_dir)
                  if os.path.isfile(os.path.join(gray_dir, f))}
    color_files = {os.path.splitext(f)[0]: f for f in os.listdir(color_dir)
                   if os.path.isfile(os.path.join(color_dir, f))}


    common_names = set(gray_files.keys()) & set(color_files.keys())

    if not common_names:
        print("警告: 灰度图像文件夹和彩色图像文件夹中没有找到匹配的文件名!")
        return

    print(f"找到 {len(common_names)} 对匹配的图像")

    for name in tqdm(common_names, desc="处理图像"):
        gray_path = os.path.join(gray_dir, gray_files[name])
        color_path = os.path.join(color_dir, color_files[name])

        base_name, ext = os.path.splitext(gray_files[name])
        output_name = f"{base_name}{suffix}{ext}"
        output_path = os.path.join(output_dir, output_name)

        try:
            merge_yuv_channels(gray_path, color_path, output_path, resize)
        except Exception as e:
            print(f"处理图像 {name} 时出错: {e}")


if __name__ == "__main__":

    GRAY_DIR = r"\fusion_out\Y\SPECT_MRI"  # 灰度图像文件夹路径

    COLOR_DIR = r"datasets\test_datasets\SPECT_MRI\SPECT"
    OUTPUT_DIR = r"\fusion_out\YUV\SPECT_MRI"
    RESIZE = True  # 是否调整图像大小以匹配
    SUFFIX = "" # 输出文件名后缀

    # 处理图像文件夹
    process_image_folders(
        gray_dir=GRAY_DIR,
        color_dir=COLOR_DIR,
        output_dir=OUTPUT_DIR,
        resize=RESIZE,
        suffix=SUFFIX
    )