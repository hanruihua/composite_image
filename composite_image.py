"""
    File: composite_image.py
    Author: renyunfan
    Email: renyf@connect.hku.hk
    Description: [ A python script to create a composite image from a video.]
    All Rights Reserved 2023
"""

import cv2
import numpy as np
from enum import Enum
import argparse


class CompositeMode(Enum):
    MAX_VARIATION = 0
    MIN_VALUE = 1
    MAX_VALUE = 2


class CompositeImage:

    def __init__(self, mode, video_path, start_t=0, end_t=999, skip_frame=1, transparent=0.5, apply_blur=True, extract_foreground=False):
        self.video_path = video_path
        self.skip_frame = skip_frame
        self.start_t = start_t
        self.end_t = end_t
        self.mode = mode
        self.transparent = transparent
        self.apply_blur = apply_blur  # 是否应用高斯模糊
        self.extract_foreground = extract_foreground  # 是否提取前景动态物体

        if self.extract_foreground:
            # 初始化前景提取器
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

    def max_variation_update(self, image, cnt=0):
        delta_img = image - self.ave_img
        image_norm = np.linalg.norm(image, axis=2)
        delta_norm = image_norm - self.ave_img_norm
        abs_delta_norm = np.abs(delta_norm)
        delta_mask = abs_delta_norm > self.abs_diff_norm
        diff_mask = abs_delta_norm <= self.abs_diff_norm

        # 使用软阈值代替硬阈值
        alpha = 10  # Sigmoid 函数的斜率参数，可以根据需要调整
        delta_mask = 1 / (1 + np.exp(-alpha * (abs_delta_norm - self.abs_diff_norm)))
        diff_mask = 1 - delta_mask

        delta_mask = np.stack((delta_mask, delta_mask, delta_mask), axis=2)
        diff_mask = np.stack((diff_mask, diff_mask, diff_mask), axis=2)

        if cnt == self.img_num:
            self.diff_img = self.transparent * self.diff_img * diff_mask + delta_img * delta_mask
        else:
            self.diff_img = self.diff_img * diff_mask + delta_img * delta_mask

        self.diff_norm = np.linalg.norm(self.diff_img, axis=2)
        self.abs_diff_norm = np.abs(self.diff_norm)

    def min_value_update(self, image, cnt=0):
        image_norm = np.linalg.norm(image, axis=2)
        cur_min_image = self.diff_img + self.ave_img
        cur_min_image_norm = np.linalg.norm(cur_min_image, axis=2)
        delta_mask = cur_min_image_norm > image_norm
        min_mask = cur_min_image_norm <= image_norm
        delta_mask = np.stack((delta_mask.T, delta_mask.T, delta_mask.T)).T.astype(np.float32)
        min_mask = np.stack((min_mask.T, min_mask.T, min_mask.T)).T.astype(np.float32)

        if cnt == self.img_num:
            new_min_img = image * delta_mask + self.transparent * min_mask * cur_min_image
        else:
            new_min_img = image * delta_mask + min_mask * cur_min_image

        self.diff_img = new_min_img - self.ave_img

    def max_value_update(self, image, cnt=0):
        image_norm = np.linalg.norm(image, axis=2)
        cur_min_image = self.diff_img + self.ave_img
        cur_min_image_norm = np.linalg.norm(cur_min_image, axis=2)
        delta_mask = cur_min_image_norm < image_norm
        min_mask = cur_min_image_norm >= image_norm
        delta_mask = np.stack((delta_mask, delta_mask, delta_mask), axis=2).astype(np.float32)
        min_mask = np.stack((min_mask, min_mask, min_mask), axis=2).astype(np.float32)
        new_min_img = image * delta_mask + min_mask * cur_min_image
        self.diff_img = new_min_img - self.ave_img

    def extract_frames(self):
        # 打开视频文件
        if self.video_path is None:
            return None
        video = cv2.VideoCapture(self.video_path)
        if not video.isOpened():
            print(f"Error: 无法打开视频文件 {self.video_path}")
            return None

        # 获取视频帧率
        fps = video.get(cv2.CAP_PROP_FPS)
        # 计算开始和结束帧的索引
        start_frame = int(self.start_t * fps)
        end_frame = int(self.end_t * fps)

        # 设置视频的当前帧为开始帧
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_count = 0  # 记录提取的帧数

        imgs = []
        # 循环读取视频帧
        while video.isOpened() and frame_count <= (end_frame - start_frame):
            ret, frame = video.read()

            if ret:
                frame_count += 1
                if (frame_count % self.skip_frame != 0):
                    continue
                # 添加当前帧到图片列表
                imgs.append(frame)

                if frame_count > (end_frame - start_frame):
                    break
            else:
                break

        # 释放视频对象
        video.release()

        return imgs

    def merge_images(self):
        image_files = self.extract_frames()
        if image_files is None or len(image_files) < 1:
            print("Error: 未提取到任何帧，请检查输入视频路径：", self.video_path)
            exit(1)

        first_image = image_files[0].astype(np.float32)
        height, width, _ = first_image.shape

        sum_image = np.zeros((height, width, 3), dtype=np.float32)
        self.img_num = len(image_files)

        for idx, image_file in enumerate(image_files):
            image = image_file.astype(np.float32)
            sum_image += image

        self.ave_img = sum_image / self.img_num
        self.ave_img_norm = np.linalg.norm(self.ave_img, axis=2)
        self.diff_norm = np.zeros((height, width), dtype=np.float32)
        self.abs_diff_norm = np.zeros((height, width), dtype=np.float32)
        self.diff_img = np.zeros((height, width, 3), dtype=np.float32)

        cnt = 0
        for image_file in image_files:
            cnt += 1
            print(f"Processing {cnt} / {self.img_num}")
            image = image_file.astype(np.float32)
            if self.mode == CompositeMode.MAX_VARIATION:
                self.max_variation_update(image, cnt)
            elif self.mode == CompositeMode.MIN_VALUE:
                self.min_value_update(image, cnt)
            elif self.mode == CompositeMode.MAX_VALUE:
                self.max_value_update(image, cnt)

        merged_image = self.ave_img + self.diff_img
        merged_image = np.clip(merged_image, 0, 255).astype(np.uint8)

        # 如果启用了前景提取，叠加动态物体
        if self.extract_foreground:
            print("启用前景提取，处理动态物体...")
            foreground_image = self.extract_foreground_objects(image_files)
            # 将前景物体叠加到背景上
            merged_image = cv2.addWeighted(merged_image, 1.0, foreground_image, self.transparent, 0)

        # 应用高斯模糊以软化重叠区域
        if self.apply_blur:
            merged_image = cv2.GaussianBlur(merged_image, (5, 5), 0)

        return merged_image

    def extract_foreground_objects(self, image_files):
        """
        提取前景（动态物体），并生成前景图像。
        """
        # 使用背景建模器逐帧提取前景
        foreground_sum = np.zeros_like(image_files[0], dtype=np.float32)
        count = 0

        for idx, frame in enumerate(image_files):
            fg_mask = self.bg_subtractor.apply(frame)
            # 去除阴影部分
            _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
            fg_mask = fg_mask.astype(np.float32) / 255.0
            fg_mask = cv2.merge([fg_mask, fg_mask, fg_mask])

            # 叠加前景
            foreground_sum += frame.astype(np.float32) * fg_mask
            count += 1

        # 计算平均前景
        if count > 0:
            foreground_avg = foreground_sum / count
        else:
            foreground_avg = foreground_sum

        # 转换为 uint8 类型
        foreground_avg = np.clip(foreground_avg, 0, 255).astype(np.uint8)
        return foreground_avg


def main():
    parser = argparse.ArgumentParser(
        prog='CompositeImage',
        description='Convert video to composite image.',
        epilog='-')
    parser.add_argument('-p', '--video_path', type=str, default='./03_limo_327.mp4',
                        help='path of input video file.')
    parser.add_argument('-m', '--mode', default='VAR', choices=['VAR', 'MAX', 'MIN'],
                        help='mode of composite image.')
    parser.add_argument('-s', '--start_t', default=5, type=float,
                        help='start time of composite image.')
    parser.add_argument('-e', '--end_t', default=10, type=float,
                        help='end time of composite image.')
    parser.add_argument('-k', '--skip_frame', default=25, type=int,
                        help='skip frame when extract frames.')
    parser.add_argument('-o', '--output_name', default='composite_image.png', type=str,
                        help='the output file name.')
    parser.add_argument('-t', '--transparent', default=0.5, type=float,
                        help='the transparency of the image.')
    parser.add_argument('--blur', action='store_true', default=True,
                        help='whether to apply Gaussian blur to the final image.')
    parser.add_argument('-x', '--extract_foreground', action='store_true', default=False,
                        help='whether to extract and overlay dynamic objects.')

    args = parser.parse_args()

    # 读取命令行参数
    path = args.video_path
    mode = args.mode
    start_t = args.start_t
    end_t = args.end_t
    skip_frame = args.skip_frame
    output_name = args.output_name
    transparent = args.transparent
    apply_blur = args.blur
    extract_foreground = args.extract_foreground

    print(" -- Load Param: video path", path)
    print(" -- Load Param: mode", mode)
    print(" -- Load Param: start_t", start_t)
    print(" -- Load Param: end_t", end_t)
    print(" -- Load Param: skip_frame", skip_frame)
    print(" -- Load Param: output_name", output_name)
    print(" -- Load Param: transparent", transparent)
    print(" -- Load Param: apply_blur", apply_blur)
    print(" -- Load Param: extract_foreground", extract_foreground)

    if mode == 'MAX':
        mode_enum = CompositeMode.MAX_VALUE
    elif mode == 'MIN':
        mode_enum = CompositeMode.MIN_VALUE
    elif mode == 'VAR':
        mode_enum = CompositeMode.MAX_VARIATION

    # 初始化合成器，传入是否提取前景的参数
    merger = CompositeImage(mode_enum, path, start_t, end_t, skip_frame, transparent, apply_blur, extract_foreground)
    merged_image = merger.merge_images()
    cv2.imwrite(output_name, merged_image)
    print(f"Composite image saved as {output_name}")


if __name__ == '__main__':
    main()