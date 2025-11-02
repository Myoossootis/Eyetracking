#ifndef DETECTION_H
#define DETECTION_H

#include <opencv2/opencv.hpp>
#include <string>
#include <fstream>

/**
 * @brief 使用明暗瞳法检测瞳孔中心。
 *
 * @param light_image_path 明瞳图像的路径（红外光与视轴同轴）。
 * @param dark_image_path 暗瞳图像的路径（红外光与视轴异轴）。
 * @param output_file 用于保存检测到的瞳孔中心坐标的文本文件路径。
 */
void detect_pupil(const std::string& light_image_path, const std::string& dark_image_path, const std::string& output_file);

/**
 * @brief 检测眼睛图像中的普尔钦斑（角膜反射光斑）中心。
 *
 * @param image_path 包含普尔钦斑的眼睛图像路径。
 * @param output_file 用于保存检测到的光斑中心坐标的文本文件路径。
 */
void detect_reflection(const std::string& image_path, const std::string& output_file);

#endif // DETECTION_H
