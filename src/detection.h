#ifndef DETECTION_H
#define DETECTION_H

#include <opencv2/opencv.hpp>
#include <string>
#include <fstream>

// 瞳孔检测功能
void detect_pupil(const std::string& light_image_path, const std::string& dark_image_path, const std::string& output_file);

// 反射点检测功能
void detect_reflection(const std::string& image_path, const std::string& output_file);

#endif
