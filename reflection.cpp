// reflection.cpp
#include "detection.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <iomanip> 
// 加载图像
cv::Mat load_image(const std::string& file_path) {
    cv::Mat image = cv::imread(file_path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Unable to load image: " << file_path << std::endl;
    }
    return image;
}

// 检测眼睛
std::vector<cv::Rect> detect_eyes(const cv::Mat& image) {
    std::vector<cv::Rect> eyes;
    cv::CascadeClassifier eye_cascade;
    if (!eye_cascade.load("haarcascades/haarcascade_eye.xml")) {
        std::cerr << "Error: Unable to load eye cascade classifier!" << std::endl;
        return eyes;
    }
    eye_cascade.detectMultiScale(image, eyes, 1.1, 4, 0, cv::Size(30, 30));
    return eyes;
}

// 对眼睛区域进行处理（最大值滤波 + 中值滤波）
cv::Mat process_eye_area(const cv::Mat& eye, int max_filter_size, int median_filter_size) {
    // 确保核大小为奇数
    max_filter_size = max_filter_size % 2 == 0 ? max_filter_size + 1 : max_filter_size;
    median_filter_size = median_filter_size % 2 == 0 ? median_filter_size + 1 : median_filter_size;

    // 最大值滤波
    cv::Mat max_filtered;
    cv::dilate(eye, max_filtered, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(max_filter_size, max_filter_size)));

    // 中值滤波
    cv::Mat median_filtered;
    cv::medianBlur(eye, median_filtered, median_filter_size);

    // 计算最大值滤波结果减去中值滤波结果
    cv::Mat result;
    cv::subtract(max_filtered, median_filtered, result);

    return result;
}

// 提取亮斑中心坐标并绘制
void extract_bright_spot_center(const cv::Mat& result, const cv::Rect& eye_rect, cv::Mat& original, std::ofstream& outFile) {
    // 阈值分割提取亮斑区域
    cv::Mat binary;
    cv::threshold(result, binary, 50, 255, cv::THRESH_BINARY); // 调整阈值以适应亮斑

    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty()) {
        std::cerr << "No bright spot detected!" << std::endl;
        return;
    }

    // 假设最大的轮廓是亮斑
    auto largest_contour = *std::max_element(contours.begin(), contours.end(), [](const auto& a, const auto& b) {
        return cv::contourArea(a) < cv::contourArea(b);
        });

    // 计算质心
    cv::Moments m = cv::moments(largest_contour);
    if (m.m00 == 0) {
        std::cerr << "No bright spot detected!" << std::endl;
        return;
    }

    int cx = static_cast<int>(m.m10 / m.m00);
    int cy = static_cast<int>(m.m01 / m.m00);

    // 将亮斑中心绘制在原图上
    cv::circle(original, cv::Point(eye_rect.x + cx, eye_rect.y + cy), 3, cv::Scalar(0, 0, 255), -1); // 红色圆点
    outFile << std::fixed << std::setprecision(3);  // 设置小数点后3位
    // 写入文件
    outFile << "Bright Spot Center: (" << (eye_rect.x + cx) << ", " << (eye_rect.y + cy) << ")" << std::endl;
}

// 反射点检测功能实现
void detect_reflection(const std::string& image_path, const std::string& output_file) {
    // 加载图像
    cv::Mat image = load_image(image_path);
    if (image.empty()) {
        return;
    }

    // 检测眼睛
    std::vector<cv::Rect> eyes = detect_eyes(image);
    if (eyes.size() != 2) {
        std::cerr << "Error: Exactly two eyes are required for this operation!" << std::endl;
        return;
    }

    // 按x坐标排序，确保左眼在前，右眼在后
    std::sort(eyes.begin(), eyes.end(), [](const cv::Rect& a, const cv::Rect& b) {
        return a.x < b.x;
        });

    // 提取左眼和右眼区域
    cv::Mat left_eye = image(eyes[0]);
    cv::Mat right_eye = image(eyes[1]);

    // 处理眼睛区域
    cv::Mat result_left = process_eye_area(left_eye, 5, 3);
    cv::Mat result_right = process_eye_area(right_eye, 5, 3);

    // 打开输出文件
    std::ofstream outFile(output_file);
    if (!outFile.is_open()) {
        std::cerr << "Error: Unable to open output file!" << std::endl;
        return;
    }

    // 提取左眼和右眼的亮斑中心并绘制
    extract_bright_spot_center(result_left, eyes[0], image, outFile);
    extract_bright_spot_center(result_right, eyes[1], image, outFile);

    // 关闭文件
    outFile.close();

    // 显示结果
    cv::imshow("Original Image with Bright Spot Centers", image);
    cv::imshow("Processed Left Eye", result_left);
    cv::imshow("Processed Right Eye", result_right);

    // 等待按键
    cv::waitKey(0);
}
