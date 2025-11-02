#include "detection.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>

/**
 * @brief 对眼睛区域进行预处理，以凸显反射光斑。
 *
 * @param eye_region 眼睛区域的图像。
 * @return 经过处理后，可能包含光斑的二值图像。
 */
cv::Mat preprocess_for_reflection(const cv::Mat& eye_region) {
    cv::Mat gray, blurred, binary;
    if (eye_region.channels() == 3) {
        cv::cvtColor(eye_region, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = eye_region.clone();
    }

    // 使用高斯模糊平滑图像，为阈值化做准备
    cv::GaussianBlur(gray, blurred, cv::Size(3, 3), 0);

    // 使用一个较高的阈值来分离出明亮的反射光斑
    cv::threshold(blurred, binary, 230, 255, cv::THRESH_BINARY);

    return binary;
}

/**
 * @brief 从轮廓中找到最大的一个，并计算其中心。
 *
 * @param contours 轮廓的向量。
 * @param eye_region_offset 眼睛区域在原图中的偏移量。
 * @return 反射光斑的中心坐标。如果未找到，则返回 (-1, -1)。
 */
cv::Point2f find_largest_contour_center(const std::vector<std::vector<cv::Point>>& contours, const cv::Point& eye_region_offset) {
    if (contours.empty()) {
        return cv::Point2f(-1, -1);
    }

    // 寻找面积最大的轮廓
    double max_area = 0;
    int max_area_idx = -1;
    for (size_t i = 0; i < contours.size(); ++i) {
        double area = cv::contourArea(contours[i]);
        if (area > max_area) {
            max_area = area;
            max_area_idx = i;
        }
    }

    if (max_area_idx != -1) {
        // 计算最大轮廓的矩
        cv::Moments mu = cv::moments(contours[max_area_idx]);
        // 计算中心点，并加上眼睛区域的偏移
        return cv::Point2f(mu.m10 / mu.m00 + eye_region_offset.x, mu.m01 / mu.m00 + eye_region_offset.y);
    }

    return cv::Point2f(-1, -1);
}

void detect_reflection(const std::string& image_path, const std::string& output_file) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Error: Could not load image for reflection detection." << std::endl;
        return;
    }

    // 使用Haar级联分类器检测眼睛
    cv::CascadeClassifier eye_cascade;
    if (!eye_cascade.load("haarcascades/haarcascade_eye.xml")) {
        std::cerr << "Error: Could not load eye cascade classifier." << std::endl;
        return;
    }

    std::vector<cv::Rect> eyes;
    eye_cascade.detectMultiScale(image, eyes, 1.1, 4, 0, cv::Size(30, 30));

    if (eyes.empty()) {
        std::cerr << "Warning: No eyes detected in reflection image." << std::endl;
        return;
    }

    // 假设我们只关心第一个检测到的眼睛
    cv::Rect eye_roi = eyes[0];
    cv::Mat eye_region = image(eye_roi);

    // 预处理眼睛区域以寻找光斑
    cv::Mat binary_reflection = preprocess_for_reflection(eye_region);

    // 寻找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary_reflection, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 找到最大轮廓的中心
    cv::Point2f reflection_center = find_largest_contour_center(contours, eye_roi.tl());

    // 保存结果
    if (reflection_center.x != -1) {
        std::ofstream outfile(output_file);
        if (outfile.is_open()) {
            outfile << reflection_center.x << " " << reflection_center.y << std::endl;
            outfile.close();
        }

        // 在图像上标记光斑中心
        cv::circle(image, reflection_center, 3, cv::Scalar(0, 0, 255), -1);
        // 可选：显示结果
        // cv::imshow("Reflection Detection", image);
        // cv::waitKey(0);
    }
}
