#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

/**
 * @brief 验证检测到的眼睛区域是否有效。
 *
 * @param eyes 检测到的眼睛矩形列表。
 * @param image_area 整个图像的面积，用于相对大小的验证。
 * @return std::vector<cv::Rect> 经过验证和排序的眼睛矩形列表。
 */
std::vector<cv::Rect> validate_and_sort_eyes(std::vector<cv::Rect>& eyes, int image_area) {
    std::vector<cv::Rect> validated_eyes;

    // 1. 按x坐标排序
    std::sort(eyes.begin(), eyes.end(), [](const cv::Rect& a, const cv::Rect& b) {
        return a.x < b.x;
    });

    // 2. 验证每个眼睛区域的有效性
    for (const auto& eye : eyes) {
        // 验证宽高比
        float aspect_ratio = static_cast<float>(eye.width) / eye.height;
        if (aspect_ratio < 0.5 || aspect_ratio > 2.0) continue;

        // 验证区域大小相对于图像
        float relative_size = static_cast<float>(eye.area()) / image_area;
        if (relative_size < 0.005 || relative_size > 0.1) continue;

        validated_eyes.push_back(eye);
    }

    // 3. 如果验证后眼睛数量少于2，则认为检测失败
    if (validated_eyes.size() < 2) {
        return {};
    }

    // 4. 验证两只眼睛的相对位置和大小
    const cv::Rect& left_eye = validated_eyes[0];
    const cv::Rect& right_eye = validated_eyes[1];

    // 验证水平距离
    float distance = static_cast<float>(right_eye.x - (left_eye.x + left_eye.width));
    if (distance < 0 || distance > left_eye.width * 2) {
        return {}; // 重叠或距离太远
    }

    // 验证大小相似性
    float size_ratio = static_cast<float>(left_eye.area()) / right_eye.area();
    if (size_ratio < 0.6 || size_ratio > 1.6) {
        return {};
    }

    // 验证垂直位置相似性
    float vertical_diff = std::abs(left_eye.y - right_eye.y);
    if (vertical_diff > left_eye.height * 0.5) {
        return {};
    }

    return {left_eye, right_eye};
}

int main() {
    cv::Mat image = cv::imread("input/sample.png", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Could not load image." << std::endl;
        return -1;
    }

    cv::CascadeClassifier eye_cascade;
    if (!eye_cascade.load("haarcascades/haarcascade_eye_tree_eyeglasses.xml")) {
        std::cerr << "Error: Could not load eye cascade classifier." << std::endl;
        return -1;
    }

    std::vector<cv::Rect> eyes;
    // 使用更严格的参数进行检测，以减少误报
    eye_cascade.detectMultiScale(image, eyes, 1.1, 5, 0, cv::Size(40, 40));

    if (eyes.size() < 2) {
        std::cout << "Could not detect two eyes." << std::endl;
        return -1;
    }

    std::vector<cv::Rect> valid_eyes = validate_and_sort_eyes(eyes, image.rows * image.cols);

    if (valid_eyes.size() == 2) {
        cv::Mat color_image;
        cv::cvtColor(image, color_image, cv::COLOR_GRAY2BGR);

        // 在图像上绘制验证后的眼睛区域
        cv::rectangle(color_image, valid_eyes[0], cv::Scalar(0, 255, 0), 2); // 左眼
        cv::rectangle(color_image, valid_eyes[1], cv::Scalar(0, 0, 255), 2); // 右眼

        cv::imshow("Validated Eyes", color_image);
        cv::waitKey(0);
        std::cout << "Validated eye regions found and displayed." << std::endl;
    } else {
        std::cout << "Validation failed. Could not confirm two valid eyes." << std::endl;
    }

    return 0;
}