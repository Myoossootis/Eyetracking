#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>  // 用于文件操作

class FaceEyeDetector {
public:
    static std::pair<cv::Rect, cv::Rect> detectEyeRegions(const cv::Mat& image) {
        // 加载检测器
        cv::CascadeClassifier eye_cascade;
        cv::CascadeClassifier face_cascade_front;
        cv::CascadeClassifier face_cascade_profile;

        // 检查所有检测器是否正确加载
        if (!eye_cascade.load("haarcascades/haarcascade_eye.xml")) {
            throw std::runtime_error("Failed to load eye detector");
        }
        if (!face_cascade_front.load("haarcascades/haarcascade_frontalface_alt.xml")) {
            throw std::runtime_error("Failed to load front face detector");
        }
        if (!face_cascade_profile.load("haarcascades/haarcascade_profileface.xml")) {
            throw std::runtime_error("Failed to load profile face detector");
        }

        // 首先尝试正面人脸检测
        std::vector<cv::Rect> faces_front;
        face_cascade_front.detectMultiScale(image, faces_front, 1.1, 3, 0, cv::Size(150, 150));
        
        if (!faces_front.empty()) {
            return extractEyeRegionsFromFace(image, faces_front[0]);
        }

        // 尝试侧面人脸检测
        std::vector<cv::Rect> faces_profile;
        face_cascade_profile.detectMultiScale(image, faces_profile, 1.1, 3, 0, cv::Size(150, 150));
        
        if (!faces_profile.empty()) {
            return extractEyeRegionsFromFace(image, faces_profile[0]);
        }

        // 人脸检测失败，尝试直接检测眼睛
        std::vector<cv::Rect> eyes;
        eye_cascade.detectMultiScale(image, eyes, 1.1, 3, 0, cv::Size(30, 30));

        if (eyes.size() >= 2) {
            // 按x坐标排序，确保左眼在前
            std::sort(eyes.begin(), eyes.end(),
                [](const cv::Rect& a, const cv::Rect& b) { return a.x < b.x; });
            return std::make_pair(eyes[0], eyes[1]);
        }

        // 所有检测都失败
        throw std::runtime_error("No eyes or face detected");
    }

private:
    static std::pair<cv::Rect, cv::Rect> extractEyeRegionsFromFace(const cv::Mat& image, const cv::Rect& face) {
        int eye_region_height = face.height / 3;
        int eye_region_width = face.width / 3;  // 修改为三分之一宽度
        int eye_region_top = face.y + face.height / 4;

        // 修改眼睛区域的计算方式
        cv::Rect left_eye_rect(
            face.x + face.width / 6,  // 修改起始位置
            eye_region_top,
            eye_region_width,         // 使用固定宽度
            eye_region_height
        );

        cv::Rect right_eye_rect(
            face.x + face.width / 2,  // 右眼起始位置
            eye_region_top,
            eye_region_width,         // 使用固定宽度
            eye_region_height
        );

        // 确保矩形在图像范围内
        left_eye_rect &= cv::Rect(0, 0, image.cols, image.rows);
        right_eye_rect &= cv::Rect(0, 0, image.cols, image.rows);

        return std::make_pair(left_eye_rect, right_eye_rect);
    }
};

int main() {
    // 加载图像
    cv::Mat image = cv::imread("17.png", cv::IMREAD_GRAYSCALE); // 替换为你的图像路径
    if (image.empty()) {
        std::cerr << "Error: Unable to load image!" << std::endl;
        return -1;
    }

    try {
        // 使用FaceEyeDetector检测眼睛区域
        std::pair<cv::Rect, cv::Rect> eye_regions = FaceEyeDetector::detectEyeRegions(image);
        cv::Rect left_eye_rect = eye_regions.first;
        cv::Rect right_eye_rect = eye_regions.second;

        // 提取左眼和右眼区域
        cv::Mat left_eye = image(left_eye_rect);
        cv::Mat right_eye = image(right_eye_rect);

    // 定义最大值滤波和中值滤波的核大小
    int max_filter_size = 5;  // 最大值滤波的核大小
    int median_filter_size = 3;  // 中值滤波的核大小

    // 确保核大小为奇数
    max_filter_size = max_filter_size % 2 == 0 ? max_filter_size + 1 : max_filter_size;
    median_filter_size = median_filter_size % 2 == 0 ? median_filter_size + 1 : median_filter_size;

    // 左眼处理
    // 最大值滤波（使用圆形滤波核）
    cv::Mat max_filtered_left;
    cv::dilate(left_eye, max_filtered_left, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(max_filter_size, max_filter_size)));

    // 中值滤波
    cv::Mat median_filtered_left;
    cv::medianBlur(left_eye, median_filtered_left, median_filter_size);

    // 计算最大值滤波结果减去中值滤波结果
    cv::Mat result_left;
    cv::subtract(max_filtered_left, median_filtered_left, result_left);

    // 右眼处理
    // 最大值滤波（使用圆形滤波核）
    cv::Mat max_filtered_right;
    cv::dilate(right_eye, max_filtered_right, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(max_filter_size, max_filter_size)));

    // 中值滤波
    cv::Mat median_filtered_right;
    cv::medianBlur(right_eye, median_filtered_right, median_filter_size);

    // 计算最大值滤波结果减去中值滤波结果
    cv::Mat result_right;
    cv::subtract(max_filtered_right, median_filtered_right, result_right);

    // 提取亮斑中心坐标
    auto extractBrightSpotCenter = [](const cv::Mat& result, const cv::Rect& eyeRect, cv::Mat& original, std::ofstream& outFile) {
        // 阈值分割提取亮斑区域
        cv::Mat binary;
        cv::threshold(result, binary, 50, 255, cv::THRESH_BINARY); // 调整阈值以适应亮斑

        // 查找轮廓
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // 如果没有检测到轮廓，返回
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
        cv::circle(original, cv::Point(eyeRect.x + cx, eyeRect.y + cy), 3, cv::Scalar(0, 0, 255), -1); // 红色圆点

        // 写入文件
        outFile << "Left Eye Highlight Center: (" << (eyeRect.x + cx) << ", " << (eyeRect.y + cy) << ")" << std::endl;
        };

    // 打开输出文件
    std::ofstream outFile("output/highlight_centers.txt");
    if (!outFile.is_open()) {
        std::cerr << "Error: Unable to open output file!" << std::endl;
        return -1;
    }

    // 提取左眼和右眼的亮斑中心并绘制在原图上
    extractBrightSpotCenter(result_left, left_eye_rect, image, outFile);
    extractBrightSpotCenter(result_right, right_eye_rect, image, outFile);

    // 关闭文件
    outFile.close();

    // 显示结果
    cv::imshow("Original Image with Bright Spot Centers", image);
    cv::imshow("Result Left Eye (Max - Median)", result_left);
    cv::imshow("Result Right Eye (Max - Median)", result_right);

    // 等待按键
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}