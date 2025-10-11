#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

class ReflectionDetector {
public:
    static cv::Point detectReflection(const cv::Mat& eye_image) {
        // 定义滤波核大小
        int max_filter_size = 3;
        int median_filter_size = 3;

        // 确保核大小为奇数
        max_filter_size = max_filter_size % 2 == 0 ? max_filter_size + 1 : max_filter_size;
        median_filter_size = median_filter_size % 2 == 0 ? median_filter_size + 1 : median_filter_size;

        // 最大值滤波（使用圆形滤波核）
        cv::Mat max_filtered;
        cv::dilate(eye_image, max_filtered, cv::getStructuringElement(cv::MORPH_ELLIPSE, 
            cv::Size(max_filter_size, max_filter_size)));

        // 中值滤波
        cv::Mat median_filtered;
        cv::medianBlur(eye_image, median_filtered, median_filter_size);

        // 计算最大值滤波结果减去中值滤波结果
        cv::Mat result;
        cv::subtract(max_filtered, median_filtered, result);
        cv::imshow("result",result);

        // 阈值分割提取亮斑区域
        cv::Mat binary;
        cv::threshold(result, binary, 50, 255, cv::THRESH_BINARY);
        cv::imshow("binary",binary);

        // 查找轮廓
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // 如果没有检测到轮廓，返回无效点
        if (contours.empty()) {
            return cv::Point(-1, -1);
        }

        // 存储所有亮点的中心
        std::vector<cv::Point> centers;
        
        // 遍历所有轮廓，计算每个轮廓的中心点
        for (const auto& contour : contours) {
            cv::Moments m = cv::moments(contour);
            if (m.m00 != 0) {
                cv::Point center(
                    static_cast<int>(m.m10 / m.m00),
                    static_cast<int>(m.m01 / m.m00)
                );
                centers.push_back(center);
            }
        }

        // 如果没有有效的亮点，返回无效点
        if (centers.empty()) {
            return cv::Point(-1, -1);
        }

        // 返回第一个亮点的位置
        return centers[0];
    }

    // 添加新方法返回所有亮点
    static std::vector<cv::Point> detectAllReflections(const cv::Mat& eye_image) {
        // 定义滤波核大小
        int max_filter_size = 3;
        int median_filter_size = 3;

        // 确保核大小为奇数
        max_filter_size = max_filter_size % 2 == 0 ? max_filter_size + 1 : max_filter_size;
        median_filter_size = median_filter_size % 2 == 0 ? median_filter_size + 1 : median_filter_size;

        // 最大值滤波（使用圆形滤波核）
        cv::Mat max_filtered;
        cv::dilate(eye_image, max_filtered, cv::getStructuringElement(cv::MORPH_ELLIPSE, 
            cv::Size(max_filter_size, max_filter_size)));
        cv::imshow("max", max_filtered);
        cv::imwrite("C:\\Users\\杜锦鸿\\Desktop\\论文图片\\max_filtered.jpg", max_filtered);
        // 中值滤波
        cv::Mat median_filtered;
        cv::medianBlur(eye_image, median_filtered, median_filter_size);
        cv::imshow("median", median_filtered);
        cv::imwrite("C:\\Users\\杜锦鸿\\Desktop\\论文图片\\median_filtered.jpg", median_filtered);

        // 计算最大值滤波结果减去中值滤波结果
        cv::Mat result;
        cv::subtract(max_filtered, median_filtered, result);
        cv::imshow("result", result);
        cv::imwrite("C:\\Users\\杜锦鸿\\Desktop\\论文图片\\sutract_result.jpg", result);

        // 阈值分割提取亮斑区域
        cv::Mat binary;
        cv::threshold(result, binary, 50, 255, cv::THRESH_BINARY);
        cv::imshow("binary", binary);
        cv::imwrite("C:\\Users\\杜锦鸿\\Desktop\\论文图片\\binary.jpg", binary);

        // 查找轮廓
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        std::vector<cv::Point> centers;
        for (const auto& contour : contours) {
            cv::Moments m = cv::moments(contour);
            if (m.m00 != 0) {
                centers.push_back(cv::Point(
                    static_cast<int>(m.m10 / m.m00),
                    static_cast<int>(m.m01 / m.m00)
                ));
            }
        }
        return centers;
    }
};

int main() {
    cv::Mat eye_image = cv::imread("28.png", cv::IMREAD_GRAYSCALE);
    if (eye_image.empty()) {
        std::cerr << "Error: Unable to load eye image!" << std::endl;
        return -1;
    }

    try {
        // 检测所有亮斑中心
        std::vector<cv::Point> reflection_centers = ReflectionDetector::detectAllReflections(eye_image);

        if (reflection_centers.empty()) {
            std::cout << "No reflection detected!" << std::endl;
            return -1;
        }

        // 在原图上标记所有亮斑中心
        cv::Mat result_image = eye_image.clone();
        for (const auto& center : reflection_centers) {
            cv::circle(result_image, center, 3, cv::Scalar(255), -1);
            std::cout << "Reflection center at: (" << center.x << ", " 
                      << center.y << ")" << std::endl;
        }

        // 显示结果
        cv::imshow("Original Eye", eye_image);
        cv::imwrite("C:\\Users\\杜锦鸿\\Desktop\\论文图片\\origin.jpg", eye_image);

        cv::imshow("Result with All Reflection Centers", result_image);
        cv::imwrite("C:\\Users\\杜锦鸿\\Desktop\\论文图片\\result_image.jpg", result_image);

        cv::waitKey(0);
        cv::destroyAllWindows();
        
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}