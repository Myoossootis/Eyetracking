#include "detection.h"
#include <opencv2/opencv.hpp>
#include <vector>

// 定义椭圆筛选的最小和最大面积
const double MIN_ELLIPSE_AREA = 500.0;
const double MAX_ELLIPSE_AREA = 3000.0;

/**
 * @brief 对图像进行预处理，包括灰度化、高斯模糊和二值化。
 *
 * @param image 输入图像。
 * @return 经过预处理的二值图像。
 */
cv::Mat preprocess_image(const cv::Mat& image) {
    cv::Mat gray, blurred, binary;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
    cv::threshold(blurred, binary, 50, 255, cv::THRESH_BINARY);
    return binary;
}

/**
 * @brief 在图像中寻找符合条件的瞳孔轮廓。
 *
 * @param binary_image 预处理后的二值图像。
 * @return 瞳孔轮廓的向量。
 */
std::vector<std::vector<cv::Point>> find_pupil_contours(const cv::Mat& binary_image) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary_image, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    return contours;
}

/**
 * @brief 从轮廓中筛选出最可能是瞳孔的椭圆。
 *
 * @param contours 轮廓向量。
 * @param image_for_drawing 用于绘制的原始图像。
 * @return 检测到的瞳孔中心点。如果未找到，则返回 (-1, -1)。
 */
cv::Point2f find_best_pupil_ellipse(const std::vector<std::vector<cv::Point>>& contours, cv::Mat& image_for_drawing) {
    cv::Point2f pupil_center(-1, -1);

    for (const auto& contour : contours) {
        if (contour.size() > 5) { // 椭圆拟合至少需要6个点
            cv::RotatedRect ellipse_rect = cv::fitEllipse(contour);
            double area = ellipse_rect.size.width * ellipse_rect.size.height * CV_PI / 4.0;

            // 根据面积和形状筛选
            if (area > MIN_ELLIPSE_AREA && area < MAX_ELLIPSE_AREA) {
                float aspect_ratio = ellipse_rect.size.width / ellipse_rect.size.height;
                if (aspect_ratio > 0.75 && aspect_ratio < 1.25) { // 接近圆形
                    pupil_center = ellipse_rect.center;
                    cv::ellipse(image_for_drawing, ellipse_rect, cv::Scalar(0, 255, 0), 2);
                    break; // 找到一个就停止
                }
            }
        }
    }
    return pupil_center;
}

void detect_pupil(const std::string& light_image_path, const std::string& dark_image_path, const std::string& output_file) {
    cv::Mat light_image = cv::imread(light_image_path);
    cv::Mat dark_image = cv::imread(dark_image_path);

    if (light_image.empty() || dark_image.empty()) {
        std::cerr << "Error: Could not load images for pupil detection." << std::endl;
        return;
    }

    // 1. 图像差分
    cv::Mat diff_image;
    cv::absdiff(light_image, dark_image, diff_image);

    // 2. 预处理
    cv::Mat binary_diff = preprocess_image(diff_image);

    // 3. 寻找轮廓
    std::vector<std::vector<cv::Point>> contours = find_pupil_contours(binary_diff);

    // 4. 寻找最佳瞳孔椭圆
    cv::Mat image_for_drawing = light_image.clone();
    cv::Point2f pupil_center = find_best_pupil_ellipse(contours, image_for_drawing);

    // 5. 保存结果
    if (pupil_center.x != -1) {
        std::ofstream outfile(output_file);
        if (outfile.is_open()) {
            outfile << pupil_center.x << " " << pupil_center.y << std::endl;
            outfile.close();
        }
        // 可选：显示结果图像
        // cv::imshow("Pupil Detection", image_for_drawing);
        // cv::waitKey(0);
    }
}
