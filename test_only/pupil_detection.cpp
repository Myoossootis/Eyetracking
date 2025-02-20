#include "pupil_detection.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>

cv::Mat load_image(const std::string& file_path) {
    cv::Mat image = cv::imread(file_path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Unable to load image " << file_path << std::endl;
    }
    return image;
}

cv::Mat get_pupil_position(const cv::Mat& image_dark, const cv::Mat& image_light) {
    // 获取瞳孔位置
    cv::Mat pupil_position;
    cv::absdiff(image_dark, image_light, pupil_position);
    cv::normalize(pupil_position, pupil_position, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    return pupil_position;
}

cv::Mat apply_gaussian_blur(const cv::Mat& image) {
    // 高斯滤波
    cv::Mat blurred_image;
    cv::GaussianBlur(image, blurred_image, cv::Size(5, 5), 0);
    return blurred_image;
}

cv::Mat apply_threshold(const cv::Mat& image) {
    // 阈值化处理
    cv::Mat binary;
    double threshold_value = 0;
    cv::threshold(image, binary, threshold_value, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    return binary;
}

cv::Mat detect_edges(const cv::Mat& binary_image) {
    // 边缘检测
    cv::Mat edges;
    cv::Canny(binary_image, edges, 50, 150);
    return edges;
}

std::vector<cv::Point> find_largest_contour(const std::vector<std::vector<cv::Point>>& contours) {
    // 查找最大轮廓
    std::vector<cv::Point> largest_contour;
    double max_area = 0;
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area > max_area) {
            max_area = area;
            largest_contour = contour;
        }
    }
    return largest_contour;
}

void process_ellipse_and_blobs(const std::vector<std::vector<cv::Point>>& contours,
    const cv::Mat& image_light, const std::string& output_file) {
    std::ofstream outFile(output_file);
    if (!outFile.is_open()) {
        std::cerr << "Error opening file for writing!" << std::endl;
        return;
    }

    outFile << std::fixed << std::setprecision(3); // 设置精度为3位小数
    cv::Mat result = image_light.clone();
    cv::Mat detected_blobs = image_light.clone();
    std::vector<cv::Point2f> centers;

    for (const auto& contour : contours) {
        if (contour.size() < 5) continue; // 跳过点数不足5个的轮廓

        // 拟合椭圆
        cv::RotatedRect fitted_ellipse = cv::fitEllipse(contour);
        double area = cv::contourArea(contour);
        double aspectRatio = std::abs(fitted_ellipse.size.width / fitted_ellipse.size.height);

        // 根据面积和长宽比筛选合适的椭圆
        if (area > 100 && area < 1000 && aspectRatio > 0.5 && aspectRatio < 2.0) {
            cv::Point2f center = fitted_ellipse.center;

            bool isDuplicate = false;
            for (const auto& existing_center : centers) {
                if (cv::norm(center - existing_center) < 10) { // 假设10像素以内认为是重复
                    isDuplicate = true;
                    break;
                }
            }

            if (!isDuplicate) {
                centers.push_back(center);
                cv::ellipse(result, fitted_ellipse, cv::Scalar(0, 255, 0), 2);  // 绘制椭圆
                cv::circle(result, center, 2, cv::Scalar(0, 0, 255), -1);  // 绘制中心点

                outFile << "Pupil_Center X: " << center.x << ", Pupil_Center Y: " << center.y << std::endl;

                // 提取椭圆内的斑点
                cv::Mat mask_raw = cv::Mat::zeros(image_light.size(), CV_8UC1);
                cv::Mat mask;
                cv::equalizeHist(mask_raw, mask);
                cv::ellipse(mask, fitted_ellipse, cv::Scalar(255), -1);
                cv::Mat masked_image;
                image_light.copyTo(masked_image, mask);

                // 手动设置阈值，重新进行阈值分割
                double manual_threshold_value = 105; // 根据需要调整
                cv::Mat binary_masked;
                cv::threshold(masked_image, binary_masked, manual_threshold_value, 255, cv::THRESH_BINARY);

                // 再次检测斑点
                std::vector<std::vector<cv::Point>> masked_contours;
                cv::findContours(binary_masked, masked_contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
                // 过滤并绘制斑点
                int blob_count = 0;
                for (const auto& m_contour : masked_contours) {
                    double m_area = cv::contourArea(m_contour);
                    if (m_area > 3 && m_area < 10) {
                        cv::drawContours(detected_blobs, std::vector<std::vector<cv::Point>>{m_contour}, -1, cv::Scalar(255, 0, 0), 2);
                        blob_count++;
                        // 计算质心
                        cv::Moments moments = cv::moments(m_contour);
                        cv::Point2f centroid = cv::Point2f(moments.m10 / moments.m00, moments.m01 / moments.m00);
                        outFile << "refle_Center X: " << centroid.x << ", refle_Center Y: " << centroid.y << std::endl;

                        // 绘制质心
                        cv::circle(detected_blobs, centroid, 2, cv::Scalar(0, 255, 255), -1);
                    }
                }

                std::cout << "blobs amount: " << blob_count << std::endl;

                // 找到亮度最大的点
                cv::Point brightest_point;
                double max_val;
                cv::minMaxLoc(masked_image, NULL, &max_val, NULL, &brightest_point, mask);
                std::cout << "Brightest Point: (" << brightest_point.x << ", " << brightest_point.y << ") with intensity " << max_val << std::endl;
            }
        }
    }
    outFile.close();
}

void save_results(const cv::Mat& pupil_position, const cv::Mat& binary,
    const cv::Mat& result, const cv::Mat& detected_blobs) {
    // 保存处理后的结果图像
    cv::imwrite("output/pupil_position.jpg", pupil_position);
    cv::imwrite("output/binary.jpg", binary);
    cv::imwrite("output/reflection_center.jpg", detected_blobs);
    cv::imwrite("output/result.jpg", result);
}
