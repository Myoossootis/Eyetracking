#ifndef PUPIL_DETECTION_H
#define PUPIL_DETECTION_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// 加载图像
cv::Mat load_image(const std::string& file_path);

// 获取瞳孔位置
cv::Mat get_pupil_position(const cv::Mat& image_dark, const cv::Mat& image_light);

// 高斯滤波
cv::Mat apply_gaussian_blur(const cv::Mat& image);

// 阈值化处理
cv::Mat apply_threshold(const cv::Mat& image);

// 边缘检测
cv::Mat detect_edges(const cv::Mat& binary_image);

// 查找最大的轮廓
std::vector<cv::Point> find_largest_contour(const std::vector<std::vector<cv::Point>>& contours);

// 绘制椭圆和提取斑点信息
void process_ellipse_and_blobs(
    const std::vector<std::vector<cv::Point>>& contours,
    const cv::Mat& image_light,
    const std::string& output_file
);

// 保存结果图像
void save_results(const cv::Mat& pupil_position, const cv::Mat& binary,
    const cv::Mat& result, const cv::Mat& detected_blobs);

#endif // PUPIL_DETECTION_H
