#include <opencv2/opencv.hpp>
#include "pupil_detection.h"

int main() {
    // 加载图像
    cv::Mat image_dark = load_image("1.bmp");
    cv::Mat image_light = load_image("2.bmp");

    // 检查图像是否加载成功
    if (image_light.empty() || image_dark.empty()) {
        return -1;
    }

    // 获取瞳孔位置
    cv::Mat pupil_position = get_pupil_position(image_dark, image_light);

    // 进行高斯滤波
    cv::Mat blurred_image_light = apply_gaussian_blur(image_light);

    // 阈值化处理
    cv::Mat binary = apply_threshold(blurred_image_light);

    // 边缘检测
    cv::Mat edges = detect_edges(binary);

    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    // 处理椭圆和斑点
    process_ellipse_and_blobs(contours, image_light, "output/parameter.txt");

    // 保存结果
    save_results(pupil_position, binary, image_light, image_light);

    return 0;
}
