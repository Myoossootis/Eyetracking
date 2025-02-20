#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 读取图像
    cv::Mat img = cv::imread("7.bmp", cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "无法打开图像!" << std::endl;
        return -1;
    }

    // 转换为灰度图像
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // 使用中值滤波去除小亮斑
    cv::Mat medianFiltered;
    cv::medianBlur(gray, medianFiltered, 5);  // 使用 5x5 的中值滤波器
    cv::imshow("Median Filtered", medianFiltered);

    // 使用 Canny 边缘检测
    cv::Mat edges;
    cv::Canny(medianFiltered, edges, 50, 150);
    cv::imshow("edges", edges);

    // 形态学操作：膨胀边缘，以增强轮廓
    cv::Mat dilatedEdges;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));  // 椭圆形结构元素
    cv::dilate(edges, dilatedEdges, kernel);
    cv::imshow("dilatedEdges", dilatedEdges);

    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(dilatedEdges, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 筛选轮廓：过滤掉小亮斑，保留较大的轮廓
    std::vector<std::vector<cv::Point>> filteredContours;
    for (size_t i = 0; i < contours.size(); i++) {
        if (cv::contourArea(contours[i]) > 200) {  // 过滤掉小面积的轮廓，阈值可以根据实际情况调整
            filteredContours.push_back(contours[i]);
        }
    }

    // 绘制和拟合椭圆
    for (size_t i = 0; i < filteredContours.size(); i++) {
        if (filteredContours[i].size() >= 5) {  // 至少有 5 个点才能拟合椭圆
            cv::RotatedRect ellipse = cv::fitEllipse(filteredContours[i]);  // 拟合椭圆
            cv::ellipse(img, ellipse, cv::Scalar(0, 255, 0), 2);  // 绘制椭圆
        }
    }

    // 显示图像
    cv::imshow("Ellipses Detected", img);
    cv::waitKey(0);

    return 0;
}
