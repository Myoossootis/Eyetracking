#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 加载明瞳图像和暗瞳图像
    cv::Mat image_light = cv::imread("1.bmp", cv::IMREAD_GRAYSCALE);
    cv::Mat image_dark = cv::imread("2.bmp", cv::IMREAD_GRAYSCALE);

    // 检查图像是否加载成功
    if (image_light.empty() || image_dark.empty()) {
        std::cerr << "Error: Unable to load images!" << std::endl;
        return -1;
    }

    // Step 1: 明瞳图像减去暗瞳图像，得到瞳孔位置
    cv::Mat pupil_position;
    cv::absdiff(image_light, image_dark, pupil_position);

    // Step 2: 将明瞳图像上的瞳孔区域去除
    cv::Mat pupil_removed;
    cv::subtract(image_light, pupil_position, pupil_removed);

    // Step 3: 用暗瞳图像减去上一部分，得到反射亮斑位置
    cv::Mat reflection_spot;
    cv::subtract(image_dark, pupil_removed, reflection_spot);

    // 显示结果
    cv::imshow("Reflection Spot", reflection_spot);
    cv::imwrite("pupil_position.jpg", pupil_position);
    cv::imwrite("pupil_removed.jpg", pupil_removed);
    cv::imwrite("reflection_spot.jpg", reflection_spot);

    cv::Mat reflection = cv::imread("reflection_spot.jpg", cv::IMREAD_GRAYSCALE);

    // 检查图像是否加载成功
    if (reflection.empty()) {
        std::cerr << "Error: Unable to load image!" << std::endl;
        return -1;
    }
    // CLAHE 提高对比度
    cv::Mat enhanced;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));  // 设定 CLAHE 参数
    clahe->apply(reflection, enhanced);

    // 高斯模糊，减少噪声
    cv::Mat blurred;
    cv::GaussianBlur(reflection, blurred, cv::Size(5, 5), 1.5);

    // Canny 边缘检测
    cv::Mat edges;
    cv::Canny(blurred, edges, 50, 150);  // Canny算法的低阈值和高阈值
    // 形态学闭运算
    cv::Mat closed;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(10, 10));  // 使用椭圆形结构元素
    cv::morphologyEx(edges, closed, cv::MORPH_CLOSE, kernel);
    // 显示边缘图像
    cv::imshow("Canny Edges", edges);
    cv::imshow("Morphological Closing", closed);
    // 获取轮廓并拟合圆形
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(closed, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    // 创建一个用于显示结果的图像
    cv::Mat result = cv::Mat::zeros(closed.size(), CV_8UC3); // 创建一个彩色图像，方便绘制圆形

    for (size_t i = 0; i < contours.size(); i++) {
        // 拟合最小外接圆
        cv::Point2f center;
        float radius;
        cv::minEnclosingCircle(contours[i], center, radius);

        // 增加绘制精度，圆的线条可以增加到细的1像素
        cv::circle(result, center, cvRound(radius), cv::Scalar(0, 255, 0), 1);  // 绿色圆形
        cv::circle(result, center, 1, cv::Scalar(0, 0, 255), -1);  // 圆心红色标记

        // 输出拟合圆心和半径的精度
        std::cout << "Circle center: " << center << ", radius: " << radius << std::endl;
    }

    // 显示结果
    cv::imshow("Fitted Circles", result);

    // 保存结果
    cv::imwrite("fitted_circles.jpg", result);

    // 加载反射亮斑图像（灰度图）和拟合圆形图像（彩色图）

    cv::Mat fitted_circles = cv::imread("fitted_circles.jpg", cv::IMREAD_COLOR);

    // 检查图像是否加载成功
    if (reflection_spot.empty() || fitted_circles.empty()) {
        std::cerr << "Error: Unable to load images!" << std::endl;
        return -1;
    }

    // 将反射亮斑图像从灰度转换为彩色
    cv::Mat reflection_color;
    cv::cvtColor(reflection_spot, reflection_color, cv::COLOR_GRAY2BGR);  // 转换为彩色图像

    // 调整两张图的大小，使它们能够相加
    if (reflection_color.size() != fitted_circles.size()) {
        cv::resize(fitted_circles, fitted_circles, reflection_color.size());
    }

    // 合并两张图，保持彩色标注

    cv::addWeighted(reflection_color, 0.7, fitted_circles, 0.3, 0, result);

    // 显示合并结果
    cv::imshow("Combined Result", result);

    // 保存合并后的图像
    cv::imwrite("combined_result.jpg", result);



    // 等待按键退出
    cv::waitKey(0);
    return 0;
}