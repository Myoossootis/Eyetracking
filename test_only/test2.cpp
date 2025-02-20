#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <vector>
#include <fstream>

int main() {
    // 加载图像
    cv::Mat image = cv::imread("6.bmp", cv::IMREAD_GRAYSCALE); // 替换为你的图像路径
    if (image.empty()) {
        std::cerr << "Error: Unable to load image!" << std::endl;
        return -1;
    }

    // 加载眼睛检测器
    cv::CascadeClassifier eye_cascade;
    if (!eye_cascade.load("haarcascades/haarcascade_eye.xml")) { // 确保路径正确
        std::cerr << "Error: Unable to load eye cascade classifier!" << std::endl;
        return -1;
    }

    // 检测眼睛
    std::vector<cv::Rect> eyes;
    eye_cascade.detectMultiScale(image, eyes, 1.1, 4, 0, cv::Size(30, 30)); // 最小检测尺寸

    // 在图像上绘制矩形框
    for (const auto& eye : eyes) {
        cv::rectangle(image, eye, cv::Scalar(0, 255, 0), 2); // 绿色矩形框
    }

    // 假设检测到的两个眼睛是左右眼（按x坐标排序）
    if (eyes.size() < 2) {
        std::cerr << "Less than two eyes detected. Cannot process further." << std::endl;
        return -1;
    }

    // 按x坐标排序，确保左眼在前，右眼在后
    std::sort(eyes.begin(), eyes.end(), [](const cv::Rect& a, const cv::Rect& b) {
        return a.x < b.x;
        });

    // 转换为彩色图像用于绘制
    cv::Mat result;
    cv::cvtColor(image, result, cv::COLOR_GRAY2BGR);

    // 打开输出文件
    std::ofstream outFile("output/ellipse_centers.txt");
    if (!outFile.is_open()) {
        std::cerr << "Error: Unable to open output file for writing!" << std::endl;
        return -1;
    }
    outFile << std::fixed << std::setprecision(2); // 设置精度为2位小数

    // 处理左眼
    cv::Mat left_eye = image(eyes[0]);
    cv::Mat blurred_left_eye;
    cv::medianBlur(left_eye, blurred_left_eye, 3); // 中值滤波
    cv::imshow("blurr", blurred_left_eye);
    cv::Mat binary_left_eye;
    cv::threshold(blurred_left_eye, binary_left_eye, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU); // 大津法阈值分割
    cv::imshow("binaryleft", binary_left_eye);
    std::vector<std::vector<cv::Point>> left_contours;
    cv::findContours(binary_left_eye, left_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const auto& contour : left_contours) {
        if (contour.size() >= 5) { // 确保轮廓点数足够拟合椭圆
            cv::RotatedRect fitted_ellipse = cv::fitEllipse(contour);
            double area = cv::contourArea(contour);
            double aspectRatio = std::abs(fitted_ellipse.size.width / fitted_ellipse.size.height);

            if (area > 20 && area < 150 && aspectRatio > 0.5 && aspectRatio < 2.0) {
                // 转换为全局坐标
                cv::Point2f center = fitted_ellipse.center;
                center.x += eyes[0].x;
                center.y += eyes[0].y;

                // 转换椭圆的边界框为全局坐标
                cv::RotatedRect global_ellipse(fitted_ellipse.center + cv::Point2f(eyes[0].x, eyes[0].y),
                    fitted_ellipse.size, fitted_ellipse.angle);

                // 绘制椭圆和中心点
                cv::ellipse(result, global_ellipse, cv::Scalar(0, 255, 0), 2); // 绿色椭圆
                cv::circle(result, center, 2, cv::Scalar(0, 0, 255), -1); // 红色中心点

                // 写入文件
                outFile << "Left Eye Ellipse Center: (" << center.x << ", " << center.y << ")" << std::endl;
            }
        }
    }

    // 处理右眼
    cv::Mat right_eye = image(eyes[1]);
    cv::Mat blurred_right_eye;
    cv::medianBlur(right_eye, blurred_right_eye,7); // 中值滤波
    cv::Mat binary_right_eye;
    cv::threshold(blurred_right_eye, binary_right_eye, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU); // 大津法阈值分割

    std::vector<std::vector<cv::Point>> right_contours;
    cv::findContours(binary_right_eye, right_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const auto& contour : right_contours) {
        if (contour.size() >= 5) { // 确保轮廓点数足够拟合椭圆
            cv::RotatedRect fitted_ellipse = cv::fitEllipse(contour);
            double area = cv::contourArea(contour);
            double aspectRatio = std::abs(fitted_ellipse.size.width / fitted_ellipse.size.height);

            if (area > 20 && area < 150 && aspectRatio > 0.5 && aspectRatio < 2.0) {
                // 转换为全局坐标
                cv::Point2f center = fitted_ellipse.center;
                center.x += eyes[1].x;
                center.y += eyes[1].y;

                // 转换椭圆的边界框为全局坐标
                cv::RotatedRect global_ellipse(fitted_ellipse.center + cv::Point2f(eyes[1].x, eyes[1].y),
                    fitted_ellipse.size, fitted_ellipse.angle);

                // 绘制椭圆和中心点
                cv::ellipse(result, global_ellipse, cv::Scalar(0, 255, 0), 2); // 绿色椭圆
                cv::circle(result, center, 2, cv::Scalar(0, 0, 255), -1); // 红色中心点

                // 写入文件
                outFile << "Right Eye Ellipse Center: (" << center.x << ", " << center.y << ")" << std::endl;
            }
        }
    }

    outFile.close();

    // 显示结果
    cv::imshow("Result", result);
    cv::waitKey(0);
    return 0;
}