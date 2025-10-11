#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

int main() {
    // Step 1: 加载明瞳和暗瞳图像
    cv::Mat image_light = cv::imread("8.bmp", cv::IMREAD_GRAYSCALE); // 明瞳图像
    cv::Mat image_dark = cv::imread("9.bmp", cv::IMREAD_GRAYSCALE);   // 暗瞳图像

    if (image_light.empty() || image_dark.empty()) {
        std::cerr << "Error: Unable to load images!" << std::endl;
        return -1;
    }

    // Step 2: 使用明瞳图像进行眼睛检测
    // 加载眼睛检测器
    cv::CascadeClassifier eye_cascade;
    if (!eye_cascade.load("haarcascades/haarcascade_eye.xml")) { // 确保路径正确
        std::cerr << "Error: Unable to load eye cascade classifier!" << std::endl;
        return -1;
    }

    // 检测眼睛（使用明瞳图像）
    std::vector<cv::Rect> eyes;
    eye_cascade.detectMultiScale(image_light, eyes, 1.1, 4, 0, cv::Size(30, 30)); // 最小检测尺寸

    // 检查是否检测到眼睛
    if (eyes.empty()) {
        std::cerr << "Error: No eyes detected!" << std::endl;
        return -1;
    }

    // 按x坐标排序，确保左眼在前，右眼在后
    std::sort(eyes.begin(), eyes.end(), [](const cv::Rect& a, const cv::Rect& b) {
        return a.x < b.x;
        });

    // Step 3: 在检测到的眼睛区域内对明瞳和暗瞳图像进行处理
    cv::Mat pupil_position = image_light.clone(); // 创建一个与明瞳图像相同大小的矩阵用于显示结果
    
    // 创建彩色图像用于显示结果
    cv::Mat image_color;
    cv::cvtColor(image_light, image_color, cv::COLOR_GRAY2BGR);
    
    // 创建左右眼区域的彩色图像
    cv::Mat left_eye_result, right_eye_result;

    // 处理每个眼睛区域
    for (size_t i = 0; i < eyes.size() && i < 2; ++i) {
        const auto& eye = eyes[i];
        // 提取明瞳和暗瞳图像的眼睛区域
        cv::Mat eye_light = image_light(eye);
        cv::Mat eye_dark = image_dark(eye);
        
        // 创建当前眼睛区域的彩色图像
        cv::Mat eye_color;
        cv::cvtColor(eye_light, eye_color, cv::COLOR_GRAY2BGR);

        cv::imshow("pupil_position", eye_light);
        cv::imshow("pupil_position", eye_dark);
        // 明瞳减去暗瞳，得到瞳孔位置
        cv::Mat eye_pupil_position;
        cv::absdiff(eye_light, eye_dark, eye_pupil_position);

        // 将结果放回原图对应位置
        eye_pupil_position.copyTo(pupil_position(eye));
        cv::imshow("pupil", eye_pupil_position);
        // 高斯模糊，减少噪声
        cv::Mat blurred_eye;
        cv::GaussianBlur(eye_pupil_position, blurred_eye, cv::Size(5, 5), 1.5);
        cv::imshow("gaussion", blurred_eye);
        // 使用 Canny 边缘检测
        cv::Mat edges;
        cv::Canny(blurred_eye, edges, 50, 150);  // 调整阈值以适应图像
        cv::imshow("Edges", edges);

        // 查找轮廓
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // 检查是否找到轮廓
        if (contours.empty()) {
            std::cerr << "No contours found in the eye region!" << std::endl;
            continue;
        }

        // 椭圆拟合
        for (const auto& contour : contours) {
            if (contour.size() >= 5) {  // 椭圆拟合至少需要 5 个点
                cv::RotatedRect ellipse = cv::fitEllipse(contour);
                cv::Point2f center = ellipse.center;

                // 转换为全局坐标
                cv::Point2f global_center(center.x + eye.x, center.y + eye.y);

                // 创建一个调整后的椭圆
                cv::RotatedRect adjusted_ellipse = ellipse;
                adjusted_ellipse.center.x += eye.x;
                adjusted_ellipse.center.y += eye.y;

                // 在原始彩色图像上绘制椭圆和中心点
                cv::ellipse(image_color, adjusted_ellipse, cv::Scalar(0, 255, 0), 2);  // 绿色椭圆
                cv::circle(image_color, global_center, 3, cv::Scalar(0, 0, 255), -1);  // 红色中心点

                // 在眼睛区域图像上绘制椭圆和中心点
                cv::ellipse(eye_color, ellipse, cv::Scalar(0, 255, 0), 1);  // 绿色椭圆
                cv::circle(eye_color, center, 1, cv::Scalar(0, 0, 255), -1);  // 红色中心点
            }
        }

        // 保存处理后的眼睛区域图像
        if (i == 0) {
            // 保存左眼原图（转为彩色但未绘制椭圆和中心点）
            cv::Mat left_eye_original;
            cv::cvtColor(eye_light, left_eye_original, cv::COLOR_GRAY2BGR);
            cv::imwrite("C:\\Users\\杜锦鸿\\Desktop\\论文图片\\left_eye_original_color2.jpg", left_eye_original);
            
            // 保存左眼绘制结果
            left_eye_result = eye_color.clone();
            cv::imwrite("C:\\Users\\杜锦鸿\\Desktop\\论文图片\\left_eye_result2.jpg", left_eye_result);
        } else {
            right_eye_result = eye_color.clone();
        }
    }

    // 显示结果
    cv::imshow("Original Image with Ellipses and Centers", image_color);
    cv::imshow("Left Eye Result", left_eye_result);
    cv::imshow("Right Eye Result", right_eye_result);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}