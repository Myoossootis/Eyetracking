#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream>  // 用于文件操作

int main() {
    // Step 1: 加载明瞳和暗瞳图像
    cv::Mat image_light = cv::imread("2.bmp", cv::IMREAD_GRAYSCALE); // 明瞳图像
    cv::Mat image_dark = cv::imread("1.bmp", cv::IMREAD_GRAYSCALE);   // 暗瞳图像

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

    // 打开输出文件
    std::ofstream outFile("output/ellipse_centers.txt");
    if (!outFile.is_open()) {
        std::cerr << "Error: Unable to open output file!" << std::endl;
        return -1;
    }

    // 处理每个眼睛区域
    for (const auto& eye : eyes) {
        // 提取明瞳和暗瞳图像的眼睛区域
        cv::Mat eye_light = image_light(eye);
        cv::Mat eye_dark = image_dark(eye);

        // 明瞳减去暗瞳，得到瞳孔位置
        cv::Mat eye_pupil_position;
        cv::absdiff(eye_light, eye_dark, eye_pupil_position);

        // 将结果放回原图对应位置
        eye_pupil_position.copyTo(pupil_position(eye));

        // 高斯模糊，减少噪声
        cv::Mat blurred_eye;
        cv::GaussianBlur(eye_pupil_position, blurred_eye, cv::Size(5, 5), 1.5);

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
                center.x += eye.x;
                center.y += eye.y;

                // 在原始图像上绘制椭圆
                cv::ellipse(image_light, ellipse, cv::Scalar(0, 255, 0), 2);

                // 在原始图像上绘制中心点
                cv::circle(image_light, center, 3, cv::Scalar(0, 0, 255), -1);

                // 写入文件
                outFile << "Ellipse Center: (" << center.x << ", " << center.y << ")" << std::endl;
            }
        }
    }

    // 显示和保存瞳孔位置图像
    cv::imshow("Pupil Position", pupil_position);
    cv::imwrite("pupil_position.bmp", pupil_position);

    // 显示原始图像
    cv::imshow("Original Image with Ellipses and Centers", image_light);

    // 关闭文件
    outFile.close();

    cv::waitKey(0);
    return 0;
}