#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

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
    /*cv::imshow("pupil_position", pupil_position);*/
    // 打开输出文件
    std::ofstream outFile("output/ellipse_centers.txt");
    if (!outFile.is_open()) {
        std::cerr << "Error: Unable to open output file!" << std::endl;
        return -1;
    }

    // 创建可调整大小的窗口
    cv::namedWindow("Original Image with Ellipses and Centers", cv::WINDOW_NORMAL);
    cv::namedWindow("Left Eye Gaussian", cv::WINDOW_NORMAL);
    cv::namedWindow("Right Eye Gaussian", cv::WINDOW_NORMAL);
    cv::namedWindow("pupil_position", cv::WINDOW_NORMAL);
    cv::namedWindow("pupil", cv::WINDOW_NORMAL);
    cv::namedWindow("Edges", cv::WINDOW_NORMAL);

    // 处理每个眼睛区域
    for (size_t i = 0; i < eyes.size() && i < 2; ++i) {
        const auto& eye = eyes[i];
        // 提取明瞳和暗瞳图像的眼睛区域
        cv::Mat eye_light = image_light(eye);
        cv::Mat eye_dark = image_dark(eye);
        cv::imshow("pupil_position", eye_light);

        // 保存眼部区域原图（包括明瞳和暗瞳）
        if (i == 0) {
            // 对左眼进行CLAHE处理
            cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(4.0, cv::Size(2,2));
            cv::Mat clahe_result;
            clahe->apply(eye_light, clahe_result);
            
            // 保存原图、暗瞳图和CLAHE处理后的图像
            cv::imwrite("C:\\Users\\杜锦鸿\\Desktop\\论文图片\\left_eye_original.jpg", eye_light);
            cv::imwrite("C:\\Users\\杜锦鸿\\Desktop\\论文图片\\left_eye_dark.jpg", eye_dark);
            cv::imwrite("C:\\Users\\杜锦鸿\\Desktop\\论文图片\\left_eye_clahe.jpg", clahe_result);
            
            // 使用CLAHE处理后的图像继续后续处理
            eye_light = clahe_result;
        } else {
            cv::imwrite("C:\\Users\\杜锦鸿\\Desktop\\论文图片\\right_eye_original.jpg", eye_light);
            cv::imwrite("C:\\Users\\杜锦鸿\\Desktop\\论文图片\\right_eye_dark.jpg", eye_dark);
        }

        // 直接使用明瞳图像进行处理
        cv::Mat eye_pupil_position = eye_light.clone();

        // 将结果放回原图对应位置
        eye_pupil_position.copyTo(pupil_position(eye));
        cv::imshow("pupil", eye_pupil_position);

        // 高斯模糊，减少噪声
        cv::Mat blurred_eye;
        cv::GaussianBlur(eye_pupil_position, blurred_eye, cv::Size(5, 5), 1.5);
        // 分别显示和保存左右眼的高斯模糊结果
        if (i == 0) {
            cv::imshow("Left Eye Gaussian", blurred_eye);
            cv::imwrite("C:\\Users\\杜锦鸿\\Desktop\\论文图片\\left_eye_gaussian.jpg", blurred_eye);
        } else {
            cv::imshow("Right Eye Gaussian", blurred_eye);
            cv::imwrite("C:\\Users\\杜锦鸿\\Desktop\\论文图片\\right_eye_gaussian.jpg", blurred_eye);
        }

        // 进行二值化处理
        cv::Mat binary;
        cv::threshold(blurred_eye, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        
        // 保存二值化结果
        if (i == 0) {
            cv::imwrite("C:\\Users\\杜锦鸿\\Desktop\\论文图片\\left_eye_binary.jpg", binary);
        } else {
            cv::imwrite("C:\\Users\\杜锦鸿\\Desktop\\论文图片\\right_eye_binary.jpg", binary);
        }

        // 进行闭运算
        cv::Mat closed;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::morphologyEx(binary, closed, cv::MORPH_CLOSE, kernel);

        // 保存闭运算结果
        if (i == 0) {
            cv::imwrite("C:\\Users\\杜锦鸿\\Desktop\\论文图片\\left_eye_closed.jpg", closed);
        } else {
            cv::imwrite("C:\\Users\\杜锦鸿\\Desktop\\论文图片\\right_eye_closed.jpg", closed);
        }

        // 进行开运算
        cv::Mat opened;
        cv::morphologyEx(closed, opened, cv::MORPH_OPEN, kernel);

        // 保存开运算结果
        if (i == 0) {
            cv::imwrite("C:\\Users\\杜锦鸿\\Desktop\\论文图片\\left_eye_opened.jpg", opened);
        } else {
            cv::imwrite("C:\\Users\\杜锦鸿\\Desktop\\论文图片\\right_eye_opened.jpg", opened);
        }

        // 使用开运算后的图像进行边缘检测
        cv::Mat edges;
        cv::Canny(opened, edges, 50, 150);  // 使用开运算后的图像进行边缘检测
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

            }
        }
    }

    // 调整窗口大小
    cv::resizeWindow("Original Image with Ellipses and Centers", 800, 600);
    cv::resizeWindow("Left Eye Gaussian", 400, 300);
    cv::resizeWindow("Right Eye Gaussian", 400, 300);
    cv::resizeWindow("pupil_position", 400, 300);
    cv::resizeWindow("pupil", 400, 300);
    cv::resizeWindow("Edges", 400, 300);

    // 显示结果
    cv::imshow("Original Image with Ellipses and Centers", image_light);
    cv::waitKey(0);

    // 销毁所有窗口
    cv::destroyAllWindows();

    return 0;
}