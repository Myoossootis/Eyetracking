#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    // 读取图像
    cv::Mat image = cv::imread("2.bmp", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Unable to load image!" << std::endl;
        return -1;
    }

    // 加载眼睛检测器
    cv::CascadeClassifier eye_cascade;
    if (!eye_cascade.load("haarcascades/haarcascade_eye.xml")) {
        std::cerr << "Error: Unable to load eye cascade classifier!" << std::endl;
        return -1;
    }

    // 眼睛检测
    std::vector<cv::Rect> eyes;
    eye_cascade.detectMultiScale(image, eyes, 1.1, 4, 0, cv::Size(30, 30));

    if (eyes.empty()) {
        std::cerr << "Error: No eyes detected!" << std::endl;
        return -1;
    }

    // 按x坐标排序，确保左眼在前
    std::sort(eyes.begin(), eyes.end(), [](const cv::Rect& a, const cv::Rect& b) {
        return a.x < b.x;
    });

    // 获取左眼区域
    cv::Mat left_eye = image(eyes[0]);

    // 处理方式0：原图直接处理
    cv::Mat original_result = left_eye.clone();

    // 处理方式1：CLAHE（自适应直方图均衡化）
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(4.0, cv::Size(2,2));
    cv::Mat clahe_result;
    clahe->apply(left_eye, clahe_result);

    // 处理方式2：CLAHE+高斯
    cv::Mat clahe_gaussian_result;
    clahe->apply(left_eye, clahe_gaussian_result);
    cv::GaussianBlur(clahe_gaussian_result, clahe_gaussian_result, cv::Size(5, 5), 1.5);

    // 处理方式3：CLAHE+高斯+形态学处理
    cv::Mat morph_result;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(clahe_gaussian_result, morph_result, cv::MORPH_OPEN, kernel);

    // 对每种处理方式进行二值化和边缘检测
    std::vector<cv::Mat> processed_images = {original_result, clahe_result, clahe_gaussian_result, morph_result};
    std::vector<std::string> window_names = {
        "Original", "Original Binary", "Original Edges",
        "CLAHE", "CLAHE Binary", "CLAHE Edges",
        "CLAHE+Gaussian", "CLAHE+Gaussian Binary", "CLAHE+Gaussian Edges",
        "CLAHE+Gaussian+Morphological", "CLAHE+Gaussian+Morphological Binary", "CLAHE+Gaussian+Morphological Edges"
    };

    // 创建窗口并调整大小
    for (const auto& name : window_names) {
        cv::namedWindow(name, cv::WINDOW_NORMAL);
        cv::resizeWindow(name, 400, 300);
    }

    // 处理和显示每种方法的结果
    for (size_t i = 0; i < processed_images.size(); i++) {
        // 显示处理后的图像
        cv::imshow(window_names[i*3], processed_images[i]);
        cv::imwrite("output/pre_output/" + window_names[i*3] + ".jpg", processed_images[i]);

        // 二值化
        cv::Mat binary;
        cv::threshold(processed_images[i], binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        cv::imshow(window_names[i*3 + 1], binary);
        cv::imwrite("output/pre_output/" + window_names[i*3 + 1] + ".jpg", binary);

        // Canny边缘检测
        cv::Mat edges;
        cv::Canny(processed_images[i], edges, 50, 150);
        cv::imshow(window_names[i*3 + 2], edges);
        cv::imwrite("output/pre_output/" + window_names[i*3 + 2] + ".jpg", edges);
    }

    std::cout << "处理完成。按任意键退出..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}