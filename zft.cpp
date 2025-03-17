#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
// 函数：生成灰度直方图并显示
void showGrayHistogram(const cv::Mat& image, const std::string& windowName) {
    // 检查图像是否为灰度图
    if (image.channels() != 1) {
        std::cerr << "Error: Image must be grayscale." << std::endl;
        return;
    }

    // 直方图参数
    const int histSize = 256; // 灰度范围 [0, 255]
    float range[] = { 0, 256 }; // 灰度值范围
    const float* histRange = { range };

    // 计算直方图
    cv::Mat hist;
    int channels[] = { 0 }; // 灰度图只有一个通道
    cv::calcHist(&image, 1, channels, cv::Mat(), hist, 1, &histSize, &histRange);

    // 归一化直方图
    double maxVal = 0;
    cv::minMaxLoc(hist, nullptr, &maxVal);
    hist.convertTo(hist, -1, 255.0 / maxVal);

    // 绘制直方图
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);
    cv::Mat histImage(hist_h, hist_w, CV_8UC1, cv::Scalar(0));

    for (int i = 0; i < histSize; i++) {
        cv::line(histImage, cv::Point(bin_w * i, hist_h),
            cv::Point(bin_w * (i + 1), hist_h - cvRound(hist.at<float>(i))),
            cv::Scalar(255), 1, cv::LINE_8, 0);
    }
    for (int i = 0; i <= histSize; i += 32) { // 每32个灰度值绘制一个刻度
        int x = (i * bin_w);
        cv::line(histImage, cv::Point(x, hist_h - 10), cv::Point(x, hist_h), cv::Scalar(255), 1);
        cv::putText(histImage, std::to_string(i), cv::Point(x, hist_h + 20),
            cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255), 1);
    }

    // 添加坐标轴标签
    cv::putText(histImage, "Gray Level", cv::Point(hist_w / 2 - 50, hist_h + 40),
        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255), 1);
    cv::putText(histImage, "Frequency", cv::Point(0, hist_h / 2),
        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255), 1, cv::LINE_AA, true);
    // 显示直方图
    cv::imshow(windowName, histImage);
}
int main() {
    // 加载图像
    cv::Mat image = cv::imread("1.bmp", cv::IMREAD_GRAYSCALE);
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
    if (eyes.size() != 2) {
        std::cerr << "Error: Exactly two eyes are required for this operation!" << std::endl;
        return -1;
    }

    // 按x坐标排序，确保左眼在前，右眼在后
    std::sort(eyes.begin(), eyes.end(), [](const cv::Rect& a, const cv::Rect& b) {
        return a.x < b.x;
        });

    
    // 处理左眼
    cv::Mat left_eye = image(eyes[0]);
    
    // 梯度检测
    cv::Mat left_grad_x, left_grad_y, left_gradient;
    cv::Sobel(left_eye, left_grad_x, CV_32F, 1, 0);
    cv::Sobel(left_eye, left_grad_y, CV_32F, 0, 1);
    cv::magnitude(left_grad_x, left_grad_y, left_gradient);
    cv::normalize(left_gradient, left_gradient, 0, 255, cv::NORM_MINMAX);
    left_gradient.convertTo(left_gradient, CV_8U);

    // 形态学处理增强边缘
    cv::Mat left_morph;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(left_gradient, left_morph, cv::MORPH_CLOSE, kernel);

    // 二值化处理
    cv::Mat left_binary;
    cv::threshold(left_morph, left_binary, 50, 255, cv::THRESH_BINARY);

    // 寻找轮廓
    std::vector<std::vector<cv::Point>> left_contours;
    cv::findContours(left_binary, left_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 在原图上绘制检测到的瞳孔
    cv::Mat left_result = left_eye.clone();
    for (const auto& contour : left_contours) {
        double area = cv::contourArea(contour);
        if (area > 100 && area < 1000) {  // 面积阈值需要根据实际情况调整
            cv::RotatedRect ellipse = cv::fitEllipse(contour);
            double ratio = ellipse.size.width / ellipse.size.height;
            if (ratio > 0.8 && ratio < 1.2) {  // 近似圆形
                cv::ellipse(left_result, ellipse, cv::Scalar(255), 2);
            }
        }
    }

    // 创建窗口显示结果
    cv::namedWindow("left_gradient", cv::WINDOW_NORMAL);
    cv::namedWindow("left_morph", cv::WINDOW_NORMAL);
    cv::namedWindow("left_binary", cv::WINDOW_NORMAL);
    cv::namedWindow("left_result", cv::WINDOW_NORMAL);

    cv::imshow("left_gradient", left_gradient);
    cv::imshow("left_morph", left_morph);
    cv::imshow("left_binary", left_binary);
    cv::imshow("left_result", left_result);

    // 处理右眼（类似处理）
    cv::Mat right_eye = image(eyes[1]);
    
    // 梯度检测
    cv::Mat right_grad_x, right_grad_y, right_gradient;
    cv::Sobel(right_eye, right_grad_x, CV_32F, 1, 0);
    cv::Sobel(right_eye, right_grad_y, CV_32F, 0, 1);
    cv::magnitude(right_grad_x, right_grad_y, right_gradient);
    cv::normalize(right_gradient, right_gradient, 0, 255, cv::NORM_MINMAX);
    right_gradient.convertTo(right_gradient, CV_8U);

    // 形态学处理增强边缘
    cv::Mat right_morph;
    cv::morphologyEx(right_gradient, right_morph, cv::MORPH_CLOSE, kernel);

    // 二值化处理
    cv::Mat right_binary;
    cv::threshold(right_morph, right_binary, 50, 255, cv::THRESH_BINARY);

    // 寻找轮廓
    std::vector<std::vector<cv::Point>> right_contours;
    cv::findContours(right_binary, right_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 在原图上绘制检测到的瞳孔
    cv::Mat right_result = right_eye.clone();
    for (const auto& contour : right_contours) {
        double area = cv::contourArea(contour);
        if (area > 100 && area < 1000) {
            cv::RotatedRect ellipse = cv::fitEllipse(contour);
            double ratio = ellipse.size.width / ellipse.size.height;
            if (ratio > 0.8 && ratio < 1.2) {
                cv::ellipse(right_result, ellipse, cv::Scalar(255), 2);
            }
        }
    }

    // 创建窗口显示结果
    cv::namedWindow("right_gradient", cv::WINDOW_NORMAL);
    cv::namedWindow("right_morph", cv::WINDOW_NORMAL);
    cv::namedWindow("right_binary", cv::WINDOW_NORMAL);
    cv::namedWindow("right_result", cv::WINDOW_NORMAL);

    cv::imshow("right_gradient", right_gradient);
    cv::imshow("right_morph", right_morph);
    cv::imshow("right_binary", right_binary);
    cv::imshow("right_result", right_result);
    cv::Mat left_eye_inverted;
    cv::bitwise_not(left_eye, left_eye_inverted);

    
    // 创建可调整大小的窗口
    cv::namedWindow("left_eye", cv::WINDOW_NORMAL);
    cv::namedWindow("left_eye_inverted", cv::WINDOW_NORMAL);
    cv::namedWindow("left_binary", cv::WINDOW_NORMAL);
    cv::namedWindow("Left Eye Inverted Histogram", cv::WINDOW_NORMAL);
    
    cv::imshow("left_eye", left_eye);
    cv::imshow("left_eye_inverted", left_eye_inverted);

    showGrayHistogram(left_eye_inverted, "Left Eye Inverted Histogram");

    // 处理右眼

    cv::Mat right_eye_inverted;
    cv::bitwise_not(right_eye, right_eye_inverted);

    
    // 创建可调整大小的窗口
    cv::namedWindow("right_eye", cv::WINDOW_NORMAL);
    cv::namedWindow("right_eye_inverted", cv::WINDOW_NORMAL);

    cv::namedWindow("Right Eye Inverted Histogram", cv::WINDOW_NORMAL);
    
    cv::imshow("right_eye", right_eye);
    cv::imshow("right_eye_inverted", right_eye_inverted);

    showGrayHistogram(right_eye_inverted, "Right Eye Inverted Histogram");

    // 创建可调整大小的主窗口
    cv::namedWindow("Detected Eyes with Ellipses", cv::WINDOW_NORMAL);

    cv::waitKey(0);
    return 0;
}
