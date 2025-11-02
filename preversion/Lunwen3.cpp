#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

class GradientIntersect {
private:
    // 创建网格向量场
    cv::Mat createGrid(int rows, int cols) {
        cv::Mat grid(2 * rows - 1, 2 * cols - 1, CV_32FC2);

        for (int y = 1 - rows; y < rows; y++) {
            for (int x = 1 - cols; x < cols; x++) {
                float norm = std::sqrt(x * x + y * y);
                if (norm == 0) norm = 1;
                int grid_y = y + rows - 1;
                int grid_x = x + cols - 1;
                grid.at<cv::Vec2f>(grid_y, grid_x) = cv::Vec2f(y / norm, x / norm);
            }
        }
        return grid;
    }

    // 计算归一化梯度
    cv::Mat createGradient(const cv::Mat& image) {
        cv::Mat grad_x, grad_y;
        cv::Sobel(image, grad_x, CV_32F, 1, 0);
        cv::Sobel(image, grad_y, CV_32F, 0, 1);

        cv::Mat gradient(image.size(), CV_32FC2);
        for (int y = 0; y < image.rows; y++) {
            for (int x = 0; x < image.cols; x++) {
                float gx = grad_x.at<float>(y, x);
                float gy = grad_y.at<float>(y, x);
                float norm = std::sqrt(gx * gx + gy * gy);
                if (norm == 0) norm = 1;
                gradient.at<cv::Vec2f>(y, x) = cv::Vec2f(gy / norm, gx / norm);
            }
        }
        return gradient;
    }

public:
    cv::Point locate(const cv::Mat& image, double sigma = 2, int accuracy = 1) {
        // 转换图像为浮点型并归一化
        cv::Mat float_image;
        image.convertTo(float_image, CV_32F);
        cv::normalize(float_image, float_image, 0, 1, cv::NORM_MINMAX);

        // 高斯模糊
        cv::Mat blurred;
        cv::GaussianBlur(float_image, blurred, cv::Size(0, 0), sigma);

        // 获取网格和梯度
        cv::Mat grid = createGrid(image.rows, image.cols);
        cv::Mat gradient = createGradient(float_image);

        // 创建得分矩阵
        cv::Mat scores = cv::Mat::zeros(image.size(), CV_32F);

        // 计算每个像素的得分
        for (int cy = 0; cy < image.rows; cy += accuracy) {
            for (int cx = 0; cx < image.cols; cx += accuracy) {
                float score = 0;
                for (int y = 0; y < image.rows; y++) {
                    for (int x = 0; x < image.cols; x++) {
                        cv::Vec2f disp = grid.at<cv::Vec2f>(
                            image.rows - cy - 1 + y,
                            image.cols - cx - 1 + x
                        );
                        cv::Vec2f grad = gradient.at<cv::Vec2f>(y, x);
                        float dot = disp[0] * grad[0] + disp[1] * grad[1];
                        score += dot * dot;
                    }
                }
                scores.at<float>(cy, cx) = score * (1 - blurred.at<float>(cy, cx));
            }
        }

        // 找到得分最高的点
        cv::Point maxLoc;
        cv::minMaxLoc(scores, nullptr, nullptr, nullptr, &maxLoc);
        return maxLoc;
    }

    // 添加获取梯度图的方法
    cv::Mat getGradientImage(const cv::Mat& image) {
        cv::Mat float_image;
        image.convertTo(float_image, CV_32F);
        cv::normalize(float_image, float_image, 0, 1, cv::NORM_MINMAX);

        cv::Mat gradient = createGradient(float_image);

        // 转换梯度为可视化图像
        cv::Mat gradient_vis(image.size(), CV_8UC1);
        for (int y = 0; y < image.rows; y++) {
            for (int x = 0; x < image.cols; x++) {
                cv::Vec2f grad = gradient.at<cv::Vec2f>(y, x);
                float magnitude = std::sqrt(grad[0] * grad[0] + grad[1] * grad[1]);
                gradient_vis.at<uchar>(y, x) = cv::saturate_cast<uchar>(magnitude * 255);
            }
        }
        return gradient_vis;
    }
};

int main() {
    // 读取图像
    cv::Mat image = cv::imread("1.bmp", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Unable to load image!" << std::endl;
        return -1;
    }
    std::cout << "Image loaded successfully. Size: " << image.size() << std::endl;

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

    // 创建检测器
    GradientIntersect detector;

    // 处理左眼
    cv::Mat left_eye = image(eyes[0]);
    cv::Point left_pupil = detector.locate(left_eye);
    cv::Mat left_gradient = detector.getGradientImage(left_eye);

    // 处理右眼
    cv::Mat right_eye = image(eyes[1]);
    cv::Point right_pupil = detector.locate(right_eye);
    cv::Mat right_gradient = detector.getGradientImage(right_eye);

    // 计算原图中的瞳孔绝对坐标
    cv::Point left_pupil_abs(eyes[0].x + left_pupil.x, eyes[0].y + left_pupil.y);
    cv::Point right_pupil_abs(eyes[1].x + right_pupil.x, eyes[1].y + right_pupil.y);

    // 在原图上标记结果
    cv::Mat result = image.clone();

    // 标记左眼
    cv::circle(result, left_pupil_abs, 3, cv::Scalar(255), -1);
    cv::circle(result, left_pupil_abs, 10, cv::Scalar(255), 1);

    // 标记右眼
    cv::circle(result, right_pupil_abs, 3, cv::Scalar(255), -1);
    cv::circle(result, right_pupil_abs, 10, cv::Scalar(255), 1);

    // 创建并显示窗口
    cv::namedWindow("Original", cv::WINDOW_NORMAL);
    cv::namedWindow("Result", cv::WINDOW_NORMAL);
    cv::namedWindow("Left Eye", cv::WINDOW_NORMAL);
    cv::namedWindow("Right Eye", cv::WINDOW_NORMAL);
    cv::namedWindow("Left Gradient", cv::WINDOW_NORMAL);
    cv::namedWindow("Right Gradient", cv::WINDOW_NORMAL);

    // 设置窗口大小
    cv::resizeWindow("Original", 800, 600);
    cv::resizeWindow("Result", 800, 600);
    cv::resizeWindow("Left Eye", 400, 300);
    cv::resizeWindow("Right Eye", 400, 300);
    cv::resizeWindow("Left Gradient", 400, 300);
    cv::resizeWindow("Right Gradient", 400, 300);

    // 显示结果
    cv::imshow("Original", image);
    cv::imshow("Result", result);
    cv::imshow("Left Eye", left_eye);
    cv::imshow("Right Eye", right_eye);
    cv::imshow("Left Gradient", left_gradient);
    cv::imshow("Right Gradient", right_gradient);

    // 保存结果
    cv::imwrite("result1.jpg", result);
    //cv::imwrite("left_gradient.jpg", left_gradient);
    //cv::imwrite("right_gradient.jpg", right_gradient);

    std::cout << "Left pupil absolute position: (" << left_pupil_abs.x << ", " << left_pupil_abs.y << ")" << std::endl;
    std::cout << "Right pupil absolute position: (" << right_pupil_abs.x << ", " << right_pupil_abs.y << ")" << std::endl;
    std::cout << "Press any key to exit..." << std::endl;

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}