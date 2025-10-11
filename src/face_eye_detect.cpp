#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

class FaceEyeDetector {
public:
    static std::pair<cv::Rect, cv::Rect> detectEyeRegions(const cv::Mat& image) {
        // 加载检测器
        cv::CascadeClassifier eye_cascade;
        cv::CascadeClassifier face_cascade_front;
        cv::CascadeClassifier face_cascade_profile;

        // 检查所有检测器是否正确加载
        if (!eye_cascade.load("haarcascades/haarcascade_eye.xml")) {
            throw std::runtime_error("Failed to load eye detector");
        }
        if (!face_cascade_front.load("haarcascades/haarcascade_frontalface_alt.xml")) {
            throw std::runtime_error("Failed to load front face detector");
        }
        if (!face_cascade_profile.load("haarcascades/haarcascade_profileface.xml")) {
            throw std::runtime_error("Failed to load profile face detector");
        }

        // 首先尝试正面人脸检测
        std::vector<cv::Rect> faces_front;
        face_cascade_front.detectMultiScale(image, faces_front, 1.1, 3, 0, cv::Size(150, 150));
        
        if (!faces_front.empty()) {
            std::cout << "front success" << std::endl;
            return extractEyeRegionsFromFace(image, faces_front[0]);
        } else {
            std::cout << "front failed" << std::endl;
        }

        // 尝试侧面人脸检测
        std::vector<cv::Rect> faces_profile;
        face_cascade_profile.detectMultiScale(image, faces_profile, 1.1, 3, 0, cv::Size(150, 150));
        
        if (!faces_profile.empty()) {
            std::cout << "profile success" << std::endl;
            return extractEyeRegionsFromFace(image, faces_profile[0]);
        } else {
            std::cout << "profile failed" << std::endl;
        }

        // 人脸检测失败，尝试直接检测眼睛
        std::vector<cv::Rect> eyes;
        // 调整参数以提高精确度：
        // - scaleFactor 改为 1.05 使尺度变化更细腻
        // - minNeighbors 增加到 6 减少误检
        // - flags 保持为 0
        // - minSize 40以避免小区域误检
        eye_cascade.detectMultiScale(image, eyes, 1.05, 6, 0, cv::Size(40, 40));

        if (eyes.size() >= 2) {
            // 按x坐标排序
            std::sort(eyes.begin(), eyes.end(),
                [](const cv::Rect& a, const cv::Rect& b) { return a.x < b.x; });

            // 添加额外的验证步骤
            std::vector<cv::Rect> validated_eyes;
            for (const auto& eye : eyes) {
                // 验证宽高比
                float aspect_ratio = static_cast<float>(eye.width) / eye.height;
                if (aspect_ratio < 0.4 || aspect_ratio > 2.5) continue;

                // 验证区域大小相对于图像
                float relative_size = static_cast<float>(eye.area()) / (image.rows * image.cols);
                if (relative_size < 0.01 || relative_size > 0.15) continue;

                validated_eyes.push_back(eye);
            }

            // 验证两个眼睛的相对位置和大小
            if (validated_eyes.size() >= 2) {
                const cv::Rect& left = validated_eyes[0];
                const cv::Rect& right = validated_eyes[1];

                // 验证水平距离
                float distance = static_cast<float>(right.x - (left.x + left.width));
                if (distance < 0) {
                    std::cout << "Eye positions overlap, likely false detection" << std::endl;
                    throw std::runtime_error("No valid eyes detected");
                }

                // 验证大小相似性
                float size_ratio = static_cast<float>(left.area()) / right.area();
                if (size_ratio < 0.5 || size_ratio > 2.0) {
                    std::cout << "Eye sizes too different, likely false detection" << std::endl;
                    throw std::runtime_error("No valid eyes detected");
                }

                // 验证垂直位置相似性
                float vertical_diff = std::abs(left.y - right.y);
                if (vertical_diff > left.height) {
                    std::cout << "Eyes not horizontally aligned, likely false detection" << std::endl;
                    throw std::runtime_error("No valid eyes detected");
                }

                std::cout << "Eye detection validated successfully" << std::endl;
                return std::make_pair(validated_eyes[0], validated_eyes[1]);
            }
        }

        std::cout << "Eye detection failed or validation failed" << std::endl;
        throw std::runtime_error("No valid eyes detected");

        // 所有检测都失败
        throw std::runtime_error("No eyes or face detected");
    }

private:
    static std::pair<cv::Rect, cv::Rect> extractEyeRegionsFromFace(const cv::Mat& image, const cv::Rect& face) {
        int eye_region_height = face.height / 3;
        int eye_region_width = face.width / 3;  // 修改为三分之一宽度
        int eye_region_top = face.y + face.height / 4;

        // 修改眼睛区域的计算方式
        cv::Rect left_eye_rect(
            face.x + face.width / 6,  // 修改起始位置
            eye_region_top,
            eye_region_width,         // 使用固定宽度
            eye_region_height
        );

        cv::Rect right_eye_rect(
            face.x + face.width / 2,  // 右眼起始位置
            eye_region_top,
            eye_region_width,         // 使用固定宽度
            eye_region_height
        );

        // 确保矩形在图像范围内
        left_eye_rect &= cv::Rect(0, 0, image.cols, image.rows);
        right_eye_rect &= cv::Rect(0, 0, image.cols, image.rows);

        return std::make_pair(left_eye_rect, right_eye_rect);
    }
}; 
// 添加 GradientIntersect 类
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
        cv::Mat float_image;
        image.convertTo(float_image, CV_32F);
        cv::normalize(float_image, float_image, 0, 1, cv::NORM_MINMAX);

        cv::Mat blurred;
        cv::GaussianBlur(float_image, blurred, cv::Size(0, 0), sigma);

        // 限制搜索范围到中心区域
        int border = 5;  // 边界像素
        int start_y = border;
        int end_y = image.rows - border;
        int start_x = border;
        int end_x = image.cols - border;

        cv::Mat grid = createGrid(image.rows, image.cols);
        cv::Mat gradient = createGradient(float_image);

        cv::Mat scores = cv::Mat::zeros(image.size(), CV_32F);

        // 使用OpenMP并行计算
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int cy = start_y; cy < end_y; cy += accuracy) {
            for (int cx = start_x; cx < end_x; cx += accuracy) {
                float score = 0;
                float blur_val = blurred.at<float>(cy, cx);
                
                int window_size = 10;
                int start_wy = std::max(0, cy - window_size);
                int end_wy = std::min(image.rows, cy + window_size);
                int start_wx = std::max(0, cx - window_size);
                int end_wx = std::min(image.cols, cx + window_size);

                for (int y = start_wy; y < end_wy; y++) {
                    for (int x = start_wx; x < end_wx; x++) {
                        cv::Vec2f disp = grid.at<cv::Vec2f>(
                            image.rows - cy - 1 + y,
                            image.cols - cx - 1 + x
                        );
                        cv::Vec2f grad = gradient.at<cv::Vec2f>(y, x);
                        float dot = disp[0] * grad[0] + disp[1] * grad[1];
                        score += dot * dot;
                    }
                }
                scores.at<float>(cy, cx) = score * (1 - blur_val);
            }
        }

        cv::Point maxLoc;
        cv::minMaxLoc(scores, nullptr, nullptr, nullptr, &maxLoc);
        return maxLoc;
    }

    cv::Mat getGradientImage(const cv::Mat& image) {
        cv::Mat float_image;
        image.convertTo(float_image, CV_32F);
        cv::normalize(float_image, float_image, 0, 1, cv::NORM_MINMAX);
        
        cv::Mat gradient = createGradient(float_image);
        
        cv::Mat gradient_vis(image.size(), CV_8UC1);
        for (int y = 0; y < image.rows; y+=2) {
            for (int x = 0; x < image.cols; x+=2) {
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
    cv::Mat image = cv::imread("26.bmp", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Unable to load image!" << std::endl;
        return -1;
    }
    std::cout << "Image loaded successfully. Size: " << image.size() << std::endl;

    try {
        // 使用检测器提取眼睛区域
        std::pair<cv::Rect, cv::Rect> eye_regions = FaceEyeDetector::detectEyeRegions(image);
        cv::Rect left_eye_rect = eye_regions.first;
        cv::Rect right_eye_rect = eye_regions.second;
        
        // 提取眼睛图像
        cv::Mat left_eye = image(left_eye_rect);
        cv::Mat right_eye = image(right_eye_rect);

        // 创建瞳孔检测器
        GradientIntersect detector;

        // 处理左眼
        cv::Point left_pupil = detector.locate(left_eye);
        cv::Mat left_gradient = detector.getGradientImage(left_eye);
        
        // 处理右眼
        cv::Point right_pupil = detector.locate(right_eye);
        cv::Mat right_gradient = detector.getGradientImage(right_eye);

        // 计算原图中的瞳孔绝对坐标
        cv::Point left_pupil_abs(left_eye_rect.x + left_pupil.x, left_eye_rect.y + left_pupil.y);
        cv::Point right_pupil_abs(right_eye_rect.x + right_pupil.x, right_eye_rect.y + right_pupil.y);

        // 在原图上标记结果
        cv::Mat result = image.clone();
        
        // 标记眼睛区域
        cv::rectangle(result, left_eye_rect, cv::Scalar(255), 1);
        cv::rectangle(result, right_eye_rect, cv::Scalar(255), 1);
        
        // 标记瞳孔位置
        cv::circle(result, left_pupil_abs, 3, cv::Scalar(255), -1);
        cv::circle(result, left_pupil_abs, 10, cv::Scalar(255), 1);
        cv::circle(result, right_pupil_abs, 3, cv::Scalar(255), -1);
        cv::circle(result, right_pupil_abs, 10, cv::Scalar(255), 1);

        // 显示结果
        cv::namedWindow("Original", cv::WINDOW_NORMAL);
        cv::namedWindow("Result", cv::WINDOW_NORMAL);
        cv::namedWindow("Left Eye", cv::WINDOW_NORMAL);
        cv::namedWindow("Right Eye", cv::WINDOW_NORMAL);
        cv::namedWindow("Left Gradient", cv::WINDOW_NORMAL);
        cv::namedWindow("Right Gradient", cv::WINDOW_NORMAL);

        cv::resizeWindow("Original", 800, 600);
        cv::resizeWindow("Result", 800, 600);
        cv::resizeWindow("Left Eye", 400, 300);
        cv::resizeWindow("Right Eye", 400, 300);
        cv::resizeWindow("Left Gradient", 400, 300);
        cv::resizeWindow("Right Gradient", 400, 300);

        cv::imshow("Original", image);
        cv::imshow("Result", result);
        cv::imshow("Left Eye", left_eye);
        cv::imshow("Right Eye", right_eye);
        cv::imshow("Left Gradient", left_gradient);
        cv::imshow("Right Gradient", right_gradient);

        // 输出瞳孔坐标
        std::cout << "Left pupil absolute position: (" << left_pupil_abs.x << ", " << left_pupil_abs.y << ")" << std::endl;
        std::cout << "Right pupil absolute position: (" << right_pupil_abs.x << ", " << right_pupil_abs.y << ")" << std::endl;
        std::cout << "Press any key to exit..." << std::endl;
        
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}