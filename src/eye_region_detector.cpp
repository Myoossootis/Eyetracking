#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

class EyeRegionDetector {
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
                    throw std::runtime_error("Eye positions overlap, likely false detection");
                }

                // 验证大小相似性
                float size_ratio = static_cast<float>(left.area()) / right.area();
                if (size_ratio < 0.5 || size_ratio > 2.0) {
                    throw std::runtime_error("Eye sizes too different, likely false detection");
                }

                // 验证垂直位置相似性
                float vertical_diff = std::abs(left.y - right.y);
                if (vertical_diff > left.height) {
                    throw std::runtime_error("Eyes not horizontally aligned, likely false detection");
                }

                return std::make_pair(validated_eyes[0], validated_eyes[1]);
            }
        }

        throw std::runtime_error("No valid eyes detected");
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

int main() {
    // 读取图像
    cv::Mat image = cv::imread("18.png", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Unable to load image!" << std::endl;
        return -1;
    }

    try {
        // 使用检测器提取眼睛区域
        std::pair<cv::Rect, cv::Rect> eye_regions = EyeRegionDetector::detectEyeRegions(image);
        cv::Rect left_eye_rect = eye_regions.first;
        cv::Rect right_eye_rect = eye_regions.second;
        
        // 提取眼睛图像
        cv::Mat left_eye = image(left_eye_rect);
        cv::Mat right_eye = image(right_eye_rect);

        // 在原图上标记结果
        cv::Mat result = image.clone();
        
        // 标记眼睛区域
        cv::rectangle(result, left_eye_rect, cv::Scalar(255), 1);
        cv::rectangle(result, right_eye_rect, cv::Scalar(255), 1);

        // 显示结果
        cv::namedWindow("Original", cv::WINDOW_NORMAL);
        cv::namedWindow("Result", cv::WINDOW_NORMAL);
        cv::namedWindow("Left Eye", cv::WINDOW_NORMAL);
        cv::namedWindow("Right Eye", cv::WINDOW_NORMAL);

        cv::resizeWindow("Original", 800, 600);
        cv::resizeWindow("Result", 800, 600);
        cv::resizeWindow("Left Eye", 400, 300);
        cv::resizeWindow("Right Eye", 400, 300);

        cv::imshow("Original", image);
        cv::imshow("Result", result);
        cv::imshow("Left Eye", left_eye);
        cv::imshow("Right Eye", right_eye);
        cv::imwrite("region4.png", result);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}