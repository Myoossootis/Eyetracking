#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>

class EyeDetector {
public:
    static std::pair<cv::Rect, cv::Rect> detectEyeRegions(const cv::Mat& image_gray) {
        cv::CascadeClassifier eye_cascade;
        if (!eye_cascade.load("haarcascades/haarcascade_eye.xml")) {
            throw std::runtime_error("无法加载眼睛检测器");
        }

        std::vector<cv::Rect> eyes;
        eye_cascade.detectMultiScale(image_gray, eyes, 1.1, 3, 0, cv::Size(30, 30));

        if (eyes.size() >= 2) {
            std::sort(eyes.begin(), eyes.end(), [](const cv::Rect& a, const cv::Rect& b) {
                return a.x < b.x;
                });
            return std::make_pair(eyes[0], eyes[1]);
        }

        throw std::runtime_error("未检测到两个以上的眼睛区域");
    }
};

int main() {
    cv::Mat image_color = cv::imread("1.bmp", cv::IMREAD_COLOR);
    if (image_color.empty()) {
        std::cerr << "错误：无法加载图像！" << std::endl;
        return -1;
    }

    cv::Mat image_gray;
    cv::cvtColor(image_color, image_gray, cv::COLOR_BGR2GRAY);

    try {
        std::pair<cv::Rect, cv::Rect> eye_regions = EyeDetector::detectEyeRegions(image_gray);
        cv::Rect left_eye_rect = eye_regions.first;
        cv::Rect right_eye_rect = eye_regions.second;

        cv::Mat left_eye = image_gray(left_eye_rect);
        cv::Mat right_eye = image_gray(right_eye_rect);

        // 彩色版本用于画红点
        cv::Mat left_eye_color, right_eye_color;
        cv::cvtColor(left_eye, left_eye_color, cv::COLOR_GRAY2BGR);
        cv::cvtColor(right_eye, right_eye_color, cv::COLOR_GRAY2BGR);

        int max_filter_size = 5;
        int median_filter_size = 3;

        max_filter_size = max_filter_size % 2 == 0 ? max_filter_size + 1 : max_filter_size;
        median_filter_size = median_filter_size % 2 == 0 ? median_filter_size + 1 : median_filter_size;

        cv::Mat max_filtered_left, median_filtered_left, result_left;
        cv::dilate(left_eye, max_filtered_left,
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(max_filter_size, max_filter_size)));
        cv::medianBlur(left_eye, median_filtered_left, median_filter_size);
        cv::subtract(max_filtered_left, median_filtered_left, result_left);

        cv::Mat max_filtered_right, median_filtered_right, result_right;
        cv::dilate(right_eye, max_filtered_right,
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(max_filter_size, max_filter_size)));
        cv::medianBlur(right_eye, median_filtered_right, median_filter_size);
        cv::subtract(max_filtered_right, median_filtered_right, result_right);

        auto extractAndDraw = [](const cv::Mat& result, cv::Mat& drawImage, std::ofstream& outFile, const std::string& label) {
            cv::Mat binary;
            cv::threshold(result, binary, 50, 255, cv::THRESH_BINARY);

            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            if (contours.empty()) {
                std::cerr << label << " 未检测到亮点！" << std::endl;
                return;
            }

            auto largest_contour = *std::max_element(contours.begin(), contours.end(), [](const auto& a, const auto& b) {
                return cv::contourArea(a) < cv::contourArea(b);
                });

            cv::Moments m = cv::moments(largest_contour);
            if (m.m00 == 0) {
                std::cerr << label << " 亮点中心计算失败！" << std::endl;
                return;
            }

            int cx = static_cast<int>(m.m10 / m.m00);
            int cy = static_cast<int>(m.m01 / m.m00);

            // 在该局部眼睛图像上画红点
            cv::circle(drawImage, cv::Point(cx, cy), 3, cv::Scalar(0, 0, 255), -1);
            //outFile << label << " Eye Highlight Center in Local Image: (" << cx << ", " << cy << ")" << std::endl;
            };

        std::ofstream outFile("output/highlight_centers_local.txt");
        if (!outFile.is_open()) {
            std::cerr << "错误：无法打开输出文件！" << std::endl;
            return -1;
        }

        extractAndDraw(result_left, left_eye_color, outFile, "Left");
        extractAndDraw(result_right, right_eye_color, outFile, "Right");

        outFile.close();

        // 显示处理后的左右眼图像
        cv::imshow("Left Eye with Highlight", left_eye_color);
        cv::imshow("Right Eye with Highlight", right_eye_color);
        cv::imwrite("reflection3.png", left_eye_color);
        cv::imwrite("reflection4.png", right_eye_color);
        cv::waitKey(0);
        cv::destroyAllWindows();
        return 0;

    }
    catch (const std::exception& e) {
        std::cerr << "错误：" << e.what() << std::endl;
        return -1;
    }
}
