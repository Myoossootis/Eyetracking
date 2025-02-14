#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <queue>
#include <iomanip>


int main() {
    // 加载明瞳图像和暗瞳图像
    cv::Mat image_dark = cv::imread("1.bmp", cv::IMREAD_GRAYSCALE);
    cv::Mat image_light = cv::imread("2.bmp", cv::IMREAD_GRAYSCALE);

    // 检查图像是否加载成功
    if (image_light.empty() || image_dark.empty()) {
        std::cerr << "Error: Unable to load images!" << std::endl;
        return -1;
    }

    // 暗瞳图像减去明瞳图像，得到瞳孔位置
    cv::Mat pupil_position;
    cv::absdiff(image_dark, image_light, pupil_position);
    cv::normalize(pupil_position, pupil_position, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    // 高斯滤波
    cv::Mat blurred_image_light;
    cv::GaussianBlur(image_light, blurred_image_light, cv::Size(5, 5), 0);

    // 阈值化处理
    cv::Mat binary;
    double threshold_value = 0; // 大津法会自动计算阈值
    cv::threshold(blurred_image_light, binary, threshold_value, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    cv::imshow("1", binary);

    // 边缘检测
    cv::Mat edges;
    cv::Canny(binary, edges, 50, 150);

    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Point> pupil_contour;
    if (!contours.empty()) {
        double max_area = 0;
        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);
            if (area > max_area) {
                max_area = area;
                pupil_contour = contour;
            }
        }
    }

    // 绘制椭圆
    cv::Mat result = image_light.clone();
    cv::Mat detected_blobs = image_light.clone();
    std::ofstream outFile("test/pupil_centers.txt");
    if (outFile.is_open()) {
        outFile << std::fixed << std::setprecision(3);  // 设置精度为3位小数
    }
    else {
        std::cerr << "Error opening file for writing!" << std::endl;
    }

    std::vector<cv::Point2f> centers;
    for (const auto& contour : contours) {
        if (contour.size() < 5) continue;  // 跳过点数不足5个的轮廓

        // 拟合椭圆
        cv::RotatedRect fitted_ellipse = cv::fitEllipse(contour);

        // 计算轮廓的面积和长宽比
        double area = cv::contourArea(contour);
        double aspectRatio = std::abs(fitted_ellipse.size.width / fitted_ellipse.size.height);

        // 根据面积和长宽比筛选合适的椭圆
        if (area > 100 && area < 1000 && aspectRatio > 0.5 && aspectRatio < 2.0) {
            // 获取椭圆的中心点
            cv::Point2f center = fitted_ellipse.center;

            // 去重：如果新点和已有点之间的距离小于阈值，则忽略
            bool isDuplicate = false;
            for (const auto& existing_center : centers) {
                if (cv::norm(center - existing_center) < 10) { // 假设10像素以内认为是重复
                    isDuplicate = true;
                    break;
                }
            }

            // 如果中心点不重复，则绘制并写入文件
            if (!isDuplicate) {
                centers.push_back(center);  // 添加到已检测中心点列表
                cv::ellipse(result, fitted_ellipse, cv::Scalar(0, 255, 0), 2);  // 绘制椭圆
                cv::circle(result, center, 2, cv::Scalar(0, 0, 255), -1);  // 绘制中心点

                // 写入文件
                outFile << "Pupil_Center X: " << center.x << ", Pupil_Center Y: " << center.y << std::endl;


                // 提取椭圆内的斑点
                cv::Mat mask_raw = cv::Mat::zeros(image_light.size(), CV_8UC1);
                cv::Mat mask;
                cv::equalizeHist(mask_raw, mask);
                cv::ellipse(mask, fitted_ellipse, cv::Scalar(255), -1);
                cv::Mat masked_image;
                image_light.copyTo(masked_image, mask);

                // 手动设置阈值，重新进行阈值分割
                double manual_threshold_value = 105; // 根据需要调整
                cv::Mat binary_masked;
                cv::threshold(masked_image, binary_masked, manual_threshold_value, 255, cv::THRESH_BINARY);
                cv::imshow("binarymasked", binary_masked);
                // 再次检测斑点
                std::vector<std::vector<cv::Point>> masked_contours;
                cv::findContours(binary_masked, masked_contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
                // 过滤并绘制斑点
                for (const auto& m_contour : masked_contours) {
                    double m_area = cv::contourArea(m_contour);
                    if (m_area > 3 && m_area < 100) {
                        cv::drawContours(detected_blobs, std::vector<std::vector<cv::Point>>{m_contour}, -1, cv::Scalar(255, 0, 0), 2);
                    }
                }
            }
        }
    }
    outFile.close();

    cv::imwrite("test/pupil_position.jpg", pupil_position);
    //cv::imwrite("test/reflection_center_area.jpg", result);
    cv::imwrite("test/binary.jpg", binary);
    cv::imshow("Blobs", detected_blobs);
    cv::imshow("3", result);
    cv::imshow("4", edges);
    cv::waitKey(0);
    return 0;
}
