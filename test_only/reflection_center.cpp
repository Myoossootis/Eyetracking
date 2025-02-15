#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream>

void regionGrow(const cv::Mat& image, cv::Point seed, cv::Mat& result, int threshold = 10) {
    int rows = image.rows;
    int cols = image.cols;

    // 初始化结果图像为全零（即背景）
    result = cv::Mat::zeros(image.size(), CV_8UC1);

    std::vector<cv::Point> neighbors;
    neighbors.push_back(seed);

    // 结果图像的种子点设为1（代表已找到区域）
    result.at<uchar>(seed) = 255;

    // 创建一个队列，用来实现区域生长
    int currentIdx = 0;
    while (currentIdx < neighbors.size()) {
        cv::Point currentPoint = neighbors[currentIdx];
        currentIdx++;

        // 查看8个邻居点
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                if (dx == 0 && dy == 0) continue;

                int newX = currentPoint.x + dx;
                int newY = currentPoint.y + dy;

                // 确保点在图像范围内
                if (newX >= 0 && newX < cols && newY >= 0 && newY < rows) {
                    // 如果邻居点未被访问且灰度差值小于阈值
                    if (result.at<uchar>(newY, newX) == 0 && std::abs(image.at<uchar>(newY, newX) - image.at<uchar>(seed)) < threshold) {
                        result.at<uchar>(newY, newX) = 255; // 标记为已访问
                        neighbors.push_back(cv::Point(newX, newY)); // 加入待扩展的队列
                    }
                }
            }
        }
    }
}

int main() {
    // 加载明瞳图像和暗瞳图像
    cv::Mat image_dark = cv::imread("1.bmp", cv::IMREAD_GRAYSCALE);
    cv::Mat image_light = cv::imread("2.bmp", cv::IMREAD_GRAYSCALE);

    // 检查图像是否加载成功
    if (image_light.empty() || image_dark.empty()) {
        std::cerr << "Error: Unable to load images!" << std::endl;
        return -1;
    }

    // Step 1: 暗瞳图像减去明瞳图像，得到瞳孔位置
    cv::Mat pupil_position;
    cv::absdiff(image_dark, image_light, pupil_position);
    cv::normalize(pupil_position, pupil_position, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    // Step 2：gaussion滤波
    cv::Mat blurred_image_light;
    cv::GaussianBlur(image_light, blurred_image_light, cv::Size(5, 5), 0);

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
    std::ofstream outFile("test/pupil_centers.txt");

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
                outFile << "Center X: " << center.x << ", Center Y: " << center.y << std::endl;

                // Step 3: 在椭圆区域内寻找最亮点
                cv::Mat mask = cv::Mat::zeros(image_light.size(), CV_8UC1);
                cv::ellipse(mask, fitted_ellipse, cv::Scalar(255), -1);

                cv::Mat masked_image;
                image_light.copyTo(masked_image, mask);

                cv::Point brightest_point;
                double max_val;
                cv::minMaxLoc(masked_image, NULL, &max_val, NULL, &brightest_point, mask);

                // 使用区域生长算法扩展最亮点所在区域
                cv::Mat region_result;
                regionGrow(masked_image, brightest_point, region_result);

                // 绘制最亮点和区域生长结果
                cv::circle(result, brightest_point, 3, cv::Scalar(255, 0, 0), -1);  // 绘制最亮点
                result.setTo(cv::Scalar(255, 255, 0), region_result);  // 使用区域生长结果填充颜色
            }
        }
    }
    outFile.close();

    cv::imwrite("test/pupil_position.jpg", pupil_position);
    cv::imwrite("test/reflection_center.jpg", result);
    cv::imwrite("test/binary.jpg", binary);

    cv::imshow("3", result);
    cv::imshow("4", edges);
    cv::waitKey(0);
    return 0;
}
