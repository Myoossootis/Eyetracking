#include <opencv2/opencv.hpp>
#include <iostream>
#include<vector>
#include<fstream>
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
    //边缘检测
    cv::Mat edges;
    cv::Canny(binary, edges, 50, 150);
    //查找轮廓
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
    std::ofstream outFile("test/pupil_centers.txt"); // 创建文件流    


    std::vector<cv::Point2f> centers;
    // 遍历所有轮廓

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
            }
        }
    }
    outFile.close();

    //erosion_size = 4; // 腐蚀操作的大小
    //int dilation_size = 4; // 膨胀操作的大小
    //cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1));
    //cv::Mat opened;
    //cv::morphologyEx(pupil_position, opened, cv::MORPH_OPEN, element);

    //// Step 3: laplacion增强
    //cv::Mat laplacian;
    //cv::Laplacian(blurred_pupil_position, laplacian, CV_16S, 3); // 使用3x3的拉普拉斯核
    //cv::convertScaleAbs(laplacian, laplacian); // 转换为8位无符号整数
    // 显示结果
    //cv::imshow("pupil position", pupil_position);
    //cv::imshow("2",blurred_pupil_position);

    
    //cv::imwrite("test/pupil_position.jpg", pupil_position);
    //cv::imwrite("test/pupil_center.jpg", result);
    //cv::imwrite("test/binary.jpg", binary);
   /* cv::imwrite("test/log_result.jpg",laplacian);*/
    //cv::imwrite("test/gaussion.jpg", blurred_pupil_position);
    cv::imshow("3", result);
    cv::imshow("4",edges);
    cv::waitKey(0);
    return 0;
}