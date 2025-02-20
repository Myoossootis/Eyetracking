#include "detection.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

// Í«¿×¼ì²â¹¦ÄÜ
void detect_pupil(const std::string& light_image_path, const std::string& dark_image_path, const std::string& output_file) {
    cv::Mat image_light = cv::imread(light_image_path, cv::IMREAD_GRAYSCALE);
    cv::Mat image_dark = cv::imread(dark_image_path, cv::IMREAD_GRAYSCALE);

    if (image_light.empty() || image_dark.empty()) {
        std::cerr << "Error: Unable to load images!" << std::endl;
        return;
    }

    cv::CascadeClassifier eye_cascade;
    if (!eye_cascade.load("haarcascades/haarcascade_eye.xml")) {
        std::cerr << "Error: Unable to load eye cascade classifier!" << std::endl;
        return;
    }

    std::vector<cv::Rect> eyes;
    eye_cascade.detectMultiScale(image_light, eyes, 1.1, 4, 0, cv::Size(30, 30));

    if (eyes.empty()) {
        std::cerr << "Error: No eyes detected!" << std::endl;
        return;
    }

    std::sort(eyes.begin(), eyes.end(), [](const cv::Rect& a, const cv::Rect& b) {
        return a.x < b.x;
        });

    std::ofstream outFile(output_file);
    if (!outFile.is_open()) {
        std::cerr << "Error: Unable to open output file!" << std::endl;
        return;
    }

    for (const auto& eye : eyes) {
        cv::Mat eye_light_region = image_light(eye);
        cv::Mat eye_dark_region = image_dark(eye);

        cv::Mat eye_pupil_position;
        cv::absdiff(eye_light_region, eye_dark_region, eye_pupil_position);

        cv::Mat blurred_eye;
        cv::GaussianBlur(eye_pupil_position, blurred_eye, cv::Size(5, 5), 1.5);

        cv::Mat edges;
        cv::Canny(blurred_eye, edges, 50, 150);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (const auto& contour : contours) {
            if (contour.size() >= 5) {
                cv::RotatedRect ellipse = cv::fitEllipse(contour);
                cv::Point2f center = ellipse.center;

                center.x += eye.x;
                center.y += eye.y;

                outFile << "Ellipse Center: (" << center.x << ", " << center.y << ")" << std::endl;
            }
        }
    }

    outFile.close();
    cv::imshow("Pupil Position", image_light);
    cv::waitKey(0);
}
