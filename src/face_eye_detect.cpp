#include <opencv2/opencv.hpp>
#include <iostream>

/**
 * @brief 在图像中检测人脸和眼睛，并在检测到的人脸和眼睛上绘制矩形框。
 *
 * @param image 输入图像，函数将直接在此图像上进行绘制。
 * @param face_cascade 用于人脸检测的Haar级联分类器。
 * @param eyes_cascade 用于眼睛检测的Haar级联分类器。
 */
void detectAndDisplay(cv::Mat& image, cv::CascadeClassifier& face_cascade, cv::CascadeClassifier& eyes_cascade) {
    cv::Mat frame_gray;
    cv::cvtColor(image, frame_gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(frame_gray, frame_gray); // 直方图均衡化以提高对比度

    // -- 检测人脸
    std::vector<cv::Rect> faces;
    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));

    for (size_t i = 0; i < faces.size(); i++) {
        // 在人脸周围绘制绿色矩形
        cv::rectangle(image, faces[i], cv::Scalar(0, 255, 0), 2);

        // 在每个人脸区域内检测眼睛
        cv::Mat faceROI = frame_gray(faces[i]);
        std::vector<cv::Rect> eyes;
        eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));

        for (size_t j = 0; j < eyes.size(); j++) {
            // 眼睛坐标是相对于人脸ROI的，需要转换回原始图像坐标
            cv::Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
            int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
            // 在眼睛中心绘制蓝色圆形
            cv::circle(image, eye_center, radius, cv::Scalar(255, 0, 0), 3);
        }
    }
}

int main(int argc, char** argv) {
    // 加载级联分类器
    cv::CascadeClassifier face_cascade, eyes_cascade;
    if (!face_cascade.load("haarcascades/haarcascade_frontalface_alt.xml")) {
        std::cerr << "Error: Could not load face cascade classifier." << std::endl;
        return -1;
    }
    if (!eyes_cascade.load("haarcascades/haarcascade_eye_tree_eyeglasses.xml")) {
        std::cerr << "Error: Could not load eyes cascade classifier." << std::endl;
        return -1;
    }

    // 从摄像头或视频文件读取
    cv::VideoCapture capture;
    capture.open(0); // 打开默认摄像头
    if (!capture.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (capture.read(frame)) {
        if (frame.empty()) {
            std::cerr << "No captured frame -- Break!" << std::endl;
            break;
        }

        // 应用检测函数
        detectAndDisplay(frame, face_cascade, eyes_cascade);

        // 显示结果
        cv::imshow("Face and Eye Detection", frame);

        // 按 'q' 键退出
        if (cv::waitKey(10) == 'q') {
            break;
        }
    }
    return 0;
}