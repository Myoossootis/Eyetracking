#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 打开视频文件
    cv::VideoCapture cap("video.mp4");  // 替换成你的视频文件路径
    if (!cap.isOpened()) {
        std::cerr << "Error: Couldn't open video file!" << std::endl;
        return -1;
    }

    std::cout << "Video file opened successfully!" << std::endl;

    cv::Mat oddFrame, evenFrame;
    cv::Mat resultFrame;
    cv::Mat templateFrame;
    cv::Mat maskFrame;
    int frameCount = 0;
    bool success;
    cv::Mat frame;
    double fps = cap.get(cv::CAP_PROP_FPS);
    std::cout << "FPS:" << fps << std::endl;
    while (true) {
        success = cap.read(frame);
        if (!success || frame.empty()) {
            std::cout << "Reached end of video or empty frame." << std::endl;
            break;
        }
        if (frameCount % 2 == 0) {
            evenFrame = frame.clone(); // 存储偶数帧
        }
        else {
            oddFrame = frame.clone(); // 存储奇数帧
        }
        frameCount++;
        if (frameCount >= 2) {
            break; // 我们只需要一对奇数帧和偶数帧，因此读取两帧后就停止
        }
    }
    if (oddFrame.empty() || evenFrame.empty()) {
        std::cout << "Not enough frames to perform subtraction." << std::endl;
        return -1;
    }
    templateFrame = oddFrame - evenFrame  ; // 奇数帧减去偶数帧
    maskFrame = evenFrame - templateFrame;
    resultFrame = oddFrame - maskFrame  ;
    cv::imshow("even Frame", evenFrame);
    cv::imshow("odd Frame", oddFrame);
    cv::imshow("Template Frame", templateFrame); // 显示 templateFrame
    cv::imshow("Mask Frame", maskFrame); // 显示 maskFrame
    cv::imshow("Result Frame", resultFrame); // 显示 resultFrame

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}