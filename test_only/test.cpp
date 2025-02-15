#include <opencv2/opencv.hpp>
#include <iostream>

//cv::Mat homomorphicFilter(const cv::Mat& src) {
//    cv::Mat img;
//    src.convertTo(img, CV_32F);
//    cv::Mat logImage;
//    cv::log(img + 1, logImage);
//
//    cv::Mat dftImage;
//    cv::dft(logImage, dftImage, cv::DFT_COMPLEX_OUTPUT);
//
//    // 创建同态滤波器
//    cv::Mat filter = cv::Mat(dftImage.size(), CV_32FC2, cv::Scalar(0));
//    cv::Point center(filter.cols / 2, filter.rows / 2);
//    double D0 = 30; // 截止频率
//    double H = 1.0;
//    double L = 2.0;
//
//    for (int i = 0; i < filter.rows; ++i) {
//        for (int j = 0; j < filter.cols; ++j) {
//            double D = cv::norm(cv::Point(j, i) - center);
//            double Huv = (H - L) * (1 - exp(-D * D / (2 * D0 * D0))) + L;
//            filter.at<cv::Vec2f>(i, j)[0] = Huv;
//            filter.at<cv::Vec2f>(i, j)[1] = Huv;
//        }
//    }
//
//    // 频域滤波
//    cv::Mat filteredImage;
//    cv::mulSpectrums(dftImage, filter, filteredImage, 0);
//
//    // 逆DFT
//    cv::Mat idftImage;
//    cv::idft(filteredImage, idftImage, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
//
//    // 取指数
//    cv::Mat expImage;
//    cv::exp(idftImage, expImage);
//
//    // 转换回8位图像
//    cv::Mat result;
//    expImage.convertTo(result, CV_8U);
//    return result;
//}
//
//cv::Mat calcCumulativeHist(const cv::Mat& hist) {
//    cv::Mat cum_hist = hist.clone();
//    for (int i = 1; i < hist.rows; i++) {
//        cum_hist.at<float>(i) += cum_hist.at<float>(i - 1);
//    }
//    return cum_hist;
//}
//
//// 直方图匹配函数
//cv::Mat histogramMatching(const cv::Mat& src, const cv::Mat& ref) {
//    cv::Mat srcHist, refHist;
//    int histSize = 256;
//    float range[] = { 0, 256 };
//    const float* histRange = { range };
//
//     计算源图像和参考图像的直方图
//    cv::calcHist(&src, 1, 0, cv::Mat(), srcHist, 1, &histSize, &histRange, true, false);
//    cv::calcHist(&ref, 1, 0, cv::Mat(), refHist, 1, &histSize, &histRange, true, false);
//
//     计算累积直方图
//    cv::Mat srcCumHist = calcCumulativeHist(srcHist);
//    cv::Mat refCumHist = calcCumulativeHist(refHist);
//
//     创建查找表
//    cv::Mat lookupTable(1, 256, CV_8U);
//    for (int i = 0; i < 256; i++) {
//        float src_value = srcCumHist.at<float>(i);
//        int j = 0;
//        while (j < 256 && refCumHist.at<float>(j) < src_value) {
//            j++;
//        }
//        lookupTable.at<uchar>(i) = j;
//    }
//
//     应用查找表
//    cv::Mat matched;
//    cv::LUT(src, lookupTable, matched);
//
//    return matched;
//}


    //cv::Mat img1 = cv::imread("1.bmp", cv::IMREAD_GRAYSCALE);
    //cv::Mat img2 = cv::imread("2.bmp", cv::IMREAD_GRAYSCALE);
    //// 加载图像
    //cv::Mat img1 = cv::imread("1.bmp", cv::IMREAD_GRAYSCALE);
    //cv::Mat img2 = cv::imread("2.bmp", cv::IMREAD_GRAYSCALE);

    //if (img1.empty() || img2.empty()) {
    //    std::cerr << "Error: Unable to load images!" << std::endl;
    //    return -1;
    //}

    //// 直方图均衡化
    //cv::Mat img1_equalized, img2_equalized;
    //cv::equalizeHist(img1, img1_equalized);
    //cv::equalizeHist(img2, img2_equalized);

    //// 显示结果
    //cv::imshow("Image 1 Equalized", img1_equalized);
    //cv::imshow("Image 2 Equalized", img2_equalized);
    //cv::waitKey(0);
    //return 0;
    // 
    //----------------------------------------------------------------------------------------
    // 
    // 加载图像
    //cv::Mat img1 = cv::imread("1.bmp", cv::IMREAD_GRAYSCALE);
    //cv::Mat img2 = cv::imread("2.bmp", cv::IMREAD_GRAYSCALE);

    //if (img1.empty() || img2.empty()) {
    //    std::cerr << "Error: Unable to load images!" << std::endl;
    //    return -1;
    //}

    //// 计算平均灰度
    //double mean1 = cv::mean(img1)[0];
    //double mean2 = cv::mean(img2)[0];

    //// 调整亮度和对比度
    //cv::Mat img1_adjusted, img2_adjusted;
    //if (mean1 > mean2) {
    //    double alpha = mean2 / mean1;
    //    img1.convertTo(img1_adjusted, -1, alpha, 0);
    //    img2_adjusted = img2.clone();
    //}
    //else {
    //    double alpha = mean1 / mean2;
    //    img2.convertTo(img2_adjusted, -1, alpha, 0);
    //    img1_adjusted = img1.clone();
    //}

    //// 显示结果
    //cv::imshow("Image 1 Adjusted", img1_adjusted);
    //cv::imshow("Image 2 Adjusted", img2_adjusted);
    //cv::waitKey(0);
    //return 0;
    // 
    //-----------------------------------------------------------------------------------------

    //    // 加载图像
    //cv::Mat img1 = cv::imread("1.bmp", cv::IMREAD_GRAYSCALE);
    //cv::Mat img2 = cv::imread("2.bmp", cv::IMREAD_GRAYSCALE);

    //if (img1.empty() || img2.empty()) {
    //    std::cerr << "Error: Unable to load images!" << std::endl;
    //    return -1;
    //}


    //// 应用同态滤波器
    //cv::Mat img1_filtered = homomorphicFilter(img1);
    //cv::Mat img2_filtered = homomorphicFilter(img2);



    //// 显示结果
    //cv::imshow("Image 1 Filtered", img1_filtered);
    //cv::imshow("Image 2 Filtered", img2_filtered);
    //cv::waitKey(0);
    //return 0;
    //------------------------------------------------------------------

    //if (img1.empty() || img2.empty()) {
    //    std::cerr << "Error: Unable to load images!" << std::endl;
    //    return -1;
    //}

    //// 对图像1进行直方图匹配，使其与图像2相似
    //cv::Mat img2_matched = histogramMatching(img2, img1);


    //// 显示结果
    //cv::imshow("Image 1 Original", img1);
    //cv::imshow("Image 2 Reference", img2);
    //cv::imshow("Image 2 Matched", img2_matched);
    //cv::waitKey(0);
    //return 0;


int main() {
    // 读取主图像和模板图像
    cv::Mat image = cv::imread("3.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat templ = cv::imread("4.jpg", cv::IMREAD_GRAYSCALE);
    if (image.empty()||templ.empty()) {
        std::cerr << "无法打开图像文件！" << std::endl;
        return -1;
    }

    cv::Mat result;
    cv::matchTemplate(image, templ, result, cv::TM_CCOEFF_NORMED);

    // 寻找匹配位置
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

    // 模板匹配的位置
    cv::Point matchLoc = maxLoc;

    // 在图像中标记匹配结果
    cv::rectangle(image, matchLoc, cv::Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), cv::Scalar::all(0), 2, 8, 0);

    // 显示结果
    cv::imshow("Matched Image", image);
    cv::waitKey(0);


    return 0;
}

