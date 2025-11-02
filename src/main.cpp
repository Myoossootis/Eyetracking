#include "detection.h"
#include <iostream>

int main() {
    // --- 瞳孔检测输入与输出 ---
    // 使用明暗瞳法进行瞳孔检测，需要两张图像
    std::string pupil_light_image = "input/2.bmp";  // 明瞳图像（瞳孔亮）
    std::string pupil_dark_image = "input/1.bmp";   // 暗瞳图像（瞳孔暗）
    std::string pupil_output_file = "output/pupil_centers.txt";  // 保存瞳孔中心坐标的文件

    // --- 反射光斑检测输入与输出 ---
    // 通常使用明瞳图像，因为它有最清晰的普尔钦斑
    std::string reflection_image = "input/2.bmp";  // 用于检测普尔钦斑的图像
    std::string reflection_output_file = "output/reflection_centers.txt";  // 保存反射光斑中心坐标的文件

    // --- 执行检测 ---
    // 1. 检测瞳孔中心
    detect_pupil(pupil_light_image, pupil_dark_image, pupil_output_file);
    
    // 2. 检测反射光斑中心
    detect_reflection(reflection_image, reflection_output_file);

    return 0;
}
