#include "detection.h"
#include <iostream>

int main() {
    std::string pupil_light_image = "2.bmp";  // 明瞳图像
    std::string pupil_dark_image = "1.bmp";   // 暗瞳图像
    std::string pupil_output_file = "output/pupil_centers.txt";  // 输出文件

    std::string reflection_image = "2.bmp";  // 反射点图像(明瞳)
    std::string reflection_output_file = "output/reflection_centers.txt";  // 输出文件

    // 选择调用瞳孔检测或反射点检测
    detect_pupil(pupil_light_image, pupil_dark_image, pupil_output_file);
    detect_reflection(reflection_image, reflection_output_file);

    return 0;
}
