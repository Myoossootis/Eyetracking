## Eyetracking
Eyetracking is a prototype eye-tracking system developed based on C++, which aims to detect and track the position of the human pupil in real time through computer vision technology. It can be applied in scenarios such as human-computer interaction, attention analysis, and control of assistive devices. 

This project consists of multiple core modules (such as image preprocessing, gradient calculation, pupil location, etc.) and provides several test images of myself (such as 1.bmp, 2.bmp, etc.) for algorithm verification and result demonstration.

## Highlight
This open-source repository contains the static gaze tracking and calibration algorithm.
For the complete C#-based PC application featuring an interactive graphical interface, real-time gaze tracking & mapping, dynamic Kalman filter optimization, and user interaction controls, please refer to our collaborative open-source project (co-developed with my teammates):

👉 [https://github.com/feast107/EyeTracking]

---

🛠️ Features

✅ Pupil detection and localization based on images

✅ Image gradient calculation (e.g. left_gradient.jpg / right_gradient.jpg)

✅ Modular design (main.cpp, pupil.cpp, reflection.cpp, grad.cpp, etc.)

✅ Support for static image input and processing

✅ Provide visual output of results (e.g. result.jpg)

✅ Include multiple test images and intermediate process images (bmp/png format)

---

🚀 Quick Start
1.	Clone the Repository
git clone https://github.com/Myoossootis/Eyetracking.git

2.	Compile and Run (Windows + Visual Studio Example)
This project includes Visual Studio project files (tracking.sln &*. vcxproj). 

The recommended way to build and run the code is as follows:

Open the solution file tracking.sln using Visual Studio 2019 or 2022.

Select the appropriate platform (e.g., x64 or Win32) and build configuration (e.g., Debug or Release).

Compile the project and run the main program (main.cpp).

---

📖 Usage Example / Sample Workflow
Once the program is executed, it will load predefined test images (typically bright/dark pupil images)

The system then performs pupil detection using the following key algorithmic steps:

🚩Image differencing

🚩Morphological processing

🚩Gradient computation

🚩Threshold segmentation

🚩Hough circle fitting

🚩🚩Etc.

After processing, the program outputs the detected pupil center location and pupillary center & Purkinje spot, and visually marks them on the eye image.

---

📌 Other Information
🎊 Current Status: Prototype Development Phase
The core algorithm has achieved high robustness in gaze tracking detection, and is capable of performing gaze parameter calibration under varying lighting conditions and camera angles.

---

🔮 Future Plans
Improve algorithm runtime efficiency and reduce computational complexity

Port the core algorithm to Verilog for deployment on our custom FPGA development board, enabling parallelized image processing and real-time gaze tracking at higher speed

Commercialize our eye-tracking hardware/software system, aiming for mass production and sales of a domestically developed eye tracker

Publish academic papers to contribute to the research community# Eyetracking
