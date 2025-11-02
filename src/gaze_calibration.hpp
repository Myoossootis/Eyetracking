#ifndef GAZE_CALIBRATION_HPP
#define GAZE_CALIBRATION_HPP

#include <opencv2/opencv.hpp>
#include <vector>

/**
 * @class Matrix
 * @brief 一个基础的矩阵类，用于视线标定中的数学运算。
 *
 * 这个类封装了矩阵的基本操作，如转置、乘法和求逆，这些都是
 * 在使用最小二乘法进行视线标定时所必需的。
 */
class Matrix {
public:
    /**
     * @brief 构造一个指定尺寸的矩阵。
     * @param rows 矩阵的行数。
     * @param cols 矩阵的列数。
     */
    Matrix(int rows, int cols) : rows_(rows), cols_(cols), data_(rows * cols, 0.0) {}

    /**
     * @brief 获取矩阵的行数。
     */
    int rows() const { return rows_; }

    /**
     * @brief 获取矩阵的列数。
     */
    int cols() const { return cols_; }

    /**
     * @brief 访问矩阵元素（可修改）。
     */
    double& at(int r, int c) { return data_[r * cols_ + c]; }

    /**
     * @brief 访问矩阵元素（不可修改）。
     */
    const double& at(int r, int c) const { return data_[r * cols_ + c]; }

    /**
     * @brief 计算矩阵的转置。
     * @return 返回当前矩阵的转置矩阵。
     */
    Matrix transpose() const {
        Matrix result(cols_, rows_);
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                result.at(j, i) = at(i, j);
            }
        }
        return result;
    }

    /**
     * @brief 矩阵乘法。
     * @param other 与当前矩阵相乘的另一个矩阵。
     * @return 返回两个矩阵的乘积。
     */
    Matrix multiply(const Matrix& other) const {
        if (cols_ != other.rows()) {
            throw std::runtime_error("Matrix dimensions are not compatible for multiplication.");
        }
        Matrix result(rows_, other.cols());
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < other.cols(); ++j) {
                for (int k = 0; k < cols_; ++k) {
                    result.at(i, j) += at(i, k) * other.at(k, j);
                }
            }
        }
        return result;
    }

    /**
     * @brief 计算矩阵的逆（仅适用于方阵）。
     * @return 返回当前矩阵的逆矩阵。
     */
    Matrix inverse() const {
        if (rows_ != cols_) {
            throw std::runtime_error("Matrix must be square to be inverted.");
        }
        // 注意：这是一个简化的实现，仅适用于2x2或3x3矩阵。
        // 对于更通用的情况，需要使用如LU分解等更复杂的算法。
        cv::Mat cv_mat(rows_, cols_, CV_64F, const_cast<double*>(data_.data()));
        cv::Mat inv_mat;
        cv::invert(cv_mat, inv_mat, cv::DECOMP_SVD);

        Matrix result(rows_, cols_);
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                result.at(i, j) = inv_mat.at<double>(i, j);
            }
        }
        return result;
    }

private:
    int rows_, cols_;
    std::vector<double> data_;
};

/**
 * @class GazeCalibration
 * @brief 使用正则化最小二乘法进行视线标定。
 *
 * 这个类负责收集标定数据（瞳孔-光斑向量和屏幕坐标），
 * 并通过拟合一个线性模型来计算视线方向。
 */
class GazeCalibration {
public:
    /**
     * @brief 构造函数，可选择设置正则化参数。
     * @param lambda 正则化系数，用于防止过拟合。
     */
    GazeCalibration(double lambda = 0.1) : lambda_(lambda) {}

    /**
     * @brief 添加一个标定点。
     * @param pupil_glint_vector 瞳孔中心与普尔钦斑中心的差向量 (dx, dy)。
     * @param screen_point 对应的屏幕坐标 (x, y)。
     */
    void add_calibration_point(const cv::Point2f& pupil_glint_vector, const cv::Point2f& screen_point) {
        calibration_data_.push_back({pupil_glint_vector, screen_point});
    }

    /**
     * @brief 拟合标定模型。
     * 
     * 使用收集到的标定数据，通过正则化最小二乘法计算转换矩阵。
     * 公式为：Coefficients = (X^T * X + lambda * I)^-1 * X^T * Y
     */
    void fit_model() {
        int n = calibration_data_.size();
        if (n < 3) { // 至少需要3个点来拟合一个鲁棒的模型
            throw std::runtime_error("Not enough calibration points to fit the model.");
        }

        Matrix X(n, 3);
        Matrix Yx(n, 1), Yy(n, 1);

        for (int i = 0; i < n; ++i) {
            X.at(i, 0) = calibration_data_[i].first.x;  // dx
            X.at(i, 1) = calibration_data_[i].first.y;  // dy
            X.at(i, 2) = 1.0; // 偏置项
            Yx.at(i, 0) = calibration_data_[i].second.x; // screen_x
            Yy.at(i, 0) = calibration_data_[i].second.y; // screen_y
        }

        Matrix Xt = X.transpose();
        Matrix XtX = Xt.multiply(X);

        // 添加正则化项
        Matrix identity(3, 3);
        for (int i = 0; i < 3; ++i) identity.at(i, i) = 1.0;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                XtX.at(i, j) += lambda_ * identity.at(i, j);
            }
        }

        Matrix XtX_inv = XtX.inverse();
        Matrix XtYx = Xt.multiply(Yx);
        Matrix XtYy = Xt.multiply(Yy);

        coeffs_x_ = XtX_inv.multiply(XtYx);
        coeffs_y_ = XtX_inv.multiply(XtYy);
    }

    /**
     * @brief 根据给定的瞳孔-光斑向量估计屏幕上的注视点。
     *
     * @param pupil_glint_vector 当前的瞳孔-光斑向量。
     * @return 估计的屏幕坐标。
     */
    cv::Point2f calculate_gaze_point(const cv::Point2f& pupil_glint_vector) const {
        double screen_x = coeffs_x_.at(0, 0) * pupil_glint_vector.x + 
                          coeffs_x_.at(1, 0) * pupil_glint_vector.y + 
                          coeffs_x_.at(2, 0);
        double screen_y = coeffs_y_.at(0, 0) * pupil_glint_vector.x + 
                          coeffs_y_.at(1, 0) * pupil_glint_vector.y + 
                          coeffs_y_.at(2, 0);
        return cv::Point2f(screen_x, screen_y);
    }

private:
    double lambda_; // 正则化参数
    std::vector<std::pair<cv::Point2f, cv::Point2f>> calibration_data_;
    Matrix coeffs_x_, coeffs_y_; // 模型的系数
};

#endif // GAZE_CALIBRATION_HPP