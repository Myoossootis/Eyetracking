#pragma once
#include <vector>
#include <stdexcept>
#include <cmath>
#include <tuple>

class Matrix {
private:
    std::vector<std::vector<double>> data;
    size_t rows, cols;

public:
    Matrix(size_t r, size_t c) : rows(r), cols(c), data(r, std::vector<double>(c, 0)) {}
    
    double& operator()(size_t i, size_t j) { return data[i][j]; }
    const double& operator()(size_t i, size_t j) const { return data[i][j]; }
    
    Matrix transpose() const {
        Matrix result(cols, rows);
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                result(j, i) = data[i][j];
        return result;
    }
    
    Matrix operator*(const Matrix& other) const {
        if (cols != other.rows) throw std::runtime_error("维度不匹配");
        Matrix result(rows, other.cols);
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < other.cols; ++j)
                for (size_t k = 0; k < cols; ++k)
                    result(i, j) += data[i][k] * other(k, j);
        return result;
    }

    Matrix inverse() const {
        if (rows != cols) throw std::runtime_error("非方阵");
        Matrix augmented(rows, 2 * cols);
        
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                augmented(i, j) = data[i][j];
                augmented(i, j + cols) = (i == j) ? 1.0 : 0.0;
            }
        }
        
        for (size_t i = 0; i < rows; ++i) {
            double pivot = augmented(i, i);
            if (std::abs(pivot) < 1e-10) throw std::runtime_error("矩阵奇异");
            
            for (size_t j = 0; j < 2 * cols; ++j)
                augmented(i, j) /= pivot;
            
            for (size_t k = 0; k < rows; ++k) {
                if (k != i) {
                    double factor = augmented(k, i);
                    for (size_t j = 0; j < 2 * cols; ++j)
                        augmented(k, j) -= factor * augmented(i, j);
                }
            }
        }
        
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                result(i, j) = augmented(i, j + cols);
        return result;
    }
};

class GazeCalibration {
private:
    std::vector<double> coefficientsX;
    std::vector<double> coefficientsY;
    double lastX = 0;
    double lastY = 0;
    static constexpr double RegularizationLambda = 1e-6;

    static Matrix buildDesignMatrix(
        const std::vector<double>& deltaX, 
        const std::vector<double>& deltaY) 
    {
        Matrix matrix(deltaX.size(), 6);
        for (size_t i = 0; i < deltaX.size(); i++) {
            matrix(i, 0) = deltaX[i];
            matrix(i, 1) = deltaY[i];
            matrix(i, 2) = deltaX[i] * deltaY[i];
            matrix(i, 3) = deltaX[i] * deltaX[i];
            matrix(i, 4) = deltaY[i] * deltaY[i];
            matrix(i, 5) = 1.0;
        }
        return matrix;
    }

    static std::vector<double> fitWithRegularization(
        const std::vector<double>& deltaX,
        const std::vector<double>& deltaY,
        const std::vector<double>& screenCoords)
    {
        Matrix A = buildDesignMatrix(deltaX, deltaY);
        Matrix AT = A.transpose();
        Matrix ATA = AT * A;
        
        // 添加正则化项
        for (size_t i = 0; i < 6; ++i)
            ATA(i, i) += RegularizationLambda;
        
        Matrix ATAinv = ATA.inverse();
        Matrix temp = ATAinv * AT;
        
        std::vector<double> result(6);
        for (size_t i = 0; i < 6; ++i) {
            result[i] = 0;
            for (size_t j = 0; j < screenCoords.size(); ++j)
                result[i] += temp(i, j) * screenCoords[j];
        }
        return result;
    }

public:
    GazeCalibration(const std::vector<std::tuple<double, double, double, double>>& calibrationData) {
        if (calibrationData.size() < 3) {
            throw std::runtime_error("At least 3 calibration points required");
        }

        std::vector<double> deltaX, deltaY, screenX, screenY;
        for (const auto& [dx, dy, sx, sy] : calibrationData) {
            deltaX.push_back(dx);
            deltaY.push_back(dy);
            screenX.push_back(sx);
            screenY.push_back(sy);
        }

        coefficientsX = fitWithRegularization(deltaX, deltaY, screenX);
        coefficientsY = fitWithRegularization(deltaX, deltaY, screenY);
    }

    std::pair<double, double> calculateGazePoint(double deltaX, double deltaY) {
        std::vector<double> features = {
            deltaX, deltaY, deltaX * deltaY, 
            deltaX * deltaX, deltaY * deltaY, 1.0
        };

        double thisX = 0, thisY = 0;
        for (int i = 0; i < 6; i++) {
            thisX += coefficientsX[i] * features[i];
            thisY += coefficientsY[i] * features[i];
        }

        thisX = std::min(1900.0, thisX);
        thisY = std::min(1000.0, thisY);

        if (lastX == 0 && lastY == 0) {
            lastX = thisX;
            lastY = thisY;
        } else {
            double stepX = lastX + (thisX - lastX) / 10.0;
            double stepY = lastY + (thisY - lastY) / 10.0;
            lastX = stepX;
            lastY = stepY;
            thisX = stepX;
            thisY = stepY;
        }

        return {thisX, thisY};
    }
};