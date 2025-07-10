#pragma once
#include <algorithms/base_homographer.h>
#include <cublas_v2.h>

class CUDAHomographer : public Homographer {
public:
    CUDAHomographer();
    ~CUDAHomographer();

protected:
    void transformCoordinates(cv::Mat homogeneousCoordinates, cv::Mat inverseTransform,
                              cv::Mat transformedCoordinates) override;

    void transformImage(cv::Mat sourceImage, cv::Mat outputImage, cv::Mat transformedCoordinates) override;

private:
    /**
     * @brief Allocates device memory and copies coordinate and transform data to the GPU.
     *
     * @param homogeneousCoordinates Input matrix of homogeneous coordinates (CPU).
     * @param inverseTransform Input inverse transformation matrix (CPU).
     * @param transformedCoordinates Output matrix for transformed coordinates (CPU).
     * @param d_inverseTransformArray_fp32 Device pointer for inverse transform data (GPU).
     * @param d_homogeneousCoordsArray_fp32 Device pointer for homogeneous coordinates data (GPU).
     */
    void mallocAndCopyCoordinatesData(cv::Mat homogeneousCoordinates, cv::Mat inverseTransform,
                                      cv::Mat transformedCoordinates, float** d_inverseTransformArray_fp32,
                                      float** d_homogeneousCoordsArray_fp32);

    /**
     * @brief cuBLAS handle for performing GPU-accelerated linear algebra operations.
     */
    cublasHandle_t cublasHandle;
    /**
     * @brief Device pointer for storing transformed coordinates on the GPU.
     */
    float* d_transformCoordsArray_fp32 = nullptr;
};
