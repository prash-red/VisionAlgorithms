#include <cublas_v2.h>
#include <cuda_helpers/cuda_constants.h>
#include <cuda_runtime.h>

#include <algorithms/cuda_homographer.cuh>
#include <cuda_helpers/cuda_helper_checks.cuh>

CUDAHomographer::CUDAHomographer() {
    CHECK_CUBLAS(cublasCreate_v2(&cublasHandle));
}

CUDAHomographer::~CUDAHomographer() {
    CHECK_CUBLAS(cublasDestroy_v2(cublasHandle));
}

void CUDAHomographer::mallocAndCopyCoordinatesData(cv::Mat homogeneousCoordinates, cv::Mat inverseTransform,
                                                   cv::Mat transformedCoordinates, float** d_inverseTransformArray_fp32,
                                                   float** d_homogeneousCoordsArray_fp32) {
    auto* homogeneousCoordsArray = reinterpret_cast<float*>(homogeneousCoordinates.data);
    size_t homogeneousCoordsArraySize = homogeneousCoordinates.total() * homogeneousCoordinates.elemSize();
    auto* inverseTransformArray = reinterpret_cast<float*>(inverseTransform.data);
    size_t inverseTransformArraySize = inverseTransform.total() * inverseTransform.elemSize();
    size_t transformedCoordinatesSize = transformedCoordinates.total() * transformedCoordinates.elemSize();

    CHECK_CUDA(cudaMalloc(d_homogeneousCoordsArray_fp32, homogeneousCoordsArraySize));
    CHECK_CUDA(cudaMalloc(d_inverseTransformArray_fp32, inverseTransformArraySize));
    CHECK_CUDA(cudaMalloc(&d_transformCoordsArray_fp32, transformedCoordinatesSize));

    CHECK_CUDA(cudaMemcpy(*d_homogeneousCoordsArray_fp32,
                          homogeneousCoordsArray,
                          homogeneousCoordsArraySize,
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(*d_inverseTransformArray_fp32,
                          inverseTransformArray,
                          inverseTransformArraySize,
                          cudaMemcpyHostToDevice));
}

void CUDAHomographer::transformCoordinates(cv::Mat homogeneousCoordinates, cv::Mat inverseTransform,
                                           cv::Mat transformedCoordinates) {
    float* d_inverseTransformArray_fp32;
    float* d_homogeneousCoordsArray_fp32;

    if (d_transformCoordsArray_fp32) {
        CHECK_CUDA(cudaFree(d_transformCoordsArray_fp32));
    }

    mallocAndCopyCoordinatesData(homogeneousCoordinates,
                                 inverseTransform,
                                 transformedCoordinates,
                                 &d_inverseTransformArray_fp32,
                                 &d_homogeneousCoordsArray_fp32);

    float alpha = 1.0f, beta = 0.0f;

    CHECK_CUBLAS(cublasSgemm_v2(cublasHandle,
                                CUBLAS_OP_T,
                                CUBLAS_OP_N,
                                transformedCoordinates.cols,
                                homogeneousCoordinates.rows,
                                inverseTransform.cols,
                                &alpha,
                                d_inverseTransformArray_fp32,
                                inverseTransform.cols,
                                d_homogeneousCoordsArray_fp32,
                                homogeneousCoordinates.cols,
                                &beta,
                                d_transformCoordsArray_fp32,
                                transformedCoordinates.cols));

    CHECK_CUDA(cudaFree(d_homogeneousCoordsArray_fp32));
    CHECK_CUDA(cudaFree(d_inverseTransformArray_fp32));
}

__global__ void transformImageKernel(uint8_t* d_sourceImage, uint8_t* d_outputImage, float* d_transformedCoords,
                                     int width, int height) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (width * height)) {
        return; // Out of bounds check
    }
    int src_x = static_cast<int>(d_transformedCoords[index * 3 + 0] / d_transformedCoords[index * 3 + 2]);
    int src_y = static_cast<int>(d_transformedCoords[index * 3 + 1] / d_transformedCoords[index * 3 + 2]);

    if (src_x >= 0 && src_x < width && src_y >= 0 && src_y < height) {
        for (int c = 0; c < 3; ++c) {
            d_outputImage[index * 3 + c] = d_sourceImage[(src_y * width + src_x) * 3 + c];
        }
    }
}

void CUDAHomographer::transformImage(cv::Mat sourceImage, cv::Mat outputImage, cv::Mat transformedCoordinates) {
    uint8_t* d_sourceImageArray;
    uint8_t* d_outputImageArray;
    CHECK_CUDA(cudaMalloc(&d_sourceImageArray, sourceImage.total() * sourceImage.elemSize()));
    CHECK_CUDA(cudaMalloc(&d_outputImageArray, outputImage.total() * outputImage.elemSize()));
    CHECK_CUDA(cudaMemcpy(d_sourceImageArray,
                          sourceImage.data,
                          sourceImage.total() * sourceImage.elemSize(),
                          cudaMemcpyHostToDevice));

    dim3 blockDim = BLOCK_SIZE;
    dim3 gridDim((transformedCoordinates.total() + BLOCK_SIZE - 1) / BLOCK_SIZE);

    transformImageKernel<<<gridDim, blockDim>>>(d_sourceImageArray,
                                                d_outputImageArray,
                                                d_transformCoordsArray_fp32,
                                                sourceImage.cols,
                                                sourceImage.rows);
    cudaDeviceSynchronize();

    CHECK_CUDA(cudaMemcpy(outputImage.data,
                          d_outputImageArray,
                          outputImage.total() * outputImage.elemSize(),
                          cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_sourceImageArray));
    CHECK_CUDA(cudaFree(d_outputImageArray));
    CHECK_CUDA(cudaFree(d_transformCoordsArray_fp32));
    d_transformCoordsArray_fp32 = nullptr; // Reset pointer to avoid double free
}
