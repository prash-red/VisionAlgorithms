#include <algorithms/cpu_homographer.h>

#include <fstream>

void CPUHomographer::transformCoordinates(cv::Mat homogeneousCoordinates, cv::Mat inverseTransform,
                                          cv::Mat transformedCoordinates) {
    cv::gemm(homogeneousCoordinates, inverseTransform, 1.0, cv::Mat(), 0.0, transformedCoordinates, cv::GemmFlags::GEMM_2_T);
}

void CPUHomographer::transformImage(cv::Mat sourceImage, cv::Mat outputImage, cv::Mat transformedCoordinates) {
    for (int i = 0; i < transformedCoordinates.rows; ++i) {
        if (const float w = transformedCoordinates.at<float>(i, 2); w != 0.0f) {
            transformedCoordinates.at<float>(i, 0) /= w;
            transformedCoordinates.at<float>(i, 1) /= w;
        }
    }
    const int height = sourceImage.rows;
    const int width = sourceImage.cols;
    for (int i = 0; i < height * width; ++i) {
        int src_x = static_cast<int>(std::round(transformedCoordinates.at<float>(i, 0)));
        int src_y = static_cast<int>(std::round(transformedCoordinates.at<float>(i, 1)));

        const int dst_x = i % width;
        const int dst_y = i / width;

        if (src_x >= 0 && src_x < width && src_y >= 0 && src_y < height) {
            outputImage.at<cv::Vec3b>(dst_y, dst_x) = sourceImage.at<cv::Vec3b>(src_y, src_x);
        }
    }
}
