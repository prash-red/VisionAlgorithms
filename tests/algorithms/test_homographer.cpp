#include "constants.h"
#include <algorithms/cpu_homographer.h>
#include <catch2/catch_test_macros.hpp>
#ifdef USE_CUDA
#    include <algorithms/cuda_homographer.cuh>
#endif

TEST_CASE("Test CPUHomographer") {
    CPUHomographer homographer = CPUHomographer();

    SECTION("test calculateHomography") {
        std::array<std::array<int, AbstractHomographer::HOMOGRAPHY_2D_COORDS_SIZE>, AbstractHomographer::NUM_2D_COORDS>
            source{};
        std::array<std::array<int, AbstractHomographer::HOMOGRAPHY_2D_COORDS_SIZE>, AbstractHomographer::NUM_2D_COORDS>
            destination{};

        for (size_t i = 0; i < 4; ++i) {
            source[i][0] = DefaultConstants::DEFAULT_HOMOGRAPHY_SOURCE_COORDS[2 * i];
            source[i][1] = DefaultConstants::DEFAULT_HOMOGRAPHY_SOURCE_COORDS[2 * i + 1];
            destination[i][0] = DefaultConstants::DEFAULT_HOMOGRAPHY_DESTINATION_COORDS[2 * i];
            destination[i][1] = DefaultConstants::DEFAULT_HOMOGRAPHY_DESTINATION_COORDS[2 * i + 1];
        }

        auto homography = homographer.calculateHomography(source, destination);

        CHECK(homography.size() == AbstractHomographer::HOMOGRAPHY_SIZE);
        CHECK(homography[0] == Approx(2.0f));
        CHECK(homography[1] == Approx(0.0f));
        CHECK(homography[2] == Approx(0.0f));
        CHECK(homography[3] == Approx(0.0f));
        CHECK(homography[4] == Approx(2.0f));
        CHECK(homography[5] == Approx(0.0f));
        CHECK(homography[6] == Approx(0.0f));
        CHECK(homography[7] == Approx(0.0f));
        CHECK(homography[8] == Approx(1.0f));
    }

    SECTION("test backwardMap") {
        std::array<std::array<int, AbstractHomographer::HOMOGRAPHY_2D_COORDS_SIZE>, AbstractHomographer::NUM_2D_COORDS>
            source{};
        std::array<std::array<int, AbstractHomographer::HOMOGRAPHY_2D_COORDS_SIZE>, AbstractHomographer::NUM_2D_COORDS>
            destination{};

        for (size_t i = 0; i < 4; ++i) {
            source[i][0] = DefaultConstants::DEFAULT_HOMOGRAPHY_SOURCE_COORDS[2 * i];
            source[i][1] = DefaultConstants::DEFAULT_HOMOGRAPHY_SOURCE_COORDS[2 * i + 1];
            destination[i][0] = DefaultConstants::DEFAULT_HOMOGRAPHY_DESTINATION_COORDS[2 * i];
            destination[i][1] = DefaultConstants::DEFAULT_HOMOGRAPHY_DESTINATION_COORDS[2 * i + 1];
        }

        auto homography = homographer.calculateHomography(source, destination);

        cv::Mat sourceImage = cv::Mat(100, 100, CV_8UC3, cv::Scalar(255, 0, 0));
        cv::Mat outputImage;

        for (int i = 0; i < 100; ++i) {
            for (int j = 0; j < 100; ++j) {
                if (i <= 50 && j <= 50) {
                    sourceImage.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 255, 0);
                }
            }
        }

        homographer.backwardMap(homography, sourceImage, outputImage);

        CHECK(outputImage.rows == 100);
        CHECK(outputImage.cols == 100);

        for (int i = 0; i < outputImage.rows; ++i) {
            for (int j = 0; j < outputImage.cols; ++j) {
                cv::Vec3b pixel = outputImage.at<cv::Vec3b>(i, j);
                CHECK(pixel == cv::Vec3b(0, 255, 0));
            }
        }
    }
}

#ifdef USE_CUDA
TEST_CASE("Test CUDAHomographer") {
    CUDAHomographer homographer = CUDAHomographer();

    SECTION("test backwardMap") {
        std::array<std::array<int, AbstractHomographer::HOMOGRAPHY_2D_COORDS_SIZE>, AbstractHomographer::NUM_2D_COORDS>
            source{};
        std::array<std::array<int, AbstractHomographer::HOMOGRAPHY_2D_COORDS_SIZE>, AbstractHomographer::NUM_2D_COORDS>
            destination{};

        for (size_t i = 0; i < 4; ++i) {
            source[i][0] = DefaultConstants::DEFAULT_HOMOGRAPHY_SOURCE_COORDS[2 * i];
            source[i][1] = DefaultConstants::DEFAULT_HOMOGRAPHY_SOURCE_COORDS[2 * i + 1];
            destination[i][0] = DefaultConstants::DEFAULT_HOMOGRAPHY_DESTINATION_COORDS[2 * i];
            destination[i][1] = DefaultConstants::DEFAULT_HOMOGRAPHY_DESTINATION_COORDS[2 * i + 1];
        }

        auto homography = homographer.calculateHomography(source, destination);

        cv::Mat sourceImage = cv::Mat(100, 100, CV_8UC3, cv::Scalar(255, 0, 0));
        cv::Mat outputImage;

        for (int i = 0; i < 100; ++i) {
            for (int j = 0; j < 100; ++j) {
                if (i <= 50 && j <= 50) {
                    sourceImage.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 255, 0);
                }
            }
        }

        homographer.backwardMap(homography, sourceImage, outputImage);

        CHECK(outputImage.rows == 100);
        CHECK(outputImage.cols == 100);

        for (int i = 0; i < outputImage.rows; ++i) {
            for (int j = 0; j < outputImage.cols; ++j) {
                cv::Vec3b pixel = outputImage.at<cv::Vec3b>(i, j);
                CHECK(pixel == cv::Vec3b(0, 255, 0));
            }
        }
    }
}
#endif // USE_CUDA
