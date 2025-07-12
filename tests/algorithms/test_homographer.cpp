#include <algorithms/cpu_homographer.h>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <constants.h>
#ifdef USE_CUDA
#    include <algorithms/cuda_homographer.cuh>
#endif

template <typename T>
struct HomographerFixture {
    T homographer;

    HomographerFixture() : homographer() {}
};

using MyTypes = std::tuple<CPUHomographer
#ifdef USE_CUDA
                           ,
                           CUDAHomographer
#endif
                           >;

TEMPLATE_LIST_TEST_CASE_METHOD(HomographerFixture, "Test Homographer", "[class][template][list]", MyTypes) {
    auto homographer = HomographerFixture<CPUHomographer>().homographer;
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
        CHECK(homography[0] == Catch::Approx(2.0f));
        CHECK(homography[1] == Catch::Approx(0.0f));
        CHECK(homography[2] == Catch::Approx(0.0f));
        CHECK(homography[3] == Catch::Approx(0.0f));
        CHECK(homography[4] == Catch::Approx(2.0f));
        CHECK(homography[5] == Catch::Approx(0.0f));
        CHECK(homography[6] == Catch::Approx(0.0f));
        CHECK(homography[7] == Catch::Approx(0.0f));
        CHECK(homography[8] == Catch::Approx(1.0f));
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
