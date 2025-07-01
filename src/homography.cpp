#include "homography.h"

#include <fstream>

#include "linalg.h"

using namespace std;

Homographer::Homographer () = default;

array<float, Homographer::HOMOGRAPHY_SIZE> Homographer::calculateHomography (
const array<array<int, HOMOGRAPHY_2D_COORDS_SIZE>, NUM_2D_COORDS>& source,
const array<array<int, HOMOGRAPHY_2D_COORDS_SIZE>, NUM_2D_COORDS>& destination) {
    const array<float, 64> equationMatrix = { -static_cast<float> (source[0][0]),
        -static_cast<float> (source[0][1]), -1.0f, 0.0f, 0.0f, 0.0f,
        static_cast<float> (destination[0][0]) * static_cast<float> (source[0][0]),
        static_cast<float> (destination[0][0]) * static_cast<float> (source[0][1]),
        0.0f, 0.0f, 0.0f, -static_cast<float> (source[0][0]),
        -static_cast<float> (source[0][1]), -1.0f,
        static_cast<float> (destination[0][1]) * static_cast<float> (source[0][0]),
        static_cast<float> (destination[0][1]) * static_cast<float> (source[0][1]),
        -static_cast<float> (source[1][0]), -static_cast<float> (source[1][1]),
        -1.0f, 0.0f, 0.0f, 0.0f,
        static_cast<float> (destination[1][0]) * static_cast<float> (source[1][0]),
        static_cast<float> (destination[1][0]) * static_cast<float> (source[1][1]),
        0.0f, 0.0f, 0.0f, -static_cast<float> (source[1][0]),
        -static_cast<float> (source[1][1]), -1.0f,
        static_cast<float> (destination[1][1]) * static_cast<float> (source[1][0]),
        static_cast<float> (destination[1][1]) * static_cast<float> (source[1][1]),
        -static_cast<float> (source[2][0]), -static_cast<float> (source[2][1]),
        -1.0f, 0.0f, 0.0f, 0.0f,
        static_cast<float> (destination[2][0]) * static_cast<float> (source[2][0]),
        static_cast<float> (destination[2][0]) * static_cast<float> (source[2][1]),
        0.0f, 0.0f, 0.0f, -static_cast<float> (source[2][0]),
        -static_cast<float> (source[2][1]), -1.0f,
        static_cast<float> (destination[2][1]) * static_cast<float> (source[2][0]),
        static_cast<float> (destination[2][1]) * static_cast<float> (source[2][1]),
        -static_cast<float> (source[3][0]), -static_cast<float> (source[3][1]),
        -1.0f, 0.0f, 0.0f, 0.0f,
        static_cast<float> (destination[3][0]) * static_cast<float> (source[3][0]),
        static_cast<float> (destination[3][0]) * static_cast<float> (source[3][1]),
        0.0f, 0.0f, 0.0f, -static_cast<float> (source[3][0]),
        -static_cast<float> (source[3][1]), -1.0f,
        static_cast<float> (destination[3][1]) * static_cast<float> (source[3][0]),
        static_cast<float> (destination[3][1]) * static_cast<float> (source[3][1]) };

    const array<float, 8> constantMatrix = { -static_cast<float> (destination[0][0]),
        -static_cast<float> (destination[0][1]),
        -static_cast<float> (destination[1][0]),
        -static_cast<float> (destination[1][1]),
        -static_cast<float> (destination[2][0]),
        -static_cast<float> (destination[2][1]),
        -static_cast<float> (destination[3][0]),
        -static_cast<float> (destination[3][1]) };

    const auto incompleteHomography =
    Linalg::linalgSolve<8, 8> (equationMatrix, constantMatrix);

    const array<float, HOMOGRAPHY_SIZE> homography = { incompleteHomography[0],
        incompleteHomography[1], incompleteHomography[2],
        incompleteHomography[3], incompleteHomography[4], incompleteHomography[5],
        incompleteHomography[6], incompleteHomography[7], 1.0f };

    return homography;
}

void Homographer::backwardMap (const array<float, HOMOGRAPHY_SIZE>& homography,
cv::Mat& sourceImage,
cv::Mat& outputImage) {
    int height = sourceImage.rows;
    int width  = sourceImage.cols;

    outputImage = cv::Mat::zeros (height, width, sourceImage.type ());

    float floatTransform[9] = { homography[0], homography[1], homography[2],
        homography[3], homography[4], homography[5], homography[6],
        homography[7], homography[8] };

    auto inverseTransformMat = cv::Mat (3, 3, CV_32F, floatTransform);

    cv::invert (inverseTransformMat, inverseTransformMat);

    auto homogeneousCoordinates = cv::Mat (height * width, 3, CV_32F);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            homogeneousCoordinates.at<float> (y * width + x, 0) = x;
            homogeneousCoordinates.at<float> (y * width + x, 1) = y;
            homogeneousCoordinates.at<float> (y * width + x, 2) = 1.0f;
        }
    }

    auto transformedCoordinates = cv::Mat (height * width, 3, CV_32F);


    cv::gemm (homogeneousCoordinates, inverseTransformMat, 1.0, cv::Mat (), 0.0,
    transformedCoordinates, cv::GemmFlags::GEMM_2_T);

    for (int i = 0; i < height * width; ++i) {
        if (const float w = transformedCoordinates.at<float> (i, 2); w != 0.0f) {
            transformedCoordinates.at<float> (i, 0) /= w;
            transformedCoordinates.at<float> (i, 1) /= w;
        }
    }

    for (int i = 0; i < height * width; ++i) {
        int src_x =
        static_cast<int> (std::round (transformedCoordinates.at<float> (i, 0)));
        int src_y =
        static_cast<int> (std::round (transformedCoordinates.at<float> (i, 1)));

        int dst_x = i % width;
        int dst_y = i / width;

        if (src_x >= 0 && src_x < width && src_y >= 0 && src_y < height) {
            outputImage.at<cv::Vec3b> (dst_y, dst_x) =
            sourceImage.at<cv::Vec3b> (src_y, src_x);
        }
    }
}