#pragma once
#include <math/linalg.h>
#include <opencv2/opencv.hpp>

#include <array>

using namespace std;

/**
 * @brief Provides methods for homography computation and backward mapping.
 *
 * The AbstractHomographer class offers static methods to:
 * - Calculate a 3x3 homography matrix from four 2D source and destination point
 * pairs.
 * - Apply backward mapping to transform an image using a given homography
 * matrix.
 */
class AbstractHomographer {
public:
    static constexpr int HOMOGRAPHY_SIZE = 3 * 3;
    static constexpr int HOMOGRAPHY_2D_COORDS_SIZE = 2;
    static constexpr int NUM_2D_COORDS = 4;

    virtual ~AbstractHomographer() = default;
    AbstractHomographer() = default;
    /**
    * @brief Calculates the 3x3 homography matrix that maps four source points
    * to four destination points.
    *
    * This method computes the perspective transformation (homography) matrix
    * given four corresponding 2D points in the source and destination planes.
    * The result is a 3x3 matrix, returned as a flat array in row-major order,
    * suitable for use with OpenCV or CUDA-based image transformations.
    *
    * @param source      An array of four 2D integer points representing the
    * source coordinates.
    * @param destination An array of four 2D integer points representing the
    * destination coordinates.
    * @return            The computed 3x3 homography matrix as a flat array of
    * 9 floats (row-major).
    */

    virtual array<float, HOMOGRAPHY_SIZE>
    calculateHomography(const array<array<int, HOMOGRAPHY_2D_COORDS_SIZE>, NUM_2D_COORDS>& source,
                        const array<array<int, HOMOGRAPHY_2D_COORDS_SIZE>, NUM_2D_COORDS>& destination);

    /**
    * @brief Applies backward mapping to transform an image using a given
    * homography matrix.
    *
    * This method warps the input image (`sourceImage`) to the output image
    * (`outputImage`) using the provided 3x3 homography matrix. The
    * transformation is performed using backward mapping, which ensures that
    * every pixel in the output image is mapped from the corresponding location
    * in the source image.
    *
    * @param homography   The 3x3 homography matrix as a flat array of 9 floats
    * (row-major).
    * @param sourceImage  The input image to be transformed (cv::Mat).
    * @param outputImage  The output image after applying the homography
    * (cv::Mat).
    */
    virtual void backwardMap(const array<float, HOMOGRAPHY_SIZE>& homography, cv::Mat& sourceImage, cv::Mat& outputImage);

protected:
    /**
    * @brief Transforms a set of homogeneous coordinates using an inverse
    * transformation matrix.
    *
    * This pure virtual method applies the given inverse transformation matrix to
    * the provided homogeneous coordinates, storing the result in the
    * transformedCoordinates matrix. Implementations should ensure that the
    * transformation is performed correctly for all points.
    *
    * @param homogeneousCoordinates Input matrix of points in homogeneous
    * coordinates (cv::Mat).
    * @param inverseTransform       Inverse transformation matrix to apply
    * (cv::Mat).
    * @param transformedCoordinates Output matrix to store the transformed
    * coordinates (cv::Mat).
    */
    virtual void transformCoordinates(cv::Mat homogeneousCoordinates, cv::Mat inverseTransform,
                                      cv::Mat transformedCoordinates) = 0;

    /**
    * @brief Applies a transformation to the input image using the provided
    * transformed coordinates.
    *
    * This pure virtual method should be implemented to map pixels from the input
    * image to the output image based on the given matrix of transformed
    * coordinates. The implementation is responsible for handling interpolation
    * and boundary conditions as needed.
    *
    * @param sourceImage             The source image to be transformed
    * (cv::Mat).
    * @param outputImage            The destination image after transformation
    * (cv::Mat).
    * @param transformedCoordinates Matrix containing the transformed coordinates
    * for mapping (cv::Mat).
    */
    virtual void transformImage(cv::Mat sourceImage, cv::Mat outputImage, cv::Mat transformedCoordinates) = 0;
};
