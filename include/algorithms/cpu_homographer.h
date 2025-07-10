#pragma once
#include <algorithms/base_homographer.h>

class CPUHomographer : public Homographer {
protected:
    void transformCoordinates(cv::Mat homogeneousCoordinates, cv::Mat inverseTransform,
                              cv::Mat transformedCoordinates) override;

    void transformImage(cv::Mat sourceImage, cv::Mat outputImage, cv::Mat transformedCoordinates) override;
};