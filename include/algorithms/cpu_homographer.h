#pragma once
#include <algorithms/abstract_homographer.h>

class CPUHomographer : public AbstractHomographer {
protected:
    void transformCoordinates(cv::Mat homogeneousCoordinates, cv::Mat inverseTransform,
                              cv::Mat transformedCoordinates) override;

    void transformImage(cv::Mat sourceImage, cv::Mat outputImage, cv::Mat transformedCoordinates) override;
};