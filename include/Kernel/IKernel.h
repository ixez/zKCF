#pragma once

#include <opencv2/opencv.hpp>
#include <Feature/IFeature.h>

namespace zkcf {
    using namespace cv;
    class IKernel {
    public:
        typedef enum
        {
            GAUSSIAN    = 1,
            POLYNOMIAL  = 2,
            LINEAR      = 3
        } eType;

        virtual Mat Correlation(const Mat& x1, const Mat& x2, const IFeature::sSz& sz) const = 0;
    };
}