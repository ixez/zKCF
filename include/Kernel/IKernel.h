#pragma once

#include "Def.h"
#include <opencv2/opencv.hpp>

namespace zkcf {
    using namespace cv;

    class IKernel {
    public:
        virtual Mat Correlation(const Mat& x1, const Mat& x2, const FeatureSize& sz) const = 0;
    };
}