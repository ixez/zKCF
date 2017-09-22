#pragma once

#include "Feature/IFeature.h"
#include <opencv2/opencv.hpp>

namespace zkcf {
    using namespace cv;
    class HogFeature : public IFeature {
    public:
        HogFeature();
        Mat Extract(const Mat& patch, FeatureSize& sz) const override;
    };
}