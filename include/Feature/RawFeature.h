#pragma once

#include "Feature/IFeature.h"

namespace zkcf {
    using namespace cv;
    class RawFeature : public IFeature {
    public:
        Mat Extract(const Mat& patch, FeatureSize& sz) const override;

        int CellSize = 1;
    };
}