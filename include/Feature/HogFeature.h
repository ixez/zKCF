#pragma once

#include "Feature/IFeature.h"
namespace zkcf {
    using namespace cv;
    class HogFeature : public IFeature {
    public:
        HogFeature(IKernel::Type kt);
        Mat Extract(const Mat& patch, Sz& sz) const override;

        int CellSize = 4;
    };
}