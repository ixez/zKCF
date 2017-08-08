#pragma once

#include "Feature/IFeature.h"
namespace zkcf {
    using namespace cv;
    class HogFeature : public IFeature {
    public:
        HogFeature(IKernel::eType kt);
        Mat Extract(const Mat& patch, sSz& sz) const override;

        int CellSize = 4;
    };
}