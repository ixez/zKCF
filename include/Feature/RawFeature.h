#pragma once

#include "Feature/IFeature.h"
namespace zkcf {
    using namespace cv;
    class RawFeature : public IFeature {
    public:
        RawFeature(IKernel::eType kt);
        Mat Extract(const Mat& patch, sSz& sz) const override;

        int CellSize = 1;
    };
}