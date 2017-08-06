#pragma once

#include "Feature/IFeature.h"
namespace zkcf {
    class HogFeature : public IFeature {
    public:
        HogFeature(IKernel::Type kt);
        cv::Mat Extract(const cv::Mat& patch) const override;

        int CellSize = 4;
    };
}