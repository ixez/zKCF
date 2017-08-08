#pragma once

#include "IKernel.h"

namespace zkcf {
    class GaussianKernel : public IKernel {
    public:
        Mat Correlation(const Mat &x1, const Mat &x2, const IFeature::sSz& sz) const override;
        float Sigma;
    };
}