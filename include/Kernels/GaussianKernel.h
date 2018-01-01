/*
Author: iXez
Website: https://ixez.github.io
Email: sachika.misawa@outlook.com
*/
#pragma once

#include "IKernel.h"

namespace zkcf {
    using namespace cv;
    class GaussianKernel : public IKernel {
    public:
        Mat Correlation(const Mat &x1, const Mat &x2, const FeatureSize& sz) const override;
        float Sigma;
    };
}