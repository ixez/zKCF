/*
Author: iXez
Website: https://ixez.github.io
Email: sachika.misawa@outlook.com
*/
#pragma once

#include "Features/IFeature.h"
#include <opencv2/opencv.hpp>

namespace zkcf {
    using namespace cv;
    class HogFeature : public IFeature {
    public:
        HogFeature();
        Mat Extract(const Mat& patch, FeatureSize& sz) override;
        int CellSize;
    };
}