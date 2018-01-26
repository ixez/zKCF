/*

Author: iXez
Website: https://ixez.github.io
Email: sachika.misawa@outlook.com
*/
#pragma once

#include "Features/IFeature.h"

namespace zkcf {
    using namespace cv;
    class RawFeature : public IFeature {
    public:
        explicit RawFeature(bool color=true);
        Mat Extract(const Mat& patch, FeatureSize& sz) override;
    private:
        bool Color;
    };
}