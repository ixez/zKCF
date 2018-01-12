/*
Author: iXez
Website: https://ixez.github.io
Email: sachika.misawa@outlook.com
*/
#pragma once
#include "HogFeature.h"
#include <opencv2/opencv.hpp>

namespace zkcf {
    using namespace cv;
    using namespace std;
    class HogLabFeature : public HogFeature {
    public:
        HogLabFeature();
        Mat Extract(const Mat &patch, FeatureSize &sz) const override;

    private:
        Mat LabCentroids;
    };
}