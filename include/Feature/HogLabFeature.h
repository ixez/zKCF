#pragma once

#include "HogFeature.h"
#include <opencv2/opencv.hpp>

namespace zkcf {
    using namespace cv;
    using namespace std;
    class HogLabFeature : public HogFeature {
    public:
    private:
        HogLabFeature();

        Mat LabCentroids;
        vector<vector<float>> Clusters;
    };
}