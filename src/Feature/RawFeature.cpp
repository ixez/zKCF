#include "Feature/RawFeature.h"
#include <opencv2/opencv.hpp>
namespace zkcf {
    using namespace cv;
    Mat RawFeature::Extract(const Mat &patch, FeatureSize &sz) const {
        Mat feat;
        cvtColor(patch, feat, CV_BGR2GRAY);
        feat.convertTo(feat, CV_32F, 1 / 255.f);
        feat -= 0.5f;       // TODO: Why?
        sz.rows = feat.rows;
        sz.cols = feat.cols;
        sz.cns = 1;
        return feat.reshape(1,1);
    }
}