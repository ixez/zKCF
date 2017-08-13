#include "Feature/RawFeature.h"
namespace zkcf {
    using namespace cv;
    Mat RawFeature::Extract(const Mat &patch, FeatureSize &sz) const {
        Mat feat;
        cvtColor(patch, feat, CV_BGR2GRAY);
        feat.convertTo(feat, CV_32F, 1 / 255.f);
        sz.y = feat.rows;
        sz.x = feat.cols;
        sz.cn = 1;
    }
}