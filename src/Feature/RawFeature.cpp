#include "Feature/RawFeature.h"
namespace zkcf {
    using namespace cv;
    RawFeature::RawFeature(zkcf::IKernel::eType kt) : IFeature(kt) {
        switch(kt) {
            case IKernel::GAUSSIAN:
                ((GaussianKernel *)Kernel)->Sigma=0.2;
                break;
            default:
                break;
        }
    }

    Mat RawFeature::Extract(const Mat &patch, zkcf::IFeature::sSz &sz) const {
        Mat feat;
        cvtColor(patch, feat, CV_BGR2GRAY);
        feat.convertTo(feat, CV_32F, 1 / 255.f);
        sz.y = feat.rows;
        sz.x = feat.cols;
        sz.cn = 1;
    }
}