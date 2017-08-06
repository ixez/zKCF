#include "Feature/HogFeature.h"
namespace zkcf {
    zkcf::HogFeature::HogFeature(IKernel::Type kt) : IFeature(kt) {

    }

    cv::Mat HogFeature::Extract(const cv::Mat &patch) const {
        return cv::Mat();
    }
}
