#include "Feature/HogLabFeature.h"

namespace zkcf {
    HogLabFeature::HogLabFeature(IKernel::Type kt) : HogFeature(kt)
    {
        LabCentroids = cv::Mat(ClustersN, 3, CV_32FC1, &Clusters);
    }
}
