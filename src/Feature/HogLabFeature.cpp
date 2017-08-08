#include "Feature/HogLabFeature.h"

namespace zkcf {
    HogLabFeature::HogLabFeature(IKernel::eType kt) : HogFeature(kt)
    {
        LabCentroids = cv::Mat(ClustersN, 3, CV_32FC1, &Clusters);
        switch(kt) {
            case IKernel::GAUSSIAN:
                ((GaussianKernel *)Kernel)->Sigma=0.4;
                break;
            default:
                break;
        }
    }
}
