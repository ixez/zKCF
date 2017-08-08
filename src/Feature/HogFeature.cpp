#include <Feature/IFeature.h>
#include "Feature/HogFeature.h"
#include "fhog.hpp"

namespace zkcf {
    using namespace cv;
    HogFeature::HogFeature(IKernel::eType kt) : IFeature(kt) {
        switch(kt) {
            case IKernel::GAUSSIAN:
                ((GaussianKernel *)Kernel)->Sigma=0.6;
                break;
            default:
                break;
        }
    }

    Mat HogFeature::Extract(const Mat& patch, sSz& sz) const {
        using namespace fhog;
        IplImage z_ipl = patch;
        CvLSVMFeatureMapCaskade *map;
        getFeatureMaps(&z_ipl, CellSize, &map);
        normalizeAndTruncate(map, 0.2f);
        PCAFeatureMaps(map);
        sz.y = map->sizeY;
        sz.x = map->sizeX;
        sz.cn = map->numFeatures;
        Mat feat = Mat(Size(map->numFeatures, map->sizeX * map->sizeY), CV_32F, map->map);

        feat = feat.t();
        freeFeatureMapObject(&map);
    }
}
