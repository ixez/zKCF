#include "Feature/HogFeature.h"
#include "fhog.hpp"

namespace zkcf {
    using namespace cv;
    HogFeature::HogFeature(IKernel::Type kt) : IFeature(kt) {

    }

    Mat HogFeature::Extract(const Mat& patch, Sz& sz) const {
        using namespace fhog;
        IplImage z_ipl = patch;
        CvLSVMFeatureMapCaskade *map;
        getFeatureMaps(&z_ipl, CellSize, &map);
        normalizeAndTruncate(map, 0.2f);
        PCAFeatureMaps(map);
        sz[0] = map->sizeY;
        sz[1] = map->sizeX;
        sz[2] = map->numFeatures;
        Mat feat = Mat(Size(map->numFeatures, map->sizeX * map->sizeY), CV_32F, map->map);

        feat = feat.t();
        freeFeatureMapObject(&map);
    }
}
