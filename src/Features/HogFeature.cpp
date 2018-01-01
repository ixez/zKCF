/*
Author: iXez
Website: https://ixez.github.io
Email: sachika.misawa@outlook.com
*/
#include "Features/IFeature.h"
#include "Features/HogFeature.h"
#include "Features/Impl/HogFeature/fhog.hpp"

namespace zkcf {
    using namespace cv;
    HogFeature::HogFeature() { CellSize=4; }

    Mat HogFeature::Extract(const Mat& patch, FeatureSize& sz) const {
        using namespace fhog;
        IplImage z_ipl = patch;
        CvLSVMFeatureMapCaskade *map;
        getFeatureMaps(&z_ipl, CellSize, &map);
        normalizeAndTruncate(map, 0.2f);
        PCAFeatureMaps(map);
        sz.rows = map->sizeY;
        sz.cols = map->sizeX;
        sz.cns = map->numFeatures;
        Mat feat = Mat(Size(map->numFeatures, map->sizeX * map->sizeY), CV_32F, map->map);

        feat = feat.t();
        freeFeatureMapObject(&map);
        return feat;
    }

}
