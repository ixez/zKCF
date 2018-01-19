/*
Author: iXez
Website: https://ixez.github.io
Email: sachika.misawa@outlook.com
*/
#pragma once
#include <opencv2/opencv.hpp>

namespace zkcf {
    struct FeatureSize {
        int rows = 0;
        int cols = 0;
        int chns = 0;
        cv::Size SizeWH() {
            return cv::Size(cols,rows);
        }
        cv::Size SizeWHC() {
            return cv::Size(chns,rows*cols);
        }
    };

    enum FeatureType {
        FEAT_HOG = 1,
        FEAT_HOG_LAB = 2,
        FEAT_GRAY = 3,
        FEAT_RAW = 4,
        FEAT_VGG = 5
    };

    enum KernelType {
        KRNL_GAUSSIAN = 1,
        KRNL_POLYNOMIAL = 2,
        KRNL_LINEAR = 3
    };
}