/*
Author: iXez
Website: https://ixez.github.io
Email: sachika.misawa@outlook.com
*/
#pragma once
namespace zkcf {
    typedef struct {
        int rows = 0;
        int cols = 0;
        int chns = 0;
    } FeatureSize;

    typedef enum {
        FEAT_HOG = 1,
        FEAT_HOG_LAB = 2,
        FEAT_GRAY = 3,
        FEAT_RAW = 4,
        FEAT_VGG = 5
    } FeatureType;

    typedef enum {
        KRNL_GAUSSIAN = 1,
        KRNL_POLYNOMIAL = 2,
        KRNL_LINEAR = 3
    } KernelType;
}