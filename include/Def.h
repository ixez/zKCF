#pragma once
namespace zkcf {
    typedef struct {
        int y = 0;
        int x = 0;
        int cn = 0;
    } FeatureSize;

    typedef enum {
        FEAT_HOG = 1,
        FEAT_HOG_LAB = 2,
        FEAT_GRAY = 3,
        FEAT_RAW = 4
    } FeatureType;

    typedef enum {
        KRNL_GAUSSIAN = 1,
        KRNL_POLYNOMIAL = 2,
        KRNL_LINEAR = 3
    } KernelType;
}