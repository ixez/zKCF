#pragma once
#include "Kernel/IKernel.h"
#include "Kernel/GaussianKernel.h"
#include <opencv2/opencv.hpp>

namespace zkcf {
    using namespace cv;
    class IFeature {
    public:
        typedef enum
        {
            HOG     = 1,
            HOG_LAB = 2,
            GRAY    = 3,
            RAW     = 4
        } eType;
        typedef struct
        {
            int y=0;
            int x=0;
            int cn=0;
        } sSz;

        IFeature(IKernel::eType kt) {
            switch(kt) {
                case IKernel::GAUSSIAN:
                    Kernel=new GaussianKernel;
                    break;
            }
        }

        virtual Mat Extract(const Mat& patch, sSz& sz) const=0;

        IKernel* Kernel=nullptr;
        int CellSize = 1;
    };
}