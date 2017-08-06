#pragma once
#include "Kernel/IKernel.h"
#include "Kernel/GaussianKernel.h"
#include <opencv2/core/mat.hpp>

namespace zkcf {
    class IFeature {
    public:
        typedef enum
        {
            HOG     = 1,
            HOG_LAB = 2,
            GRAY    = 3,
            RAW     = 4
        } Type;

        virtual IFeature(IKernel::Type kt) {
            switch(kt) {
                case IKernel::GAUSSIAN:
                    Kernel=new GaussianKernel;
                    break;
            }
        }

        virtual cv::Mat Extract(const cv::Mat& patch) const=0;

        IKernel* Kernel=nullptr;
        int CellSize = 1;
    };
}