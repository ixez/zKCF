#pragma once
#include <Kernel/IKernel.h>
#include <zconf.h>
#include <Kernel/GaussianKernel.h>

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

        IKernel* Kernel=nullptr;
        int CellSize = 1;
    };
}