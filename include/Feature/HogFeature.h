#pragma once

#include "Feature/IFeature.h"
namespace zkcf {
    class HogFeature : public IFeature {
    public:
        HogFeature(IKernel::Type kt);
        int CellSize = 4;
    };
}