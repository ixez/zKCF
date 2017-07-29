#pragma once

#include "Feature/IFeature.h"
namespace zkcf {
    class HogFeature : public IFeature {
    public:
        HogFeature(IKernel::Type kt);
    private:
        int CellSize = 4;
    };
}