#pragma once

#include "Def.h"

#include "Feature/IFeature.h"
#include "Feature/HogFeature.h"
#include "Feature/HogLabFeature.h"
#include "Feature/RawFeature.h"

#include "Kernel/IKernel.h"
#include "Kernel/GaussianKernel.h"

namespace zkcf {
    void FkFactory(FeatureType ft, KernelType kt, IFeature*& f, IKernel*& k) {
        switch (ft) {
            case FEAT_HOG:
                f = new HogFeature();
                break;
            case FEAT_HOG_LAB:
                f = new HogLabFeature();
                break;
            case FEAT_RAW:
                f = new RawFeature();
                break;
            default:
                break;
        }
        switch (kt) {
            case KRNL_GAUSSIAN:
                k=new GaussianKernel;
                if(ft == FEAT_HOG) {
                    ((GaussianKernel *)k)->Sigma = 0.6;
                }
                else if(ft==FEAT_HOG_LAB) {
                    ((GaussianKernel *)k)->Sigma = 0.4;
                }
                else if (ft == FEAT_RAW) {
                    ((GaussianKernel *)k)->Sigma = 0.2;
                }
                break;
            default:
                break;
        }

    }
}