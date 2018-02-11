/*
Author: iXez
Website: https://ixez.github.io
Email: sachika.misawa@outlook.com
*/

#include "FkFactory.h"
#ifdef BUILD_VGG
#include "Features/VggFeature.h"
#endif
#ifndef BUILD_LIB
#include "Run.h"
#endif

namespace zkcf {
    void FkFactory(FeatureType ft, KernelType kt, IFeature *&f, IKernel *&k) {
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
            case FEAT_GRAY:
                f = new RawFeature(false);
                break;
#ifdef BUILD_VGG
            case FEAT_VGG:
#ifndef BUILD_LIB
                f = new VggFeature(VMap["vgg_prototxt"].as<string>(),
                                   VMap["vgg_caffemodel"].as<string>(),
                                   VMap["vgg_layer"].as<string>(),
                                   VMap["vgg_meanproto"].as<string>(),
                                   nullptr);
#endif
//                f = new VggFeature("./assets/vgg/VGG_CNN_M_2048_deploy.prototxt",
//                                   "./assets/vgg/VGG_CNN_M_2048.caffemodel",
//                                   "conv1"
//                                   );
                break;
#endif
            default:
                break;
        }
        switch (kt) {
            case KRNL_GAUSSIAN:
                k = new GaussianKernel;
                if (ft == FEAT_HOG) {
                    ((GaussianKernel *) k)->Sigma = 0.6;
                }
                else if (ft == FEAT_HOG_LAB) {
                    ((GaussianKernel *) k)->Sigma = 0.4;
                }
                else if (ft == FEAT_RAW) {
                    ((GaussianKernel *) k)->Sigma = 0.2;
                }
                else if (ft == FEAT_VGG) {
                    ((GaussianKernel *) k)->Sigma = 0.6;
                }
                else {
                    ((GaussianKernel *) k)->Sigma = 0.6;
                }
                break;
            default:
                break;
        }
    }
}