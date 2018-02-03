/*
Author: iXez
Website: https://ixez.github.io
Email: sachika.misawa@outlook.com
*/
#include "Features/VggFeature.h"
#include "FkFactory.h"
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
//                f = new VggFeature("./assets/vgg/VGG_ILSVRC_16_layers_deploy.prototxt.txt",
//                                   "./assets/vgg/VGG_ILSVRC_16_layers.caffemodel",
//                                   "conv1_1"
//                );
                f = new VggFeature("./assets/vgg/VGG_CNN_M_2048_deploy.prototxt",
                                   "./assets/vgg/VGG_CNN_M_2048.caffemodel",
                                   "conv2",
                                   "./assets/vgg/VGG_mean.binaryproto"
                                   );
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
                else {
                    ((GaussianKernel *) k)->Sigma = 0.6;
                }
                break;
            default:
                break;
        }
    }
}