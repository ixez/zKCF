/*
Author: iXez
Website: https://ixez.github.io
Email: sachika.misawa@outlook.com
*/
#include "Features/VggFeature.h"
#include "FkFactory.h"

DEFINE_string(prototxt, "VGG_ILSVRC_16_layers_deploy.prototxt.txt", "Prototxt file of the net");
DEFINE_string(caffemodel, "VGG_ILSVRC_16_layers.caffemodel", "Caffemodel file of the net");
DEFINE_string(layer, "conv5_1", "Layer to output feature maps");
DEFINE_string(meanproto, "imagenet_mean.binaryproto", "Binaryproto file of the net");

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
                f = new VggFeature(FLAGS_prototxt, FLAGS_caffemodel, FLAGS_layer, FLAGS_meanproto, nullptr);
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