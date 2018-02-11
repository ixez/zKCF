/*
Author: iXez
Website: https://ixez.github.io
Email: sachika.misawa@outlook.com
*/

#include "KCF.h"
#include "TaskConfig.h"
#include "Run.h"

boost::program_options::variables_map VMap;

int main(int argc, char* argv[])
{
    using namespace cv;
    using namespace zkcf;
    using namespace std;
    using namespace boost::program_options;

    ztrack::TaskConfig conf;
    conf.SetArgs(argc, argv);  //TODO: merge TaskConfig with options_description

    options_description desc("Allowed options");
    desc.add_options()
            ("vgg_prototxt",
             value<string>()->default_value("VGG_ILSVRC_16_layers_deploy.prototxt.txt"),
             "Prototxt file of the net")
            ("vgg_caffemodel",
             value<string>()->default_value("VGG_ILSVRC_16_layers.caffemodel"),
             "Caffemodel file of the net")
            ("vgg_layer",
             value<string>()->default_value("conv5_1"),
             "Layer to output feature maps")
            ("vgg_meanproto",
             value<string>()->default_value("imagenet_mean.binaryproto"),
             "Binaryproto file of the net")
    ;
    store(parse_command_line(argc, argv, desc), VMap);
    notify(VMap);

    Mat frm;
    Rect result;
    KCF tracker(FeatureType::FEAT_VGG, KernelType::KRNL_GAUSSIAN);
//    KCF tracker(FeatureType::FEAT_RAW, KernelType::KRNL_GAUSSIAN);
//    KCF tracker(FeatureType::FEAT_HOG, KernelType::KRNL_GAUSSIAN);
//    KCF tracker(FeatureType::FEAT_HOG_LAB, KernelType::KRNL_GAUSSIAN);

    for (int frameId = conf.StartFrmId, i = 1; frameId <= conf.EndFrmId; ++frameId, ++i)
    {
        // Read each frame from the list
        frm = conf.GetFrm(frameId);
        if (i == 1) {
            result=conf.Bbox;
            tracker.Init(frm, result);
        } else {
            result = tracker.Track(frm);
        }
        conf.PushResult(result);
    }
    conf.SaveResults();
    return EXIT_SUCCESS;
}