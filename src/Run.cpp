/*
Author: iXez
Website: https://ixez.github.io
Email: sachika.misawa@outlook.com
*/

#include "KCF.h"
#include "TaskConfig.h"
#include "Run.h"
#include <glog/logging.h>

boost::program_options::variables_map vMap;
zkcf::FeatureType feat;
zkcf::KernelType krnl;

void VMapInit(int argc, char* argv[]) {
    using namespace zkcf;
    using namespace std;
    using namespace boost::program_options;

    options_description desc("Allowed options");
    desc.add_options()
            ("feature", value<string>()->default_value("hog"),
             "Feature extractor to use, hog, hog_lab, raw, gray, vgg")
            ("kernel", value<string>()->default_value("gaussian"),
             "Correlation kernel to use")
            ("padding", value<float>(),
             "Padding ratio of search area")
            ("learning_rate", value<float>(),
             "Learning rate of the correlation filter")
            ("output_sigma_factor", value<float>(),
             "OutputSigmaFactor of Y")
            ("enable_scale", value<bool>(),
             "Enable scaling")
            ("scale_n", value<int>(),
             "Scale amount")
            ("scale_step", value<float>(),
             "Scale step")
            ("scale_weight", value<float>(),
             "Scale weight when a different scale produce a higher response score")
            ;

#ifdef BUILD_VGG
    // Only available when --feature=vgg
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
#endif
    store(parse_command_line(argc, argv, desc), vMap);
    notify(vMap);
}

void FkInit() {
    using namespace std;
    using namespace zkcf;
    string t;

    // Parsing feature
    t = vMap["feature"].as<string>();
    if(t == "hog") {
        feat = FEAT_HOG;
    }
    else if(t == "hog_lab") {
        feat = FEAT_HOG_LAB;
    }
    else if(t == "raw") {
        feat = FEAT_RAW;
    }
    else if(t == "gray") {
        feat = FEAT_GRAY;
    }
    else if(t == "vgg") {
        feat = FEAT_VGG;
    }
    else {
        LOG(ERROR) << "Unknown feature.";
    }

    // Parsing kernel
    t = vMap["kernel"].as<string>();
    if(t == "gaussian") {
        krnl = KRNL_GAUSSIAN;
    }
    else {
        LOG(ERROR) << "Unknown kernel.";
    }
}

int main(int argc, char* argv[])
{
    using namespace cv;
    using namespace zkcf;
    using namespace std;
    using namespace boost::program_options;

    ztrack::TaskConfig conf;
    conf.SetArgs(argc, argv);  //TODO: merge TaskConfig with options_description

    VMapInit(argc, argv);
    FkInit();

    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();

    KCF tracker(feat, krnl);
    Mat frm;
    Rect result;
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