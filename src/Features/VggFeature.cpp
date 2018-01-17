/*
Author: iXez
Website: https://ixez.github.io
Email: sachika.misawa@outlook.com
*/
#include "Features/VggFeature.h"
#include <opencv2/opencv.hpp>
#include <Def.h>

namespace zkcf {
    using namespace cv;
    using namespace caffe;
    VggFeature::VggFeature(const string& modelPath, const string& weightsPath, const string& meanPath) {
        CellSize=1;
        Model.reset(new Net<float>(modelPath, TEST));
        Model->CopyTrainedLayersFrom(weightsPath);
        Blob<float> *input_layer = Model->input_blobs()[0];
        ModelInputSz.chns = input_layer->channels();
        ModelInputSz.rows = input_layer->height();
        ModelInputSz.cols = input_layer->width();

        if(!meanPath.empty()) {
            ModelMean = ModelMeanInit(meanPath, ModelInputSz.chns);
        }
    }
    Mat VggFeature::Extract(const Mat &patch, FeatureSize &sz) const {
        Mat feat;

        sz.rows = feat.rows;
        sz.cols = feat.cols;
        sz.chns = 1;
        return feat.reshape(1, 1);
    }

    Mat VggFeature::ModelMeanInit(const string &path, int chns) {
        BlobProto blob_proto;
        ReadProtoFromBinaryFileOrDie(path.c_str(), &blob_proto);

        /* Convert from BlobProto to Blob<float> */
        Blob<float> mean_blob;
        mean_blob.FromProto(blob_proto);
        CHECK_EQ(mean_blob.channels(), ModelInputSz.chns)
            << "Number of channels of mean file doesn't match input layer.";

        /* The format of the mean file is planar 32-bit float BGR or grayscale. */
        std::vector<cv::Mat> channels;
        float *data = mean_blob.mutable_cpu_data();
        for (int i = 0; i < chns; ++i) {
            /* Extract an individual channel. */
            cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
            channels.push_back(channel);
            data += mean_blob.height() * mean_blob.width();
        }

        /* Merge the separate channels into a single image. */
        cv::Mat mean;
        cv::merge(channels, mean);

        /* Compute the global mean pixel value and create a mean image
         * filled with this value. */
        cv::Scalar channel_mean = cv::mean(mean);
        ModelMean = cv::Mat(Size(ModelInputSz.cols,ModelInputSz.rows), mean.type(), channel_mean);
    }
}