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
    using namespace std;
    VggFeature::VggFeature(const string& modelPath, const string& weightsPath, const string& meanPath) {
        CellSize=1;
        Model.reset(new Net<float>(modelPath, TEST));
        Model->CopyTrainedLayersFrom(weightsPath);

        InputLyrInit();
        if(!meanPath.empty()) {
            Mean = MeanInit(meanPath, InputSz.chns);
        }
    }
    Mat VggFeature::Extract(const Mat &patch, FeatureSize &sz) const {
        Mat feat;

        Blob<float> *input_layer = Model->input_blobs()[0];
        input_layer->Reshape(1, InputSz.chns,
                             InputSz.rows, InputSz.cols);
        /* Forward dimension change to all layers. */
        Model->Reshape();

        vector<Mat> input_channels;
        Preprocess(patch, InputChns);

        sz.rows = feat.rows;
        sz.cols = feat.cols;
        sz.chns = 1;
        return feat.reshape(1, 1);
    }

    Mat VggFeature::MeanInit(const string &path, int chns) {
        BlobProto blob_proto;
        ReadProtoFromBinaryFileOrDie(path.c_str(), &blob_proto);

        /* Convert from BlobProto to Blob<float> */
        Blob<float> mean_blob;
        mean_blob.FromProto(blob_proto);
        CHECK_EQ(mean_blob.channels(), InputSz.chns) << "Number of channels of mean file doesn't match input layer.";

        /* The format of the mean file is planar 32-bit float BGR or grayscale. */
        vector<Mat> channels;
        float *data = mean_blob.mutable_cpu_data();
        for (int i = 0; i < chns; ++i) {
            /* Extract an individual channel. */
            Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
            channels.push_back(channel);
            data += mean_blob.height() * mean_blob.width();
        }

        /* Merge the separate channels into a single image. */
        Mat mean;
        merge(channels, mean);

        /* Compute the global mean pixel value and create a mean image
         * filled with this value. */
        Scalar channel_mean = cv::mean(mean);
        Mean = Mat(Size(InputSz.cols,InputSz.rows), mean.type(), channel_mean);
    }

    void VggFeature::InputLyrInit() {
        InputLyr = Model->input_blobs()[0];
        InputSz.chns = InputLyr->channels();
        InputSz.rows = InputLyr->height();
        InputSz.cols = InputLyr->width();

        float *input_data = InputLyr->mutable_cpu_data();
        for (int i = 0; i < InputSz.chns; ++i) {
            Mat channel(InputSz.rows, InputSz.cols, CV_32FC1, input_data);
            InputChns.push_back(channel);
            input_data += InputSz.rows * InputSz.cols;
        }

        InputLyr->Reshape(1, InputSz.chns, InputSz.rows, InputSz.cols);
        Model->Reshape();
    }

    void VggFeature::Preprocess(const Mat &img,
                                vector<Mat>& input_channels) {
        /* Convert the input image to the input image format of the network. */
        Mat sample;
        if (img.channels() == 3 && num_channels_ == 1)
            cvtColor(img, sample, COLOR_BGR2GRAY);
        else if (img.channels() == 4 && num_channels_ == 1)
            cvtColor(img, sample, COLOR_BGRA2GRAY);
        else if (img.channels() == 4 && num_channels_ == 3)
            cvtColor(img, sample, COLOR_BGRA2BGR);
        else if (img.channels() == 1 && num_channels_ == 3)
            cvtColor(img, sample, COLOR_GRAY2BGR);
        else
            sample = img;

        Mat sample_resized;
        if (sample.size() != input_geometry_)
            resize(sample, sample_resized, input_geometry_);
        else
            sample_resized = sample;

        Mat sample_float;
        if (num_channels_ == 3)
            sample_resized.convertTo(sample_float, CV_32FC3);
        else
            sample_resized.convertTo(sample_float, CV_32FC1);

        if(mean_.empty()) {
            split(sample_float, *input_channels);
        }
        else {
            Mat sample_normalized;
            subtract(sample_float, mean_, sample_normalized);

            /* This operation will write the separate BGR planes directly to the
             * input layer of the network because it is wrapped by the Mat
             * objects in input_channels. */
            split(sample_normalized, *input_channels);
        }

        CHECK(reinterpret_cast<float *>(input_channels->at(0).data)
              == net_->input_blobs()[0]->cpu_data())
        << "Input channels are not wrapping the input layer of the network.";
    }
}

