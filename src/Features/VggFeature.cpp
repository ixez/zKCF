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
    Mat VggFeature::Extract(const Mat &patch, FeatureSize &sz) {
        Mat feat;
        vector<Mat> inputChns;
        Preprocess(patch, InputChns);

        CHECK((float *)&(InputChns.at(0).data) == Model->input_blobs()[0]->cpu_data())
        << "Input channels are not wrapping the input layer of the network.";

        CHECK(Model->has_blob(LayerName))
        << "Unknown feature blob name " << LayerName << " in the network";

        int layerId = -1;
        for(int i=0;i<Model->layer_names().size();i++) {
            if(LayerName==Model->layer_names()[i]) {
                layerId=i;
                break;
            }
        }
        CHECK_GE(layerId, 0); CHECK_LT(layerId, Model->layers().size());

        Model->ForwardFromTo(0,layerId);
        //    Model->Forward();

        /* Copy the output layer to a std::vector */
        const boost::shared_ptr<Blob<float>> blob = Model->blob_by_name(LayerName);
        float *blobData=blob->mutable_cpu_data();
        std::vector<cv::Mat> outputChns;

//        cv::Mat featureMap(blob->channels(),mapSize.height*mapSize.width,CV_32FC1);
//        size_x=mapSize.width;
//        size_y=mapSize.height;
//        size_c=blob->channels();
//        for (int d = 0; d < blob->channels(); ++d) {
//            cv::Mat tc(blob->height(),blob->width(),CV_32FC1,blobData);
//            tc=tc/256.f;
//            cv::resize(tc,tc,mapSize);
//            tc.reshape(1,1).copyTo(featureMap.row(d));
//            blobData += blob->height() * blob->width();
//        }

        sz.rows = feat.rows;
        sz.cols = feat.cols;
        sz.chns = 1;
        return feat.reshape(1, 1);
    }

    Mat VggFeature::MeanInit(const string &path, int chns) {
        BlobProto blobProto;
        ReadProtoFromBinaryFileOrDie(path.c_str(), &blobProto);

        /* Convert from BlobProto to Blob<float> */
        Blob<float> meanBlob;
        meanBlob.FromProto(blobProto);
        CHECK_EQ(meanBlob.channels(), InputSz.chns) << "Number of channels of mean file doesn't match input layer.";

        /* The format of the mean file is planar 32-bit float BGR or grayscale. */
        vector<Mat> channels;
        float *data = meanBlob.mutable_cpu_data();
        for (int i = 0; i < chns; ++i) {
            /* Extract an individual channel. */
            Mat channel(meanBlob.height(), meanBlob.width(), CV_32FC1, data);
            channels.push_back(channel);
            data += meanBlob.height() * meanBlob.width();
        }

        /* Merge the separate channels into a single image. */
        Mat mean;
        merge(channels, mean);

        /* Compute the global mean pixel value and create a mean image
         * filled with this value. */
        Scalar channel_mean = cv::mean(mean);
        Mean = Mat(InputSz.SizeWH(), mean.type(), channel_mean);
    }

    void VggFeature::InputLyrInit() {
        InputLyr = Model->input_blobs()[0];
        InputSz.chns = InputLyr->channels();
        InputSz.rows = InputLyr->height();
        InputSz.cols = InputLyr->width();

        float *data = InputLyr->mutable_cpu_data();
        for (int i = 0; i < InputSz.chns; ++i) {
            Mat channel(InputSz.rows, InputSz.cols, CV_32FC1, data);
            InputChns.push_back(channel);
            data += InputSz.rows * InputSz.cols;
        }

        InputLyr->Reshape(1, InputSz.chns, InputSz.rows, InputSz.cols);
        Model->Reshape();
    }

    void VggFeature::Preprocess(const Mat &img,
                                vector<Mat>& channels) {
        Mat img_;
        if (img.size() != InputSz.SizeWH())
            resize(img, img_, InputSz.SizeWH());
        else
            img_ = img.clone();

        img_.convertTo(img_, CV_32FC3);

        if(Mean.empty()) {
            split(img_, channels);
        }
        else {
            subtract(img_, Mean, img_);
            split(img_, channels);
        }
    }
}

