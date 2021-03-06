/*
Author: iXez
Website: https://ixez.github.io
Email: sachika.misawa@outlook.com
*/
#include "Features/VggFeature.h"

namespace zkcf {
    using namespace cv;
    using namespace caffe;
    using namespace std;

    VggFeature::VggFeature(const string &modelPath, const string &weightsPath, const string &layerName,
                           const string &meanPath, const Scalar *meanVal) {
#ifdef CPU_ONLY
        Caffe::set_mode(Caffe::CPU);
#else
        Caffe::set_mode(Caffe::GPU);
#endif
        Model.reset(new Net<float>(modelPath, TEST));
        Model->CopyTrainedLayersFrom(weightsPath);

        InputLyrInit();

        LayerName=layerName;

        if(!meanPath.empty()) {
            MeanInit(meanPath);
        }
        else if(meanVal != nullptr){
            MeanInit(*meanVal);
        }
        else {
            MeanInit(Scalar(0,0,0));
        }

    }

    Mat VggFeature::Extract(const Mat &patch, FeatureSize &sz) {
#ifdef CPU_ONLY
        Caffe::set_mode(Caffe::CPU);
#else
        Caffe::set_mode(Caffe::GPU);
#endif
        FeatureRatio.width = (float)patch.cols / Model->blob_by_name(LayerName)->width();
        FeatureRatio.height = (float)patch.rows / Model->blob_by_name(LayerName)->height();

        vector<Mat> inputChns;
        Preprocess(patch, InputMats);

        CHECK((float*)InputMats.at(0).data == Model->input_blobs()[0]->mutable_cpu_data())
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

        /* Copy the output layer to a std::vector */
        const boost::shared_ptr<Blob<float>> blob = Model->blob_by_name(LayerName);
        std::vector<cv::Mat> outputChns;

        sz.rows = blob->height();
        sz.cols = blob->width();
        sz.chns = blob->channels();
        Mat feat(sz.chns,sz.rows*sz.cols,CV_32F);

        float *blobData=blob->mutable_cpu_data();

        for (int d = 0; d < sz.chns; ++d) {
            cv::Mat blobMat(blob->height(),blob->width(),CV_32F,blobData);
            blobMat.reshape(1,1).copyTo(feat.row(d));
            blobData += blob->height() * blob->width();
        }

        double min, max;
        cv::minMaxLoc(feat, &min, &max);

        feat /= max;
        return feat;
    }

    void VggFeature::InputLyrInit() {
        InputLyr = Model->input_blobs()[0];
        InputSz.chns = InputLyr->channels();
        InputSz.rows = InputLyr->height();
        InputSz.cols = InputLyr->width();
        InputLyr->Reshape(1, InputSz.chns, InputSz.rows, InputSz.cols);
        Model->Reshape();
        float *data = InputLyr->mutable_cpu_data();
        for (int i = 0; i < InputSz.chns; ++i) {
            Mat channel(InputSz.rows, InputSz.cols, CV_32FC1, data);
            InputMats.push_back(channel);
            data += InputSz.rows * InputSz.cols;
        }
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

    void VggFeature::MeanInit(const string &path) {
        BlobProto blobProto;
        ReadProtoFromBinaryFileOrDie(path.c_str(), &blobProto);

        /* Convert from BlobProto to Blob<float> */
        Blob<float> meanBlob;
        meanBlob.FromProto(blobProto);
        CHECK_EQ(meanBlob.channels(), InputSz.chns) << "Number of channels of mean file doesn't match input layer.";

        /* The format of the mean file is planar 32-bit float BGR or grayscale. */
        vector<Mat> channels;
        float *data = meanBlob.mutable_cpu_data();
        for (int i = 0; i < InputSz.chns; ++i) {
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
        Scalar meanVal = cv::mean(mean);
        Mean = Mat(InputSz.SizeWH(), mean.type(), meanVal);
    }

    void VggFeature::MeanInit(const Scalar &meanVal) {
        Mean = Mat(InputSz.SizeWH(), CV_32FC3, meanVal);
    }
}

