/*
Author: iXez
Website: https://ixez.github.io
Email: sachika.misawa@outlook.com
*/
#pragma once
#include "Features/IFeature.h"
#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>

namespace zkcf {
    using namespace cv;
    using namespace caffe;
    class VggFeature : public IFeature {
    public:
        VggFeature(const string &modelPath, const string &weightsPath, const string &layerName,
                   const string &meanPath = "", const Scalar *meanVal = nullptr);
        Mat Extract(const Mat& patch, FeatureSize& sz) override;
    private:
        std::shared_ptr<Net<float> > Model;
        FeatureSize InputSz;
        Blob<float> *InputLyr;
        vector<Mat> InputMats;
        Mat Mean;
        string LayerName;

        void MeanInit(const string &path);
        void MeanInit(const Scalar &meanVal);
        void InputLyrInit();
        void Preprocess(const Mat &img, vector<Mat>& channels);
    };
}