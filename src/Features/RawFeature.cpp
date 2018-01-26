/*
Author: iXez
Website: https://ixez.github.io
Email: sachika.misawa@outlook.com
*/
#include "Features/RawFeature.h"
#include <opencv2/opencv.hpp>
namespace zkcf {
    using namespace cv;
    RawFeature::RawFeature(bool color) {
        CellSize=1;
        Color=color;
    }
    Mat RawFeature::Extract(const Mat &patch, FeatureSize &sz) {
        if(Color) {
            Mat feat;
            vector<Mat> chns;
            split(patch,chns);
            for(const auto &chn:chns) {
                Mat c=chn.reshape(1,1);
                c.convertTo(c,CV_32F,1/255.f);
                c-=0.5f;
                feat.push_back(c);
            }
            sz.rows = patch.rows;
            sz.cols = patch.cols;
            sz.chns = 3;
            return feat;
        }
        else {
            Mat feat;
            cvtColor(patch, feat, CV_BGR2GRAY);
            feat.convertTo(feat, CV_32F, 1 / 255.f);
            feat -= 0.5f;       // Unknown: Why?
            sz.rows = feat.rows;
            sz.cols = feat.cols;
            sz.chns = 1;
            return feat.reshape(1, 1);
        }
    }
}