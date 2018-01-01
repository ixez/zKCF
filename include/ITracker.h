/*
Author: iXez
Website: https://ixez.github.io
Email: sachika.misawa@outlook.com
*/
#pragma once
#include <opencv2/core/core.hpp>
namespace ztrack {
    using namespace cv;
    class ITracker {
    public:
        virtual bool Init(const Mat &frame, const Rect &roi) = 0;
        virtual Rect Track(const Mat &frame) = 0;
    };
}