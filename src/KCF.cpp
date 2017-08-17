#include <FkFactory.h>
#include "KCF.h"
#include "FFTTools.hpp"
#include "recttools.hpp"
#include "fhog.hpp"
#include "labdata.hpp"

namespace zkcf {
    using namespace cv;
// Update position based on the new frame
    Rect KCF::update(Mat image) {
//        if (Roi.x + Roi.width <= 0) Roi.x = -Roi.width + 1;
//        if (Roi.y + Roi.height <= 0) Roi.y = -Roi.height + 1;
//        if (Roi.x >= image.cols - 1) Roi.x = image.cols - 2;
//        if (Roi.y >= image.rows - 1) Roi.y = image.rows - 2;
//
//        float cx = Roi.x + Roi.width / 2.0f;
//        float cy = Roi.y + Roi.height / 2.0f;
//
//
//        float peak_value;
//        Point2f res = detect(x, GetFeatures(image, 0, 1.0f), peak_value);
//
//        if (ScaleStep != 1) {
//            // Test at a smaller _scale
//            float new_peak_value;
//            Point2f new_res = detect(x, GetFeatures(image, 0, 1.0f / ScaleStep), new_peak_value);
//
//            if (ScaleWeight * new_peak_value > peak_value) {
//                res = new_res;
//                peak_value = new_peak_value;
//                _scale /= ScaleStep;
//                Roi.width /= ScaleStep;
//                Roi.height /= ScaleStep;
//            }
//
//            // Test at a bigger _scale
//            new_res = detect(x, GetFeatures(image, 0, ScaleStep), new_peak_value);
//
//            if (ScaleWeight * new_peak_value > peak_value) {
//                res = new_res;
//                peak_value = new_peak_value;
//                _scale *= ScaleStep;
//                Roi.width *= ScaleStep;
//                Roi.height *= ScaleStep;
//            }
//        }
//
//        // Adjust by cell size and _scale
//        Roi.x = cx - Roi.width / 2.0f + ((float) res.x * CellSize * _scale);
//        Roi.y = cy - Roi.height / 2.0f + ((float) res.y * CellSize * _scale);
//
//        if (Roi.x >= image.cols - 1) Roi.x = image.cols - 1;
//        if (Roi.y >= image.rows - 1) Roi.y = image.rows - 1;
//        if (Roi.x + Roi.width <= 0) Roi.x = -Roi.width + 2;
//        if (Roi.y + Roi.height <= 0) Roi.y = -Roi.height + 2;
//
//        assert(Roi.width >= 0 && Roi.height >= 0);
//        Mat x = GetFeatures(image, 0);
//        train(x, LearningRate);

        return Roi;
    }


// Detect object in the current frame.
    Point2f KCF::detect(Mat z, Mat x, float &peak_value) {
//        using namespace FFTTools;

//        Mat k = gaussianCorrelation(x, z);
//        Mat res = (real(fftd(complexMultiplication(ModelAlphaF, fftd(k)), true)));
//
//        //minMaxLoc only accepts doubles for the peak, and integer points for the coordinates
//        Point2i pi;
//        double pv;
//        minMaxLoc(res, NULL, &pv, NULL, &pi);
//        peak_value = (float) pv;
//
//        //subpixel peak estimation, coordinates will be non-integer
//        Point2f p((float) pi.x, (float) pi.y);
//        Point2f p();
//
//        if (pi.x > 0 && pi.x < res.cols - 1) {
//            p.x += subPixelPeak(res.at<float>(pi.y, pi.x - 1), peak_value, res.at<float>(pi.y, pi.x + 1));
//        }
//
//        if (pi.y > 0 && pi.y < res.rows - 1) {
//            p.y += subPixelPeak(res.at<float>(pi.y - 1, pi.x), peak_value, res.at<float>(pi.y + 1, pi.x));
//        }
//
//        p.x -= (res.cols) / 2;
//        p.y -= (res.rows) / 2;

        return Point2f();
    }

    void KCF::ModelInit(const Mat& x) {
        Mat k = Krnl->Correlation(x, x, FeatSz);
        Mat kf=FFTTools::fftd(k);
        Mat alphaf = FFTTools::complexDivision(ModelYf, kf + Lambda);

        ModelAlphaF = alphaf;
        ModelXf=FFTTools::fftd(x);
    }

// train tracker with a single image
    void KCF::train(Mat x, float train_interp_factor) {
        using namespace FFTTools;

//        Mat k = gaussianCorrelation(x, x);
//        Mat alphaf = complexDivision(ModelYf, (fftd(k) + Lambda));
//
//        x = (1 - train_interp_factor) * x + (train_interp_factor) * x;
//        ModelAlphaF = (1 - train_interp_factor) * ModelAlphaF + (train_interp_factor) * alphaf;
    }

// Create Gaussian Peak. Function called only in the first frame.
    Mat KCF::CalcGaussianMap(const FeatureSize& sz, float sigma) {
        Mat_<float> res(sz.y, sz.x);

        int syh = (sz.y) / 2;
        int sxh = (sz.x) / 2;


        float mult = -0.5 / pow(sigma,2);

        for (int i = 0; i < sz.y; i++) {
            for (int j = 0; j < sz.x; j++) {
                int ih = i - syh;
                int jh = j - sxh;
                res(i, j) = std::exp(mult * (float) (ih * ih + jh * jh));
            }
        }
        return res;
    }

    // TODO: make it static
    Mat KCF::GetFeatures(const Mat &patch, FeatureSize& featSz) const {
        Rect paddedRoi;

        float cx = Roi.x + Roi.width / 2;
        float cy = Roi.y + Roi.height / 2;

        int paddedW = Roi.width * Padding;
        int paddedH = Roi.height * Padding;
        float scale;
        Size tmplSz;

        if (TmplMode==TMPL_MODE_FIXED) {
            // Fit largest dimension to the given length
            if (Roi.width >= Roi.height)  //fit to width
                scale = paddedW / (float) TmplLen;
            else
                scale = paddedH / (float) TmplLen;

            tmplSz.width = paddedW / scale;
            tmplSz.height = paddedH / scale;
        }
        else if(TmplMode==TMPL_MODE_NONE) {  //No template size given, use ROI size
            tmplSz.width = paddedW;
            tmplSz.height = paddedH;
            scale = 1;
        }
        else {
            std::cout<<"Unknown template mode."<<std::endl;
        }

        tmplSz.width=(tmplSz.width/2)*2;
        tmplSz.height=(tmplSz.height/2)*2;

        if(FeatType==FEAT_HOG || FeatType==FEAT_HOG_LAB) {
            // Round to cell size and also make it even
            tmplSz.width+=Feat->CellSize*2;
            tmplSz.height+=Feat->CellSize*2;
        }
        else {
            //Make number of pixels even (helps with some logic involving half-dimensions)
        }

        paddedRoi.width = tmplSz.width * scale;
        paddedRoi.height = tmplSz.height * scale;

        // center roi with new size
        paddedRoi.x = cx - paddedRoi.width / 2;
        paddedRoi.y = cy - paddedRoi.height / 2;

        Mat z = RectTools::subwindow(patch, paddedRoi, BORDER_REPLICATE);
        resize(z, z, tmplSz);
        Mat feat=Feat->Extract(z,featSz);
        return feat;
    }

// Initialize Hanning window. Function called only in the first frame.
    Mat KCF::CalcHann(const FeatureSize &sz) {
        Mat hann1t = Mat(Size(sz.x, 1), CV_32F, Scalar(0));
        Mat hann2t = Mat(Size(1, sz.y), CV_32F, Scalar(0));

        for (int i = 0; i < hann1t.cols; i++)
            hann1t.at<float>(0, i) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann1t.cols - 1)));
        for (int i = 0; i < hann2t.rows; i++)
            hann2t.at<float>(i, 0) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann2t.rows - 1)));

        Mat hann2d = hann2t * hann1t;

        Mat hann1d = hann2d.reshape(1, 1);
        Mat hann = Mat(Size(sz.y * sz.x, sz.cn), CV_32F, Scalar(0));
        for (int i = 0; i < sz.cn; i++) {
            for (int j = 0; j < sz.y * sz.x; j++) {
                hann.at<float>(i, j) = hann1d.at<float>(0, j);
            }
        }
        return hann;
    }

// Calculate sub-pixel peak for one dimension
    float KCF::subPixelPeak(float left, float center, float right) {
        float divisor = 2 * center - right - left;

        if (divisor == 0)
            return 0;

        return 0.5 * (right - left) / divisor;
    }

    KCF::KCF(FeatureType ft, KernelType kt) {
        FeatType=ft;
        KrnlType=kt;
    }

    void KCF::Init(const Mat &frm, Rect roi) {
        Lambda = 0.0001;
        Padding = 2.5;
        OutputSigmaFactor = 0.125;
        switch(FeatType) {
            case FEAT_HOG:
                LearningRate = 0.012;
                break;
            case FEAT_HOG_LAB:
                LearningRate = 0.005;
                OutputSigmaFactor = 0.1;
                break;
            case FEAT_RAW:
                LearningRate = 0.075;
                break;
        }
        FkFactory(FeatType,KrnlType,Feat,Krnl);
        if(EnableScale) {
            TmplMode=TMPL_MODE_FIXED;
            ScaleStep = 1.05;
            ScaleWeight = 0.95;
        }
        else {
            ScaleStep = 1;
            ScaleWeight = 1;
        }

        assert(roi.width >= 0 && roi.height >= 0);
        Roi = roi;

        Mat x = GetFeatures(frm, FeatSz);
        Mat hann= CalcHann(FeatSz);
        x = hann.mul(x);

        float outputSigma = std::sqrt((float) FeatSz.y * FeatSz.x) / Padding * OutputSigmaFactor;
        Mat y = CalcGaussianMap(FeatSz,outputSigma);

        ModelYf=FFTTools::fftd(y);
        ModelInit(x);
    }
}