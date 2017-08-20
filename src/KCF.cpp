#include <FkFactory.h>
#include "KCF.h"
#include "FFTTools.hpp"
#include "recttools.hpp"

namespace zkcf {
    using namespace cv;
    Mat KCF::CalcGaussianMap(const FeatureSize& sz, float sigma) {
        Mat_<float> res(sz.rows, sz.cols);

        int syh = (sz.rows) / 2;
        int sxh = (sz.cols) / 2;


        float mult = -0.5 / pow(sigma,2);

        for (int i = 0; i < sz.rows; i++) {
            for (int j = 0; j < sz.cols; j++) {
                int ih = i - syh;
                int jh = j - sxh;
                res(i, j) = std::exp(mult * (float) (ih * ih + jh * jh));
            }
        }
        return res;
    }

    void KCF::ExtractFeatures(const Mat &frm, const Rect &roi, Mat& feat, FeatureSize &featSz) const {
        Rect paddedRoi;

        float cx = roi.x + roi.width / 2;
        float cy = roi.y + roi.height / 2;

        int paddedW = roi.width * Padding;
        int paddedH = roi.height * Padding;
        float scale;
        Size tmplSz;

        if (TmplMode==TMPL_MODE_FIXED) {
            // Fit largest dimension to the given length
            if (roi.width >= roi.height)  //fit to width
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

        Mat z = RectTools::subwindow(frm, paddedRoi, BORDER_REPLICATE);
        resize(z, z, tmplSz);
        feat=Feat->Extract(z,featSz);
    }

    // Initialize Hanning window. Function called only in the first frame.
    Mat KCF::CalcHann(const FeatureSize &sz) {
        Mat hann1t = Mat(Size(sz.cols, 1), CV_32F, Scalar(0));
        Mat hann2t = Mat(Size(1, sz.rows), CV_32F, Scalar(0));

        for (int i = 0; i < hann1t.cols; i++)
            hann1t.at<float>(0, i) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann1t.cols - 1)));
        for (int i = 0; i < hann2t.rows; i++)
            hann2t.at<float>(i, 0) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann2t.rows - 1)));

        Mat hann2d = hann2t * hann1t;

        Mat hann1d = hann2d.reshape(1, 1);
        Mat hann = Mat(Size(sz.rows * sz.cols, sz.cns), CV_32F, Scalar(0));
        for (int i = 0; i < sz.cns; i++) {
            for (int j = 0; j < sz.rows * sz.cols; j++) {
                hann.at<float>(i, j) = hann1d.at<float>(0, j);
            }
        }
        return hann;
    }

    // Calculate sub-pixel peak for one dimension
    float KCF::CalcSubPixelPeak(float left, float center, float right) {
        float divisor = 2 * center - right - left;

        if (divisor == 0)
            return 0;

        return 0.5 * (right - left) / divisor;
    }

    KCF::KCF(FeatureType ft, KernelType kt) {
        FeatType=ft;
        KrnlType=kt;
    }

    bool KCF::Init(const Mat &frm, const Rect& roi) {
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
            TmplMode=TMPL_MODE_FIXED;       // Template size must be fixed when scaling
//            ScaleStep = 1.05;
//            ScaleWeight = 0.95;
        }
        else {
//            ScaleStep = 1;
//            ScaleWeight = 1;
        }

        assert(roi.width >= 0 && roi.height >= 0);
        Roi = roi;

        Mat x;
        ExtractFeatures(frm, Roi, x, FeatSz);
        Mat hann = CalcHann(FeatSz);
        x = hann.mul(x);

        float outputSigma = std::sqrt((float) FeatSz.rows * FeatSz.cols) / Padding * OutputSigmaFactor;
        Mat y = CalcGaussianMap(FeatSz,outputSigma);

        ModelY_f=FFTTools::fftd(y);
        ModelInit(x);
        return true;
    }

    Rect KCF::Track(const Mat &frame) {
        return Track(frame,true,true);
    }

    Rect KCF::Track(const Mat &frm, bool updateModel, bool updateRoi) {
        Rect roi = Roi;
        // Keep the roi with at least 1 pixel width and height
        // TODO: roi validate function
        if (roi.x + roi.width <= 0) roi.x = -roi.width + 1;
        if (roi.y + roi.height <= 0) roi.y = -roi.height + 1;
        if (roi.x >= frm.cols - 1) roi.x = frm.cols - 2;
        if (roi.y >= frm.rows - 1) roi.y = frm.rows - 2;

        float cx = roi.x + roi.width / 2.0f;
        float cy = roi.y + roi.height / 2.0f;

        float pv;
        Mat z;
        ExtractFeatures(frm, roi, z , FeatSz);
        Point2f res = Detect(ModelX, z, pv);
        if(EnableScale) {
            // TODO: To handle scaling problem
        }

        cx += (res.x * Feat->CellSize);
        cy += (res.y * Feat->CellSize);
        roi.x = cx - roi.width / 2.0f;
        roi.y = cy - roi.height / 2.0f;


        // TODO: roi validate function
        if (roi.x >= frm.cols - 1) roi.x = frm.cols - 1;
        if (roi.y >= frm.rows - 1) roi.y = frm.rows - 1;
        if (roi.x + roi.width <= 0) roi.x = -roi.width + 2;
        if (roi.y + roi.height <= 0) roi.y = -roi.height + 2;
        assert(roi.width >= 0 && roi.height >= 0);

        if(updateRoi) {
            Roi=roi;
        }

        if(updateModel) {
            Mat x;
            ExtractFeatures(frm, roi, x, FeatSz);
            ModelUpdate(x);
        }

        return roi;
    }

    Point2f KCF::Detect(const Mat &x, const Mat &z, float &pv) const{
        using namespace FFTTools;

        Mat res = EvalResMap(z,x);
        Point2i _pl;
        double _pv;
        minMaxLoc(res, NULL, &_pv, NULL, &_pl);

        pv = (float) _pv;
        Point2f pl((float) _pl.x, (float) _pl.y);

        if (_pl.x > 0 && _pl.x < res.cols - 1) {
            pl.x += CalcSubPixelPeak(res.at<float>(_pl.y, _pl.x - 1), _pv, res.at<float>(_pl.y, _pl.x + 1));
        }
        if (_pl.y > 0 && _pl.y < res.rows - 1) {
            pl.y += CalcSubPixelPeak(res.at<float>(_pl.y - 1, _pl.x), _pv, res.at<float>(_pl.y + 1, _pl.x));
        }

        pl.x -= (res.cols) / 2;
        pl.y -= (res.rows) / 2;

        return pl;
    }

    Mat KCF::EvalResMap(const Mat& x, const Mat& z) const {
        using namespace FFTTools;
        Mat k = Krnl->Correlation(z, x, FeatSz);
        Mat res = real(fftd(complexMultiplication(ModelAlpha_f, fftd(k)), true));
        return res;
    }

    void KCF::ModelInit(const Mat& x) {
        using namespace FFTTools;
        ModelUpdate(x,1.0f);
    }

    void KCF::ModelUpdate(const Mat &x, float lr) {
        using namespace FFTTools;
        Mat k = Krnl->Correlation(x, x, FeatSz);
        Mat k_f=fftd(k);

        Mat alpha_f = complexDivision(ModelY_f, k_f + Lambda);

        ModelAlpha_f    = (1 - lr) * ModelAlpha_f   + lr * alpha_f;
        ModelX          = (1 - lr) * ModelX         + lr * x;
    }

    void KCF::ModelUpdate(const Mat &x) {
        ModelUpdate(x,LearningRate);
    }

}