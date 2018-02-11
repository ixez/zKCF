/*
Author: iXez
Website: https://ixez.github.io
Email: sachika.misawa@outlook.com
*/
#include "FkFactory.h"
#include "KCF.h"
#include "FFTTools.hpp"
#include "recttools.hpp"
#include "Run.h"
#include <gflags/gflags.h>

DEFINE_double(padding, -1.0, "Padding ratio of search area");
DEFINE_double(learningRate, -1.0, "Learning rate of the correlation filter");
DEFINE_double(outputSigmaFactor, -1.0, "OutputSigmaFactor of Y");
DEFINE_double(scaleWeight, -1.0, "Scale weight when a different scale produce a higher response score");

namespace zkcf {
    using namespace cv;

    KCF::KCF(){
        FeatType = FEAT_HOG;
        KrnlType = KRNL_GAUSSIAN;
        ScaleRatio = 1;
    }

    KCF::KCF(FeatureType ft, KernelType kt) : KCF() {
        FeatType = ft;
        KrnlType = kt;
    }

    Mat KCF::CalcGaussianMap(const FeatureSize &sz, float sigma) {
        Mat_<float> res(sz.rows, sz.cols);

        int syh = (sz.rows) / 2;
        int sxh = (sz.cols) / 2;


        float mult = -0.5 / pow(sigma, 2);

        for (int i = 0; i < sz.rows; i++) {
            for (int j = 0; j < sz.cols; j++) {
                int ih = i - syh;
                int jh = j - sxh;
                res(i, j) = std::exp(mult * (float) (ih * ih + jh * jh));
            }
        }
        return res;
    }

    void KCF::ExtractFeatures(const Mat &frm, const Rect_<float> &roi, Mat &feat, FeatureSize &featSz, float scale) {
        Rect pRoi;

        float cx = roi.x + roi.width / 2.0f;
        float cy = roi.y + roi.height / 2.0f;

        pRoi.width = PaddedSz.width * ScaleRatio * scale;
        pRoi.height = PaddedSz.height * ScaleRatio * scale;
        pRoi.x = cx - pRoi.width / 2;
        pRoi.y = cy - pRoi.height / 2;

        Mat z = RectTools::subwindow(frm, pRoi, BORDER_REPLICATE);
        if (z.size() != TmplSz) resize(z, z, TmplSz);
        feat = Feat->Extract(z, featSz);

#ifndef NDEBUG
        Dz = z.clone();
        DpRoi = frm.clone();
        rectangle(DpRoi, pRoi, CV_RGB(255,0,0));
#endif
    }

    Mat KCF::CalcHann(const FeatureSize &sz) {
        Mat hann1t = Mat(Size(sz.cols, 1), CV_32F, Scalar(0));
        Mat hann2t = Mat(Size(1, sz.rows), CV_32F, Scalar(0));

        for (int i = 0; i < hann1t.cols; i++)
            hann1t.at<float>(0, i) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann1t.cols - 1)));
        for (int i = 0; i < hann2t.rows; i++)
            hann2t.at<float>(i, 0) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann2t.rows - 1)));

        Mat hann2d = hann2t * hann1t;

        Mat hann1d = hann2d.reshape(1, 1);
        Mat hann = Mat(Size(sz.rows * sz.cols, sz.chns), CV_32F, Scalar(0));
        for (int i = 0; i < sz.chns; i++) {
            for (int j = 0; j < sz.rows * sz.cols; j++) {
                hann.at<float>(i, j) = hann1d.at<float>(0, j);
            }
        }
        return hann;
    }

    float KCF::CalcSubPixelPeak(float left, float center, float right) {
        float divisor = 2 * center - right - left;

        if (divisor == 0)
            return 0;

        return 0.5 * (right - left) / divisor;
    }

    bool KCF::Init(const Mat &frm, const Rect &roi) {
        assert(roi.width >= 0 && roi.height >= 0);
        ParamsInit();
        FkFactory(FeatType, KrnlType, Feat, Krnl);

        Roi = roi;

        ScalesInit();
        TmplInit();

        Mat x;
        ExtractFeatures(frm, Roi, x, FeatSz);
        Hann = CalcHann(FeatSz);
        x = Hann.mul(x);

        float outputSigma = std::sqrt((float) FeatSz.rows * FeatSz.cols) / Padding * OutputSigmaFactor;
        Mat y = CalcGaussianMap(FeatSz, outputSigma);
        ModelY_f = FFTTools::fftd(y);

        ModelInit(x);
        return true;
    }

    Rect KCF::Track(const Mat &frame) {
        return Track(frame, true, true);
    }

    Rect KCF::Track(const Mat &frm, bool updateModel, bool updateRoi) {
        Rect_<float> roi = Roi;
        // Keep the roi with at least 1 pixel width and height
        // TODO: roi validate function
        if (roi.x + roi.width <= 0) roi.x = -roi.width + 1;
        if (roi.y + roi.height <= 0) roi.y = -roi.height + 1;
        if (roi.x >= frm.cols - 1) roi.x = frm.cols - 2;
        if (roi.y >= frm.rows - 1) roi.y = frm.rows - 2;

        float cx = roi.x + roi.width / 2.0f;
        float cy = roi.y + roi.height / 2.0f;

        Point2f res;
        float pv = -numeric_limits<float>::max();
        float scale = 1.f;
        for (float _scale:ScaleList) {
            Mat z;
            float _pv;
            Rect_<float> _roi = roi;
            RectTools::resize(_roi, _scale);
            ExtractFeatures(frm, _roi, z, FeatSz, _scale);
            assert(Hann.size() == z.size());

            z = Hann.mul(z);

            Rect pRoi;

            float cx = roi.x + roi.width / 2.0f;
            float cy = roi.y + roi.height / 2.0f;

            pRoi.width = PaddedSz.width * ScaleRatio * scale;
            pRoi.height = PaddedSz.height * ScaleRatio * scale;
            pRoi.x = cx - pRoi.width / 2;
            pRoi.y = cy - pRoi.height / 2;
            Point2f _res = Detect(ModelX, z, _pv);

            if (_scale != 1.0f) _pv *= ScaleWeight;
            if (_pv > pv) {
                scale = _scale;
                pv = _pv;
                res = _res;
            }

#ifndef NDEBUG
            vector<Mat> dFeatMats;
            for(int c=0; c<3; c++) {
                dFeatMats.push_back(z.row(c).reshape(1,FeatSz.rows));
            }
            merge(dFeatMats,Dfeat);
            resize(Dfeat,Dfeat,Size(300,300));
            resize(Dres, Dres, Size(300,300));
            resize(Dz, Dz, Size(300,300));

            cvtColor(Dres,Dres,CV_GRAY2BGR);
            Dres.convertTo(Dres, CV_32FC3);
            Dz.convertTo(Dz, CV_32FC3);
            Mat dres, dz;
            dz = Dz/255.f;
            addWeighted(Dres, 1, dz, 0.1, 0, dres);

            Mat debug;
            hconcat(dres, Dres, debug);
            hconcat(debug, Dfeat, debug);
            imshow("Debug", debug);
#endif
        }

        cx += (res.x * Feat->FeatureRatio.width * TmplRatio * ScaleRatio * scale);
        cy += (res.y * Feat->FeatureRatio.height * TmplRatio * ScaleRatio * scale);

        roi.x = cx - roi.width / 2.0f;
        roi.y = cy - roi.height / 2.0f;
        roi.width *= scale;
        roi.height *= scale;

        // TODO: roi validate function
        if (roi.x >= frm.cols - 1) roi.x = frm.cols - 1;
        if (roi.y >= frm.rows - 1) roi.y = frm.rows - 1;
        if (roi.x + roi.width <= 0) roi.x = -roi.width + 2;
        if (roi.y + roi.height <= 0) roi.y = -roi.height + 2;
        assert(roi.width >= 0 && roi.height >= 0);

        if (updateRoi) {
            Roi = roi;
            ScaleRatio *= scale;
        }

        if (updateModel) {
            Mat x;
            ExtractFeatures(frm, roi, x, FeatSz);
            Mat hann = CalcHann(FeatSz);
            x = hann.mul(x);
            ModelUpdate(x, LearningRate);
        }

        return roi;
    }

    Point2f KCF::Detect(const Mat &x, const Mat &z, float &pv) {
        // since some features are much smaller than original img size, return subpixel location makes sense.
        using namespace FFTTools;

        Mat res = EvalResMap(x, z);

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


#ifndef NDEBUG
        Dres = res.clone();
#endif

        return pl;
    }

    Mat KCF::EvalResMap(const Mat &x, const Mat &z) const {
        using namespace FFTTools;
        Mat k = Krnl->Correlation(z, x, FeatSz);
        Mat res = real(fftd(complexMultiplication(ModelAlpha_f, fftd(k)), true));
        return res;
    }

    void KCF::ModelInit(const Mat &x) {
        using namespace FFTTools;
        ModelUpdate(x, 1.0f);
    }

    void KCF::ModelUpdate(const Mat &x, float lr) {
        using namespace FFTTools;
        Mat k = Krnl->Correlation(x, x, FeatSz);
        Mat k_f = fftd(k);

        Mat alpha_f = complexDivision(ModelY_f, k_f + Lambda);

        if (lr == 1) {
            ModelAlpha_f = alpha_f;
            ModelX = x;
        } else {
            ModelAlpha_f = (1 - lr) * ModelAlpha_f + lr * alpha_f;
            ModelX = (1 - lr) * ModelX + lr * x;
        }
    }

    void KCF::ParamsInit() {
        TmplMode = TMPL_MODE_CUSTOM;
        // Scales
        EnableScale = true;
        ScaleN = 1;
        ScaleStep = 0.05;
        ScaleWeight = 0.95;

        TmplLen = 96;
        Padding = 3.6;
        Lambda = 0.0001;
        OutputSigmaFactor = 0.125;
        LearningRate = 0.012;
        switch (FeatType) {
            case FEAT_HOG:
                Padding = 3.4;
                break;
            case FEAT_HOG_LAB:
                TmplLen = 128;
                Padding = 3.0;
                LearningRate = 0.005;
                OutputSigmaFactor = 0.1;
                break;
            case FEAT_RAW:
                LearningRate = 0.075;
                Padding = 3.0;
            case FEAT_GRAY:break;
            case FEAT_VGG:
                OutputSigmaFactor = 0.1;
                LearningRate = 0.0001;
                Padding = 3 ;
                TmplMode = TMPL_MODE_ROI_SZ;
                EnableScale = false;
                break;
        }

    }

    void KCF::TmplInit() {
        PaddedSz.width = Roi.width * Padding;
        PaddedSz.height = Roi.height * Padding;

        if (TmplMode == TMPL_MODE_CUSTOM) {
            // Fit largest dimension to the given length
            if (Roi.width >= Roi.height)  //fit to width
                TmplRatio = PaddedSz.width / (float) TmplLen;
            else
                TmplRatio = PaddedSz.height / (float) TmplLen;

            TmplSz.width = PaddedSz.width / TmplRatio;
            TmplSz.height = PaddedSz.height / TmplRatio;
        } else if (TmplMode == TMPL_MODE_ROI_SZ) {  //No template size given, use ROI size
            TmplSz.width = PaddedSz.width;
            TmplSz.height = PaddedSz.height;
            TmplRatio = 1;
        } else {
            std::cout << "Unknown template mode." << std::endl;
        }

        if (FeatType == FEAT_HOG || FeatType == FEAT_HOG_LAB) {
            int cellSize = ((HogFeature*)Feat)->CellSize;
            // Round to cell size and also make it even (Hog is sensitive on this)
            TmplSz.width = TmplSz.width / (2 * cellSize) * (2 * cellSize);
            TmplSz.height = TmplSz.height / (2 * cellSize) * (2 * cellSize);
            TmplSz.width += cellSize * 2;
            TmplSz.height += cellSize * 2;
        } else {
        }

        PaddedSz.width = TmplSz.width * TmplRatio;
        PaddedSz.height = TmplSz.height * TmplRatio;
    }

    void KCF::ScalesInit() {
        ScaleN = EnableScale ? ScaleN : 0;
        for (int i = -ScaleN; i <= ScaleN; i++) {
            ScaleList.push_back(1 + ScaleStep * i);
        }
    }
}