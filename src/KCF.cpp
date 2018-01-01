/*
Author: iXez
Website: https://ixez.github.io
Email: sachika.misawa@outlook.com
*/
#include "FkFactory.h"
#include "KCF.h"
#include "FFTTools.hpp"
#include "recttools.hpp"

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

    void KCF::ExtractFeatures(const Mat &frm, const Rect_<float> &roi, Mat &feat, FeatureSize &featSz) const {
        Rect paddedRoi;

        float cx = roi.x + roi.width / 2.0f;
        float cy = roi.y + roi.height / 2.0f;

        // Different from origin
        paddedRoi.width = roi.width * Padding;
        paddedRoi.height = roi.height * Padding;

        // center roi with new size
        paddedRoi.x = cx - paddedRoi.width / 2;
        paddedRoi.y = cy - paddedRoi.height / 2;

        Mat z = RectTools::subwindow(frm, paddedRoi, BORDER_REPLICATE);
        if (z.size() != TmplSz) resize(z, z, TmplSz);
        feat = Feat->Extract(z, featSz);
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
        Mat hann = Mat(Size(sz.rows * sz.cols, sz.cns), CV_32F, Scalar(0));
        for (int i = 0; i < sz.cns; i++) {
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

        Mat _frm = frm.clone();
        Point2f res;
        float pv = -numeric_limits<float>::max();
        float scale = 1.f;
        for (float _scale:ScaleList) {
            Mat z;
            float _pv;
            Rect_<float> _roi = roi;
            RectTools::resize(_roi, _scale);
            ExtractFeatures(frm, _roi, z, FeatSz);
            assert(Hann.size() == z.size());
            z = Hann.mul(z);
            Point2f _res = Detect(ModelX, z, _pv);

            if (_scale != 1.0f) _pv *= ScaleWeight;
            if (_pv > pv) {
                scale = _scale;
                pv = _pv;
                res = _res;
            }

            rectangle(_frm, _roi, CV_RGB(255, 255, 255), 1);
        }

        cx += (res.x * Feat->CellSize * TmplRatio * ScaleRatio * scale);
        cy += (res.y * Feat->CellSize * TmplRatio * ScaleRatio * scale);

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

    Point2f KCF::Detect(const Mat &x, const Mat &z, float &pv) const {
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
        TmplLen = 128;

        // Scales
        EnableScale = true;
        ScaleN = 1;
        ScaleStep = 0.05;
        ScaleWeight = 0.95;

        Lambda = 0.0001;
        Padding = 2.5;
        OutputSigmaFactor = 0.125;
        switch (FeatType) {
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
    }

    void KCF::TmplInit() {
        int paddedW = Roi.width * Padding;
        int paddedH = Roi.height * Padding;

        if (TmplMode == TMPL_MODE_CUSTOM) {
            // Fit largest dimension to the given length
            if (Roi.width >= Roi.height)  //fit to width
                TmplRatio = paddedW / (float) TmplLen;
            else
                TmplRatio = paddedH / (float) TmplLen;

            TmplSz.width = paddedW / TmplRatio;
            TmplSz.height = paddedH / TmplRatio;
        } else if (TmplMode == TMPL_MODE_ROI_SZ) {  //No template size given, use ROI size
            TmplSz.width = paddedW;
            TmplSz.height = paddedH;
            TmplRatio = 1;
        } else {
            std::cout << "Unknown template mode." << std::endl;
        }

        // Round to cell size and also make it even (Hog is sensitive on this)
        TmplSz.width = TmplSz.width / (2 * Feat->CellSize) * (2 * Feat->CellSize);
        TmplSz.height = TmplSz.height / (2 * Feat->CellSize) * (2 * Feat->CellSize);

        if (FeatType == FEAT_HOG || FeatType == FEAT_HOG_LAB) {
            TmplSz.width += Feat->CellSize * 2;
            TmplSz.height += Feat->CellSize * 2;
        } else {
            //Make number of pixels even (helps with some logic involving half-dimensions)
        }
    }

    void KCF::ScalesInit() {
        ScaleN = EnableScale ? ScaleN : 0;
        for (int i = -ScaleN; i <= ScaleN; i++) {
            ScaleList.push_back(1 + ScaleStep * i);
        }
    }
}