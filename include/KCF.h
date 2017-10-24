#pragma once

#include "Def.h"
#include "Kernel/IKernel.h"
#include "Feature/IFeature.h"
#include "ITracker.h"
#include <opencv2/opencv.hpp>

namespace zkcf {
    using namespace cv;
    class KCF : ztrack::ITracker {
    public:
        //// Constants
        typedef enum {
            TMPL_MODE_CUSTOM = 1,     // Longer edge will resize to this length and exctract features
            TMPL_MODE_ROI_SZ = 0     // Not resize, keep size
        } eTemplateMode;

        //// Configuration
        // Template
        eTemplateMode TmplMode = TMPL_MODE_CUSTOM;
        int TmplLen = 96;          // Used when TEMPLATE_MODE_FIXED
        // Scale
        bool EnableScale = true;
        int ScaleN = 0;
        float ScaleStep = 0.05;
        float ScaleWeight = 0.95;

        //// Methods
        explicit KCF(FeatureType ft = FEAT_HOG, KernelType kt = KRNL_GAUSSIAN);
        bool Init(const Mat &frm, const Rect& roi) override;
        Rect Track(const Mat &frm) override;
        Rect Track(const Mat &frm, bool updateModel, bool updateRoi);

        // Predict location of the target,
        // since some features are much smaller than original img size, return subpixel location makes sense.
        Point2f Detect(const Mat &x, const Mat &z, float &pv) const;

    private:
        // padded_sz / TemplLen
        float TmplRatio;
        // Padded roi will be resize to this template size and then be extracted feature
        Size TmplSz;

        vector<float> ScaleList;
        float ScaleRatio;

        float LearningRate;
        float Lambda;
        float Padding;
        float OutputSigmaFactor;

        Rect_<float> Roi;

        FeatureType FeatType;
        KernelType KrnlType;
        IFeature* Feat = nullptr;
        IKernel* Krnl = nullptr;

        FeatureSize FeatSz;
        Mat ModelY_f;
        Mat ModelAlpha_f;
        Mat ModelX;

        static Mat CalcHann(const FeatureSize &sz);
        static Mat CalcGaussianMap(const FeatureSize& sz, float sigma);         // Generate Gaussian Peak. Function called only in the first frame.
        static float CalcSubPixelPeak(float left, float center, float right);   // Calculate sub-pixel peak for one dimension

        void TmplInit();   //  Part of Init()

        void ModelInit(const Mat &x);
        void ModelUpdate(const Mat &x, float lr);

        // Extract feature maps of roi which is padded and resized to specified template size.
        void ExtractFeatures(const Mat &frm, const Rect_<float> &roi, CV_OUT Mat &feat, CV_OUT FeatureSize &featSz) const;

        // Eval response map, x => tmpl, z => test image patch
        Mat EvalResMap(const Mat &x, const Mat &z) const;

        void ScaleInit();
    };
}
