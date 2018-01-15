/*
Author: iXez
Website: https://ixez.github.io
Email: sachika.misawa@outlook.com
*/
#pragma once

#include "Def.h"
#include "Kernels/IKernel.h"
#include "Features/IFeature.h"
#include "ITracker.h"
#include <opencv2/opencv.hpp>

namespace zkcf {
    using namespace cv;

    class KCF : ztrack::ITracker {
    public:
        //// Enums
        typedef enum {
            TMPL_MODE_CUSTOM = 1,    // Longer edge will resize to this length and exctract features
            TMPL_MODE_ROI_SZ = 0     // Not to resize, keep size
        } eTemplateMode;

        //// Configurable params
        // Template
        eTemplateMode TmplMode = TMPL_MODE_CUSTOM;
        int TmplLen;                // Used when TMPL_MODE_CUSTOM
        // Scale
        bool EnableScale;
        int ScaleN;
        float ScaleStep;
        float ScaleWeight;

        //// Constructor
        KCF();
        KCF(FeatureType ft, KernelType kt);

        //// Public methods
        bool Init(const Mat &frm, const Rect &roi) override;
        Rect Track(const Mat &frm) override;
        Rect Track(const Mat &frm, bool updateModel, bool updateRoi);

    private:
        float TmplRatio;            // padded_sz / TemplLen
        Size TmplSz;                // Padded roi will be resize to this template size and then be extracted to feature
        Size PaddedSz;              // Padded roi size

        vector<float> ScaleList;
        float ScaleRatio;

        float LearningRate;
        float Lambda;
        float Padding;
        float OutputSigmaFactor;

        Rect_<float> Roi;

        FeatureType FeatType;
        KernelType KrnlType;
        IFeature *Feat = nullptr;
        IKernel *Krnl = nullptr;

        FeatureSize FeatSz;
        Mat Hann;
        Mat ModelY_f;
        Mat ModelAlpha_f;
        Mat ModelX;

        //// Parts of Init()
        void ParamsInit();  // Parameters init, all configurable params are inited in this method
        void ScalesInit();  // Scales init
        void TmplInit();    // Template init

        //// Static methods
        static Mat CalcHann(const FeatureSize &sz);     // Initialize Hanning window. Function called only in the first frame.
        static Mat CalcGaussianMap(const FeatureSize &sz,
                                   float sigma);        // Generate Gaussian Peak. Function called only in the first frame.
        static float CalcSubPixelPeak(float left, float center, float right);   // Calculate sub-pixel peak for one dimension

        // Extract feature maps of roi which is padded and resized to specified template size.
        void ExtractFeatures(const Mat &frm, const Rect_<float> &roi, Mat &feat, FeatureSize &featSz, float scale = 1.0) const;

        // Eval response map, x => tmpl, z => test image patch
        Mat EvalResMap(const Mat &x, const Mat &z) const;

        // Predict location of the target,
        Point2f Detect(const Mat &x, const Mat &z, float &pv) const;

        // Model methods
        void ModelInit(const Mat &x);
        void ModelUpdate(const Mat &x, float lr);
    };
}
