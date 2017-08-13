#pragma once

#include "Def.h"
#include "Kernel/IKernel.h"
#include "Feature/IFeature.h"
#include <opencv2/opencv.hpp>

namespace zkcf {
    using namespace cv;
    class KCF {
    public:
        virtual Rect update(Mat image);


        int cell_sizeQ; // cell size^2, to avoid repeated operations
        float ScaleStep; // scale step for multi-scale estimation
        float ScaleWeight;  // to downweight detection scores of other scales for added stability
    protected:
        // Detect object in the current frame.
        Point2f detect(Mat z, Mat x, float &peak_value);

        // train tracker with a single image
        void train(Mat x, float train_interp_factor);

        // Evaluates a Gaussian kernel with bandwidth SIGMA for all relative shifts between input images X and Y, which must both be MxN. They must    also be periodic (ie., pre-processed with a cosine window).
        Mat gaussianCorrelation(Mat x1, Mat x2);

        // Obtain sub-window from image, with replication-padding and extract features


        // Initialize Hanning window. Function called only in the first frame.


        // Calculate sub-pixel peak for one dimension
        float subPixelPeak(float left, float center, float right);

        Mat _prob;

        Mat _num;
        Mat _den;


    private:
        int size_patch[3];

        Size _tmpl_sz;
        float _scale;
        int _gaussian_size;
        bool _hogfeatures;
        bool _labfeatures;

    public:
        bool EnableScale = true;

        typedef enum {
            TMPL_MODE_FIXED = 1,   // Longer edge will resize to this length and exctract features
            TMPL_MODE_NONE = 0     // Not resize, keep size
        } eTemplateMode;
        eTemplateMode TmplMode;
        int TmplLen = 96;          // Available when TEMPLATE_MODE_FIXED

        KCF(FeatureType ft=FEAT_HOG, KernelType kt=KRNL_GAUSSIAN);
        void Init(const Mat &frm, Rect roi);

    private:

        float LearningRate;
        float Lambda;
        float Padding;
        float OutputSigmaFactor;

        Rect Roi;

        FeatureType FeatType;
        KernelType KrnlType;
        IFeature* Feature = nullptr;
        IKernel* Kernel = nullptr;

        FeatureSize FeatSz;
        Mat ModelYf;
        Mat ModelAlphaF;
        Mat ModelXf;

        static Mat CalcHann(const FeatureSize &sz);
        static Mat CalcGaussianMap(const FeatureSize& sz, float sigma);

        void ModelInit(const Mat &x);
        Mat GetFeatures(const Mat &patch, FeatureSize& featSz) const;



    };
}
