#include <KCF.h>
#include <Kernel/GaussianKernel.h>
#include <Feature/HogFeature.h>
#include <Feature/HogLabFeature.h>
#include "KCF.h"
#include "ffttools.hpp"
#include "recttools.hpp"
#include "fhog.hpp"
#include "labdata.hpp"

namespace zkcf {
    using namespace cv;
// Update position based on the new frame
    Rect KCF::update(Mat image) {
        if (Roi.x + Roi.width <= 0) Roi.x = -Roi.width + 1;
        if (Roi.y + Roi.height <= 0) Roi.y = -Roi.height + 1;
        if (Roi.x >= image.cols - 1) Roi.x = image.cols - 2;
        if (Roi.y >= image.rows - 1) Roi.y = image.rows - 2;

        float cx = Roi.x + Roi.width / 2.0f;
        float cy = Roi.y + Roi.height / 2.0f;


        float peak_value;
        Point2f res = detect(X, GetFeatures(image, 0, 1.0f), peak_value);

        if (ScaleStep != 1) {
            // Test at a smaller _scale
            float new_peak_value;
            Point2f new_res = detect(X, GetFeatures(image, 0, 1.0f / ScaleStep), new_peak_value);

            if (ScaleWeight * new_peak_value > peak_value) {
                res = new_res;
                peak_value = new_peak_value;
                _scale /= ScaleStep;
                Roi.width /= ScaleStep;
                Roi.height /= ScaleStep;
            }

            // Test at a bigger _scale
            new_res = detect(X, GetFeatures(image, 0, ScaleStep), new_peak_value);

            if (ScaleWeight * new_peak_value > peak_value) {
                res = new_res;
                peak_value = new_peak_value;
                _scale *= ScaleStep;
                Roi.width *= ScaleStep;
                Roi.height *= ScaleStep;
            }
        }

        // Adjust by cell size and _scale
        Roi.x = cx - Roi.width / 2.0f + ((float) res.x * CellSize * _scale);
        Roi.y = cy - Roi.height / 2.0f + ((float) res.y * CellSize * _scale);

        if (Roi.x >= image.cols - 1) Roi.x = image.cols - 1;
        if (Roi.y >= image.rows - 1) Roi.y = image.rows - 1;
        if (Roi.x + Roi.width <= 0) Roi.x = -Roi.width + 2;
        if (Roi.y + Roi.height <= 0) Roi.y = -Roi.height + 2;

        assert(Roi.width >= 0 && Roi.height >= 0);
        Mat x = GetFeatures(image, 0);
        train(x, LearningRate);

        return Roi;
    }


// Detect object in the current frame.
    Point2f KCF::detect(Mat z, Mat x, float &peak_value) {
        using namespace FFTTools;

        Mat k = gaussianCorrelation(x, z);
        Mat res = (real(fftd(complexMultiplication(_alphaf, fftd(k)), true)));

        //minMaxLoc only accepts doubles for the peak, and integer points for the coordinates
        Point2i pi;
        double pv;
        minMaxLoc(res, NULL, &pv, NULL, &pi);
        peak_value = (float) pv;

        //subpixel peak estimation, coordinates will be non-integer
        Point2f p((float) pi.x, (float) pi.y);

        if (pi.x > 0 && pi.x < res.cols - 1) {
            p.x += subPixelPeak(res.at<float>(pi.y, pi.x - 1), peak_value, res.at<float>(pi.y, pi.x + 1));
        }

        if (pi.y > 0 && pi.y < res.rows - 1) {
            p.y += subPixelPeak(res.at<float>(pi.y - 1, pi.x), peak_value, res.at<float>(pi.y + 1, pi.x));
        }

        p.x -= (res.cols) / 2;
        p.y -= (res.rows) / 2;

        return p;
    }

// train tracker with a single image
    void KCF::train(Mat x, float train_interp_factor) {
        using namespace FFTTools;

        Mat k = gaussianCorrelation(x, x);
        Mat alphaf = complexDivision(_prob, (fftd(k) + Lambda));

        X = (1 - train_interp_factor) * X + (train_interp_factor) * x;
        _alphaf = (1 - train_interp_factor) * _alphaf + (train_interp_factor) * alphaf;


        /*Mat kf = fftd(gaussianCorrelation(x, x));
        Mat num = complexMultiplication(kf, _prob);
        Mat den = complexMultiplication(kf, kf + lambda);

        _tmpl = (1 - train_interp_factor) * _tmpl + (train_interp_factor) * x;
        _num = (1 - train_interp_factor) * _num + (train_interp_factor) * num;
        _den = (1 - train_interp_factor) * _den + (train_interp_factor) * den;

        _alphaf = complexDivision(_num, _den);*/

    }

// Evaluates a Gaussian kernel with bandwidth SIGMA for all relative shifts between input images X and Y, which must both be MxN. They must    also be periodic (ie., pre-processed with a cosine window).
    Mat KCF::gaussianCorrelation(Mat x1, Mat x2) {
        using namespace FFTTools;
        Mat c = Mat(Size(size_patch[1], size_patch[0]), CV_32F, Scalar(0));
        // HOG features
        if (_hogfeatures) {
            Mat caux;
            Mat x1aux;
            Mat x2aux;
            for (int i = 0; i < size_patch[2]; i++) {
                x1aux = x1.row(i);   // Procedure do deal with Mat multichannel bug
                x1aux = x1aux.reshape(1, size_patch[0]);
                x2aux = x2.row(i).reshape(1, size_patch[0]);
                mulSpectrums(fftd(x1aux), fftd(x2aux), caux, 0, true);
                caux = fftd(caux, true);
                rearrange(caux);
                caux.convertTo(caux, CV_32F);
                c = c + real(caux);
            }
        }
            // Gray features
        else {
            mulSpectrums(fftd(x1), fftd(x2), c, 0, true);
            c = fftd(c, true);
            rearrange(c);
            c = real(c);
        }
        Mat d;
        max(((sum(x1.mul(x1))[0] + sum(x2.mul(x2))[0]) - 2. * c) /
                (size_patch[0] * size_patch[1] * size_patch[2]), 0, d);

        Mat k;
        exp((-d / (Sigma * Sigma)), k);
        return k;
    }

// Create Gaussian Peak. Function called only in the first frame.
    Mat KCF::createGaussianPeak(int sizey, int sizex) {
        Mat_<float> res(sizey, sizex);

        int syh = (sizey) / 2;
        int sxh = (sizex) / 2;

        float output_sigma = std::sqrt((float) sizex * sizey) / Padding * OutputSigmaFactor;
        float mult = -0.5 / (output_sigma * output_sigma);

        for (int i = 0; i < sizey; i++)
            for (int j = 0; j < sizex; j++) {
                int ih = i - syh;
                int jh = j - sxh;
                res(i, j) = std::exp(mult * (float) (ih * ih + jh * jh));
            }
        return FFTTools::fftd(res);
    }

// Obtain sub-window from image, with replication-padding and extract features
    Mat KCF::GetFeatures(const Mat &patch) const {
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

        if(FeatType==IFeature::HOG || FeatType==IFeature::HOG_LAB) {
            // Round to cell size and also make it even
            tmplSz.width+=Feature->CellSize*2;
            tmplSz.height+=Feature->CellSize*2;
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
        IFeature::Sz featSz;
        Mat feat=Feature->Extract(z,featSz);

        // HOG features
        if (_hogfeatures) {
        } else {
            feat = RectTools::getGrayImage(z);
            feat -= (float) 0.5; // In Paper;
            size_patch[0] = z.rows;
            size_patch[1] = z.cols;
            size_patch[2] = 1;
        }

        if (inithann) {
            createHanningMats();
        }
        feat = hann.mul(feat);
        return feat;
    }

// Initialize Hanning window. Function called only in the first frame.
    void KCF::createHanningMats() {
        Mat hann1t = Mat(Size(size_patch[1], 1), CV_32F, Scalar(0));
        Mat hann2t = Mat(Size(1, size_patch[0]), CV_32F, Scalar(0));

        for (int i = 0; i < hann1t.cols; i++)
            hann1t.at<float>(0, i) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann1t.cols - 1)));
        for (int i = 0; i < hann2t.rows; i++)
            hann2t.at<float>(i, 0) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann2t.rows - 1)));

        Mat hann2d = hann2t * hann1t;
        // HOG features
        if (_hogfeatures) {
            Mat hann1d = hann2d.reshape(1, 1); // Procedure do deal with Mat multichannel bug

            hann = Mat(Size(size_patch[0] * size_patch[1], size_patch[2]), CV_32F, Scalar(0));
            for (int i = 0; i < size_patch[2]; i++) {
                for (int j = 0; j < size_patch[0] * size_patch[1]; j++) {
                    hann.at<float>(i, j) = hann1d.at<float>(0, j);
                }
            }
        }
            // Gray features
        else {
            hann = hann2d;
        }
    }

// Calculate sub-pixel peak for one dimension
    float KCF::subPixelPeak(float left, float center, float right) {
        float divisor = 2 * center - right - left;

        if (divisor == 0)
            return 0;

        return 0.5 * (right - left) / divisor;
    }

    KCF::KCF(IFeature::Type ft, IKernel::Type kt) {
        FeatType=ft;
        KernelType=kt;
    }

    void KCF::Init(const Mat &frm, Rect roi) {
        Lambda = 0.0001;
        Padding = 2.5;
        OutputSigmaFactor = 0.125;
        switch(FeatType) {
            case IFeature::HOG:
                LearningRate = 0.012;
                Sigma = 0.6;
                Feature=new HogFeature(KernelType);
                break;
            case IFeature::HOG_LAB:
                LearningRate = 0.005;
                Sigma = 0.4;
                OutputSigmaFactor = 0.1;
                Feature=new HogLabFeature(KernelType);
                break;
            case IFeature::RAW:
                LearningRate = 0.075;
                Sigma = 0.2;
//                Feature=new RawFeature(KernelType,CellSize);
                break;
        }
        Kernel=Feature->Kernel;
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
        X = GetFeatures(frm, 1);
        _prob = createGaussianPeak(size_patch[0], size_patch[1]);
        _alphaf = Mat(size_patch[0], size_patch[1], CV_32FC2, float(0));
        train(X, 1.0); // train with initial frame
    }
}