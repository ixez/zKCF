/*
Author: iXez
Website: https://ixez.github.io
Email: sachika.misawa@outlook.com
*/
#include "Kernels/GaussianKernel.h"
#include "FFTTools.hpp"

namespace zkcf {
    Mat GaussianKernel::Correlation(const Mat &x1, const Mat &x2, const FeatureSize &sz) const {
        using namespace FFTTools;
        Mat c = Mat(Size(sz.cols, sz.rows), CV_32F, Scalar(0));

        vector<Mat> caux(sz.chns);
        vector<Mat> x1aux(sz.chns);
        vector<Mat> x2aux(sz.chns);

        #pragma omp parallel for
        for (int i = 0; i < sz.chns; i++) {
            x1aux[i] = x1.row(i).reshape(1, sz.rows);
            x2aux[i] = x2.row(i).reshape(1, sz.rows);
            mulSpectrums(fftd(x1aux[i]), fftd(x2aux[i]), caux[i], 0, true);
            caux[i] = fftd(caux[i], true);
            rearrange(caux[i]);
            caux[i].convertTo(caux[i], CV_32F);
        }

        for (int i = 0; i < sz.chns; i++) {
            c = c + real(caux[i]);
        }

        Mat d;
        max(((sum(x1.mul(x1))[0] + sum(x2.mul(x2))[0]) - 2. * c) / (sz.rows * sz.cols * sz.chns), 0, d);

        Mat k;
        exp(-d / (Sigma * Sigma), k);
        return k;
    }
}
