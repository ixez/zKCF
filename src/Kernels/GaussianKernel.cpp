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

        Mat caux;
        Mat x1aux;
        Mat x2aux;
        for (int i = 0; i < sz.cns; i++) {
            x1aux = x1.row(i).reshape(1, sz.rows);
            x2aux = x2.row(i).reshape(1, sz.rows);
            mulSpectrums(fftd(x1aux), fftd(x2aux), caux, 0, true);
            caux = fftd(caux, true);
            rearrange(caux);
            caux.convertTo(caux, CV_32F);
            c = c + real(caux);
        }

        Mat d;
        max(((sum(x1.mul(x1))[0] + sum(x2.mul(x2))[0]) - 2. * c) / (sz.rows * sz.cols * sz.cns), 0, d);

        Mat k;
        exp(-d / (Sigma * Sigma), k);
        return k;
    }
}
