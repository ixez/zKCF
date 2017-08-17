#include "Kernel/GaussianKernel.h"
#include "FFTTools.hpp"

namespace zkcf {
    Mat GaussianKernel::Correlation(const Mat &x1, const Mat &x2, const FeatureSize &sz) const {
        using namespace FFTTools;
        Mat c = Mat(Size(sz.x, sz.y), CV_32F, Scalar(0));

        Mat caux;
        Mat x1aux;
        Mat x2aux;
        for (int i = 0; i < sz.cn; i++) {
            x1aux = x1.row(i).reshape(1, sz.y);
            x2aux = x2.row(i).reshape(1, sz.y);
            mulSpectrums(fftd(x1aux), fftd(x2aux), caux, 0, true);
            caux = fftd(caux, true);
            rearrange(caux);
            caux.convertTo(caux, CV_32F);
            c = c + real(caux);
        }

        Mat d;
        max(((sum(x1.mul(x1))[0] + sum(x2.mul(x2))[0]) - 2. * c) / (sz.y * sz.x * sz.cn), 0, d);

        Mat k;
        exp(-1 / (Sigma * Sigma) * d, k);
        return k;
    }
}
