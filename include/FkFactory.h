/*
Author: iXez
Website: https://ixez.github.io
Email: sachika.misawa@outlook.com
*/
#pragma once

#include "Def.h"

#include "Features/IFeature.h"
#include "Features/HogFeature.h"
#include "Features/HogLabFeature.h"
#include "Features/RawFeature.h"

#include "Kernels/IKernel.h"
#include "Kernels/GaussianKernel.h"

namespace zkcf {
    void FkFactory(FeatureType ft, KernelType kt, IFeature*& f, IKernel*& k);
}