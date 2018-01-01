# zKCF
zKCF is an extensible C++ implementation of KCF(Kernelized Correlation Filters) visual tracker.   
This project is mainly based on the code of [KCFcpp](https://github.com/joaofaro/KCFcpp) and [KCF Matlab](http://www.robots.ox.ac.uk/~joao/circulant/)[1][2].   
In zKCF, the implementation of KCF's main body, feature extractors and correlation kernels are seperated, implementation is refined and reorganized for code readability and extensibility. Moreover, with parameters tuning, zKCF obtain a slightly better performance and a faster speed.

## Usage
### Compilation
zKCF's dependencies include CMake(>3.0) and OpenCV(2/3).   
Compilation follows an ordinary procedure of CMake project and is tested under Ubuntu 16.04.   
```bash
mkdir build
cd build
cmake ..
make -j
```
Ubuntu 14.04 and older Linux OSs may have older CMake(2.x) and need to be updated. Windows should cooperate with Visual Studio and configure the dependencies. MacOS is expected to have a smooth compilation as under Ubuntu 16.04.

### Run
The `main` function in `Run.cpp` is default for OTB[3] datasets and should be called as:
```bash
./zKCF seq_name /path/to/seq/ start_frame end_frame zero_padding ext bbox_x bbox_y bbox_width bbox_height preview
``` 
A `Basketball` sequence is prepared in  `assets/seqs` for demo. The demo is called as:
```bash
./zKCF car4 /home/zeke/Documents/OTB/CVPR13/Car4/img/ 1 659 4 jpg 69 50 107 87 1
``` 

## Project Structure
* `src/Run.cpp`: `main` function.
* `src(include)/KCF`: Main class of `zKCF`.
* `include/Def.h`: Definitions of common constant variables.
* `src(include)/Features`: Feature extractors. All implement the interface `IFeature.h`.
    * `RawFeature`: To use raw pixels as features.
    * `HogFeature`: HOG feature extractor.
* `src(include)/Kernels`: Correlation kernels. All implement the interface `IKernel.h`.
    * `GaussianKernel`: Gaussian correlation kernel.
* `src(include)/FkFactory`: Factory class, generating and configuring different features and kernels.

## How to extend zKCF
To add a new feature or kernel,
1. Put implementation codes in corresponding directories `src(include)/Features` and `src(include)/Kernels`.   
2. Define a new `FeatureType`/`KernelType` in `include/Def.h`.   
3. Customize parameters initialization in `src/FkFactory.cpp`.   

Since features and kernels are based on interface `IFeature.h` and `IKernel.h`, implementation details are hidden in `KCF` class, which focusing on the pipeline of the tracker instead of features and kernels.   
Nevertheless, parameters can be initialized for different features and kernels in `ParamsInit` method of `KCF` class.

## TODOs
* Features
    * HogLabFeature
    * CNN Feature
* Kernels
    * Linear kernel
    * Polynomial kernel

## References
[1] Henriques, J. F., et al. "High-Speed Tracking with Kernelized Correlation Filters." IEEE Transactions on Pattern Analysis & Machine Intelligence 37.3(2015):583-596.
[2] Rui, Caseiro, P. Martins, and J. Batista. "Exploiting the circulant structure of tracking-by-detection with kernels." European Conference on Computer Vision Springer-Verlag, 2012:702-715.
[3] Wu, Yi, J. Lim, and M. Yang. "Online Object Tracking: A Benchmark Supplemental Material." 9.4(2013):2411-2418.