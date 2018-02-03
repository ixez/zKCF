// HogLabFeature is not complete yet.
#include <Def.h>
#include "Features/HogLabFeature.h"

namespace zkcf {

    HogLabFeature::HogLabFeature()
    {
        int nClusters=15;
        float clusters[nClusters][3] = {
            {161.317504, 127.223401, 128.609333},
            {142.922425, 128.666965, 127.532319},
            {67.879757, 127.721830, 135.903311},
            {92.705062, 129.965717, 137.399500},
            {120.172257, 128.279647, 127.036493},
            {195.470568, 127.857070, 129.345415},
            {41.257102, 130.059468, 132.675336},
            {12.014861, 129.480555, 127.064714},
            {226.567086, 127.567831, 136.345727},
            {154.664210, 131.676606, 156.481669},
            {121.180447, 137.020793, 153.433743},
            {87.042204, 137.211742, 98.614874},
            {113.809537, 106.577104, 157.818094},
            {81.083293, 170.051905, 148.904079},
            {45.015485, 138.543124, 102.402528}
        };
        LabCentroids = Mat(nClusters, 3, CV_32FC1, &clusters).clone();
    }

    Mat HogLabFeature::Extract(const Mat &patch, FeatureSize &sz) {
        Mat feat=HogFeature::Extract(patch, sz);
        cv::Mat lab;
        cvtColor(patch, lab, CV_BGR2Lab);
        unsigned char *input = (unsigned char*)(lab.data);

        // Sparse output vector
        cv::Mat outputLab = cv::Mat(LabCentroids.rows, sz.rows*sz.cols, CV_32F, float(0));

        int cntCell = 0;
        // Iterate through each cell
        for (int cY = CellSize; cY < patch.rows-CellSize; cY+=CellSize){
            for (int cX = CellSize; cX < patch.cols-CellSize; cX+=CellSize){
                // Iterate through each pixel of cell (cX,cY)
                for(int y = cY; y < cY+CellSize; ++y){
                    for(int x = cX; x < cX+CellSize; ++x){
                        // Lab components for each pixel
                        float l = input[(patch.cols * y + x) * 3];
                        float a = input[(patch.cols * y + x) * 3 + 1];
                        float b = input[(patch.cols * y + x) * 3 + 2];

                        // Iterate trough each centroid
                        float minDist = FLT_MAX;
                        int minIdx = 0;
                        float *inputCentroid = (float*)(LabCentroids.data);
                        for(int k = 0; k < LabCentroids.rows; ++k){
                            float dist = ( (l - inputCentroid[3*k]) * (l - inputCentroid[3*k]) )
                                         + ( (a - inputCentroid[3*k+1]) * (a - inputCentroid[3*k+1]) )
                                         + ( (b - inputCentroid[3*k+2]) * (b - inputCentroid[3*k+2]) );
                            if(dist < minDist){
                                minDist = dist;
                                minIdx = k;
                            }
                        }
                        // Store result at output
                        outputLab.at<float>(minIdx, cntCell) += 1.0 / (CellSize * CellSize);
                        //((float*) outputLab.data)[minIdx * (size_patch[0]*size_patch[1]) + cntCell] += 1.0 / cell_sizeQ;
                    }
                }
                cntCell++;
            }
        }

        // Update size_patch[2] and add features to FeaturesMap
        sz.chns += LabCentroids.rows;
        feat.push_back(outputLab);
        return feat;
    }
}
