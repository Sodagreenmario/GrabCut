//
// Created by Sodagreenmario on 2020-05-17.
// Reference : https://github.com/opencv/opencv/blob/master/modules/imgproc/src/grabcut.cpp
//

#ifndef GRABCUT_GMM_H
#define GRABCUT_GMM_H

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/core/types_c.h>

using namespace std;
using namespace cv;

class GMM
{
public:
    static const int componentsCount = 5;

    GMM( Mat& _model );
    double operator()( const Vec3d color ) const;
    double operator()( int ci, const Vec3d color ) const;
    int whichComponent( const Vec3d color ) const;

    void initLearning();
    void addSample( int ci, const Vec3d color );
    void endLearning();

private:
    void calcInverseCovAndDeterm(int ci, double singularFix);
    Mat model;
    double* coefs;
    double* mean;
    double* cov;

    double inverseCovs[componentsCount][3][3];
    double covDeterms[componentsCount];

    double sums[componentsCount][3];
    double prods[componentsCount][3][3];
    int sampleCounts[componentsCount];
    int totalSampleCount;
};

#endif //GRABCUT_GMM_H
