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
    // The weight of data
    double operator()( const Vec3d color ) const;
    // The possibility that a color belongs to a component
    double operator()( int ci, const Vec3d color ) const;
    // Assign the color to a component
    int whichComponent( const Vec3d color ) const;
    // Initialize the data before learning
    void initLearning();
    // Add a sample point
    void addSample( int ci, const Vec3d color );
    // Calculate the result according to the added data
    void endLearning();

private:
    // Calculate the inverse and determ
    void calcInverseCovAndDeterm(int ci);
    Mat model;
    // The coefficient, mean, cov of each gaussian distribution in GMM.
    double *coefs, *mean, *cov;
    // The inverse of Covs
    double inverseCovs[componentsCount][3][3];
    // The determ of Covs
    double covDeterms[componentsCount];

    double sums[componentsCount][3];
    double prods[componentsCount][3][3];
    int sampleCounts[componentsCount];
    int totalSampleCount;
};

#endif //GRABCUT_GMM_H
