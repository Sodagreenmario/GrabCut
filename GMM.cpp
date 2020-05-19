//
// Created by Sodagreenmario on 2020-05-17.
// Reference : https://github.com/opencv/opencv/blob/master/modules/imgproc/src/grabcut.cpp
//

#include "GMM.h"

GMM::GMM( Mat& _model )
{
    const int modelSize = 3/*mean*/ + 9/*covariance*/ + 1/*component weight*/;
    if( _model.empty() )
    {
        _model.create( 1, modelSize*componentsCount, CV_64FC1 );
        _model.setTo(Scalar(0));
    }
    else if( (_model.type() != CV_64FC1) || (_model.rows != 1) || (_model.cols != modelSize*componentsCount) )
        CV_Error( CV_StsBadArg, "_model must have CV_64FC1 type, rows == 1 and cols == 13*componentsCount" );

    model = _model;

    coefs = model.ptr<double>(0);
    mean = coefs + componentsCount;
    cov = mean + 3 * componentsCount;

    for( int ci = 0; ci < componentsCount; ci++ )
        if( coefs[ci] > 0 )
            calcInverseCovAndDeterm(ci);
    totalSampleCount = 0;
}

/*

    double GMM::operator()( const Vec3d color ) const;

    Input paramters:
      cv::Vec3d color :  the color vector.

    Output paramters:
      double res : tweight.
*/
double GMM::operator()( const Vec3d color ) const
{
    double res = 0;
    for( int ci = 0; ci < componentsCount; ci++ )
        res += coefs[ci] * (*this)(ci, color );
    return res;
}

double GMM::operator()( int ci, const Vec3d color ) const
{
    double res = 0;
    if( coefs[ci] > 0 )
    {
        CV_Assert( covDeterms[ci] > std::numeric_limits<double>::epsilon() );
        Vec3d diff = color;
        double* m = mean + 3*ci;
        diff[0] -= m[0]; diff[1] -= m[1]; diff[2] -= m[2];
        double mult = diff[0]*(diff[0]*inverseCovs[ci][0][0] + diff[1]*inverseCovs[ci][1][0] + diff[2]*inverseCovs[ci][2][0])
                      + diff[1]*(diff[0]*inverseCovs[ci][0][1] + diff[1]*inverseCovs[ci][1][1] + diff[2]*inverseCovs[ci][2][1])
                      + diff[2]*(diff[0]*inverseCovs[ci][0][2] + diff[1]*inverseCovs[ci][1][2] + diff[2]*inverseCovs[ci][2][2]);
        res = 1.0f/sqrt(covDeterms[ci]) * exp(-0.5f*mult);
    }
    return res;
}

/*
    Chose a component for a color vector.
    int GMM::whichComponent( const Vec3d color ) const;

    Input paramters:
      cv::Vec3d color : a color vector.

    Output parameters:
      int k : the index of the chosen components.
*/
int GMM::whichComponent( const Vec3d color ) const
{
    int k = 0;
    double max = 0;

    for( int ci = 0; ci < componentsCount; ci++ )
    {
        double p = (*this)( ci, color );
        if( p > max )
        {
            k = ci;
            max = p;
        }
    }
    return k;
}

/*
    Initialize 'sums', 'prods', 'sampleCounts', 'totalSampleCount' for the GMM model.
    void GMM::initLearning();
*/
void GMM::initLearning()
{
    for( int ci = 0; ci < componentsCount; ci++)
    {
        sums[ci][0] = sums[ci][1] = sums[ci][2] = 0;
        prods[ci][0][0] = prods[ci][0][1] = prods[ci][0][2] = 0;
        prods[ci][1][0] = prods[ci][1][1] = prods[ci][1][2] = 0;
        prods[ci][2][0] = prods[ci][2][1] = prods[ci][2][2] = 0;
        sampleCounts[ci] = 0;
    }
    totalSampleCount = 0;
}

/*
    Add a sample into GMM model.
    void GMM::addSample( int ci, const Vec3d color )；

    Input parameters:
      int ci : the index of the component.
      cv::Vec3d color : a color vector.
*/
void GMM::addSample( int ci, const Vec3d color )
{
    sums[ci][0] += color[0]; sums[ci][1] += color[1]; sums[ci][2] += color[2];
    prods[ci][0][0] += color[0]*color[0]; prods[ci][0][1] += color[0]*color[1]; prods[ci][0][2] += color[0]*color[2];
    prods[ci][1][0] += color[1]*color[0]; prods[ci][1][1] += color[1]*color[1]; prods[ci][1][2] += color[1]*color[2];
    prods[ci][2][0] += color[2]*color[0]; prods[ci][2][1] += color[2]*color[1]; prods[ci][2][2] += color[2]*color[2];
    sampleCounts[ci]++;
    totalSampleCount++;
}

/*
    Update the parameters in GMM.
    (1) coef
    (2) mean
    (3) cov
    void GMM::endLearning();
*/
void GMM::endLearning()
{
    for( int ci = 0; ci < componentsCount; ci++ )
    {
        int n = sampleCounts[ci];
        if( n == 0 )
            coefs[ci] = 0;
        else
        {
            CV_Assert(totalSampleCount > 0);
            double inv_n = 1.0 / n;
            // (1) coef
            coefs[ci] = (double)n/totalSampleCount;

            double* m = mean + 3*ci;
            // (2) mean
            m[0] = sums[ci][0] * inv_n; m[1] = sums[ci][1] * inv_n; m[2] = sums[ci][2] * inv_n;

            double* c = cov + 9*ci;
            // (3) cov
            c[0] = prods[ci][0][0] * inv_n - m[0]*m[0]; c[1] = prods[ci][0][1] * inv_n - m[0]*m[1]; c[2] = prods[ci][0][2] * inv_n - m[0]*m[2];
            c[3] = prods[ci][1][0] * inv_n - m[1]*m[0]; c[4] = prods[ci][1][1] * inv_n - m[1]*m[1]; c[5] = prods[ci][1][2] * inv_n - m[1]*m[2];
            c[6] = prods[ci][2][0] * inv_n - m[2]*m[0]; c[7] = prods[ci][2][1] * inv_n - m[2]*m[1]; c[8] = prods[ci][2][2] * inv_n - m[2]*m[2];

            double dtrm = c[0]*(c[4]*c[8]-c[5]*c[7]) - c[1]*(c[3]*c[8]-c[5]*c[6]) + c[2]*(c[3]*c[7]-c[4]*c[6]);
            double singularFix = 0.01;
            if (dtrm <= std::numeric_limits<double>::epsilon()  && singularFix > 0)
            {
                // Adds the white noise to avoid singular covariance matrix.
                c[0] += singularFix;
                c[4] += singularFix;
                c[8] += singularFix;
            }

            calcInverseCovAndDeterm(ci);
        }
    }
}

/*

    Calculate the inverse of Cov and Det.
    void GMM::calcInverseCovAndDeterm(int ci)

    Input parameters:
      int ci : the index of the component.
      double singularFix : the white noise to avoid singular covariance matrix.
*/
void GMM::calcInverseCovAndDeterm(int ci)
{
    if( coefs[ci] > 0 )
    {
        double *c = cov + 9*ci;
        double dtrm = c[0]*(c[4]*c[8]-c[5]*c[7]) - c[1]*(c[3]*c[8]-c[5]*c[6]) + c[2]*(c[3]*c[7]-c[4]*c[6]);
        covDeterms[ci] = dtrm;
        double inv_dtrm = 1.0 / dtrm;
        // calculate the inverse
        inverseCovs[ci][0][0] =  (c[4]*c[8] - c[5]*c[7]) * inv_dtrm;
        inverseCovs[ci][1][0] = -(c[3]*c[8] - c[5]*c[6]) * inv_dtrm;
        inverseCovs[ci][2][0] =  (c[3]*c[7] - c[4]*c[6]) * inv_dtrm;
        inverseCovs[ci][0][1] = -(c[1]*c[8] - c[2]*c[7]) * inv_dtrm;
        inverseCovs[ci][1][1] =  (c[0]*c[8] - c[2]*c[6]) * inv_dtrm;
        inverseCovs[ci][2][1] = -(c[0]*c[7] - c[1]*c[6]) * inv_dtrm;
        inverseCovs[ci][0][2] =  (c[1]*c[5] - c[2]*c[4]) * inv_dtrm;
        inverseCovs[ci][1][2] = -(c[0]*c[5] - c[2]*c[3]) * inv_dtrm;
        inverseCovs[ci][2][2] =  (c[0]*c[4] - c[1]*c[3]) * inv_dtrm;
    }
}