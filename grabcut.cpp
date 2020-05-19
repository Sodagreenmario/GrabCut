//
// Created by Sodagreenmario on 2020-05-17
// Reference : https://github.com/opencv/opencv/blob/master/modules/imgproc/src/grabcut.cpp
//

#include "grabcut.h"
#include "AdaptedGraph.h"
#include "GMM.h"

/* Test */
#include <iostream>
/* Test */

using namespace std;
using namespace cv;

/*

    Calculate beta - parameter of GrabCut algorithm.
    Euclidean distance is applied to evaluate the neighbouring pixels. The difference between GraphCut and GrabCut
is resulted by the different color space(gray and colorful).

    beta = 1 / ( 2 * avg( sqr(||color[i] - color[j]||) ) )

    static double calcBeta( const Mat& img );

    Input parameters:
      cv::Mat img : the input image.

    Output parameters:
      double beta : the constant that encourages smoothness.

*/
static double calcBeta( const Mat& img )
{
    double beta = 0;
    for( int y = 0; y < img.rows; y++ )
    {
        for( int x = 0; x < img.cols; x++ )
        {
            Vec3d color = img.at<Vec3b>(y,x);
            if( x>0 ) // left
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y,x-1);
                beta += diff.dot(diff);
            }
            if( y>0 && x>0 ) // upleft
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x-1);
                beta += diff.dot(diff);
            }
            if( y>0 ) // up
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x);
                beta += diff.dot(diff);
            }
            if( y>0 && x<img.cols-1) // upright
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x+1);
                beta += diff.dot(diff);
            }
        }
    }
    if( beta <= std::numeric_limits<double>::epsilon() )
        beta = 0;
    else
        beta = 1.f / (2 * beta/(4*img.cols*img.rows - 3*img.cols - 3*img.rows + 2) );

    return beta;
}

/*

    Calculate weights of noterminal vertices of graph. 'V' in paper.

    static void calcNWeights( const Mat& img, Mat& leftW, Mat& upleftW, Mat& upW, Mat& uprightW, double beta, double gamma );

    Input parameters:
      double beta : the constant that encourages smoothness.
      double gamma : the constant that determines the degree of smoothness.

    Output parameters:
      cv::InputOutputArray _mask : The segmentation result.

 */
static void calcNWeights( const Mat& img, Mat& leftW, Mat& upleftW, Mat& upW, Mat& uprightW, double beta, double gamma )
{

    const double gammaDivSqrt2 = gamma / std::sqrt(2.0f);
    leftW.create( img.rows, img.cols, CV_64FC1 );
    upleftW.create( img.rows, img.cols, CV_64FC1 );
    upW.create( img.rows, img.cols, CV_64FC1 );
    uprightW.create( img.rows, img.cols, CV_64FC1 );
    for( int y = 0; y < img.rows; y++ )
    {
        for( int x = 0; x < img.cols; x++ )
        {
            Vec3d color = img.at<Vec3b>(y,x);
            if( x-1>=0 ) // left
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y,x-1);
                leftW.at<double>(y,x) = gamma * exp(-beta*diff.dot(diff));
            }
            else
                leftW.at<double>(y,x) = 0;
            if( x-1>=0 && y-1>=0 ) // upleft
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x-1);
                upleftW.at<double>(y,x) = gammaDivSqrt2 * exp(-beta*diff.dot(diff));
            }
            else
                upleftW.at<double>(y,x) = 0;
            if( y-1>=0 ) // up
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x);
                upW.at<double>(y,x) = gamma * exp(-beta*diff.dot(diff));
            }
            else
                upW.at<double>(y,x) = 0;
            if( x+1<img.cols && y-1>=0 ) // upright
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x+1);
                uprightW.at<double>(y,x) = gammaDivSqrt2 * exp(-beta*diff.dot(diff));
            }
            else
                uprightW.at<double>(y,x) = 0;
        }
    }
}

/*
  Check size, type and element values of mask matrix.
 */
static void checkMask( const Mat& img, const Mat& mask )
{
    if( mask.empty() )
        CV_Error( CV_StsBadArg, "mask is empty" );
    if( mask.type() != CV_8UC1 )
        CV_Error( CV_StsBadArg, "mask must have CV_8UC1 type" );
    if( mask.cols != img.cols || mask.rows != img.rows )
        CV_Error( CV_StsBadArg, "mask must have as many rows and cols as img" );
    for( int y = 0; y < mask.rows; y++ )
    {
        for( int x = 0; x < mask.cols; x++ )
        {
            uchar val = mask.at<uchar>(y,x);
            if( val!=GC_BGD && val!=GC_FGD && val!=GC_PR_BGD && val!=GC_PR_FGD )
                CV_Error( CV_StsBadArg, "mask element value must be equal "
                                        "GC_BGD or GC_FGD or GC_PR_BGD or GC_PR_FGD" );
        }
    }
}

/*

  Initialize mask using rectangular.
  (1) Outside of the rectangle -> background.
  (2) Inside of the rectangle  -> possible foreground.

  static void initMaskWithRect( Mat& mask, Size imgSize, Rect rect );

  Input parameters:
    Size imgSize : The size of the input image.
    Rect rect : The drawn rectangle.

  Output parameters:
    cv::Mat mask : The initialized mask.

*/
static void initMaskWithRect( Mat& mask, Size imgSize, Rect rect )
{
    mask.create( imgSize, CV_8UC1 );
    // (1) Outside of the rectangle -> background.
    mask.setTo( GC_BGD );

    rect.x = std::max(0, rect.x);
    rect.y = std::max(0, rect.y);
    rect.width = std::min(rect.width, imgSize.width-rect.x);
    rect.height = std::min(rect.height, imgSize.height-rect.y);

    // (2) Inside of the rectangle  -> possible foreground.
    (mask(rect)).setTo( Scalar(GC_PR_FGD) );
}

/*
  Initialize GMM background and foreground models using kmeans algorithm.

  static void initGMMs( const Mat& img, const Mat& mask, GMM& bgdGMM, GMM& fgdGMM );

  Input parameters:
    cv::Mat img :
    cv::Mat mask :

  Output parameters:
    cv::Mat bgdGMM :
    cv::Mat fgdGMM :
*/
static void initGMMs( const Mat& img, const Mat& mask, GMM& bgdGMM, GMM& fgdGMM )
{
    const int kMeansItCount = 10;
    const int kMeansType = KMEANS_PP_CENTERS;

    Mat bgdLabels, fgdLabels;
    std::vector<Vec3f> bgdSamples, fgdSamples;
    Point p;
    for( p.y = 0; p.y < img.rows; p.y++ )
    {
        for( p.x = 0; p.x < img.cols; p.x++ )
        {
            if( mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD )
                bgdSamples.push_back( (Vec3f)img.at<Vec3b>(p) );
            else // GC_FGD | GC_PR_FGD
                fgdSamples.push_back( (Vec3f)img.at<Vec3b>(p) );
        }
    }
    CV_Assert( !bgdSamples.empty() && !fgdSamples.empty() );
    {
        Mat _bgdSamples( (int)bgdSamples.size(), 3, CV_32FC1, &bgdSamples[0][0] );
        int num_clusters = GMM::componentsCount;
        num_clusters = std::min(num_clusters, (int)bgdSamples.size());
        kmeans( _bgdSamples, num_clusters, bgdLabels,
                TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType );
    }
    {
        Mat _fgdSamples( (int)fgdSamples.size(), 3, CV_32FC1, &fgdSamples[0][0] );
        int num_clusters = GMM::componentsCount;
        num_clusters = std::min(num_clusters, (int)fgdSamples.size());
        kmeans( _fgdSamples, num_clusters, fgdLabels,
                TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType );
    }

    bgdGMM.initLearning();
    for( int i = 0; i < (int)bgdSamples.size(); i++ )
        bgdGMM.addSample( bgdLabels.at<int>(i,0), bgdSamples[i] );
    bgdGMM.endLearning();

    fgdGMM.initLearning();
    for( int i = 0; i < (int)fgdSamples.size(); i++ )
        fgdGMM.addSample( fgdLabels.at<int>(i,0), fgdSamples[i] );
    fgdGMM.endLearning();
}

/*
  Assign GMMs components for each pixel.
*/
static void assignGMMsComponents( const Mat& img, const Mat& mask, const GMM& bgdGMM, const GMM& fgdGMM, Mat& compIdxs )
{
    Point p;
    for( p.y = 0; p.y < img.rows; p.y++ )
    {
        for( p.x = 0; p.x < img.cols; p.x++ )
        {
            Vec3d color = img.at<Vec3b>(p);
            compIdxs.at<int>(p) = mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD ?
                                  bgdGMM.whichComponent(color) : fgdGMM.whichComponent(color);
        }
    }
}

/*
  Learn GMMs parameters.
*/
static void learnGMMs( const Mat& img, const Mat& mask, const Mat& compIdxs, GMM& bgdGMM, GMM& fgdGMM )
{
    bgdGMM.initLearning();
    fgdGMM.initLearning();
    Point p;
    for( int ci = 0; ci < GMM::componentsCount; ci++ )
    {
        for( p.y = 0; p.y < img.rows; p.y++ )
        {
            for( p.x = 0; p.x < img.cols; p.x++ )
            {
                if( compIdxs.at<int>(p) == ci )
                {
                    if( mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD )
                        bgdGMM.addSample( ci, img.at<Vec3b>(p) );
                    else
                        fgdGMM.addSample( ci, img.at<Vec3b>(p) );
                }
            }
        }
    }
    bgdGMM.endLearning();
    fgdGMM.endLearning();
}

/*
  Construct GCGraph
*/
static void constructGCGraph( const Mat& img, const Mat& mask, const GMM& bgdGMM, const GMM& fgdGMM, double lambda,
                              const Mat& leftW, const Mat& upleftW, const Mat& upW, const Mat& uprightW,
                              AdaptedGrpah& graph )
{
    int vtxCount = img.cols*img.rows,
            edgeCount = 2*(4*img.cols*img.rows - 3*(img.cols + img.rows) + 2);
    graph = AdaptedGrpah(vtxCount, edgeCount);
    Point p;
    for( p.y = 0; p.y < img.rows; p.y++ )
    {
        for( p.x = 0; p.x < img.cols; p.x++)
        {
            // add node
            int vtxIdx = graph.addVtx();
//            cout << vtxIdx;
            Vec3b color = img.at<Vec3b>(p);
            // set t-weights
            double fromSource, toSink;
            if( mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD )
            {
                fromSource = -log( bgdGMM(color) );
                toSink = -log( fgdGMM(color) );
            }
            else if( mask.at<uchar>(p) == GC_BGD )
            {
                fromSource = 0;
                toSink = lambda;
            }
            else // GC_FGD
            {
                fromSource = lambda;
                toSink = 0;
            }
//            cout << "B";
            graph.addTermWeights( vtxIdx, fromSource, toSink );
//            cout << "C";
            // set n-weights
            if( p.x>0 )
            {
                double w = leftW.at<double>(p);
                graph.addEdges( vtxIdx, vtxIdx-1, w, w );
            }
            if( p.x>0 && p.y>0 )
            {
                double w = upleftW.at<double>(p);
                graph.addEdges( vtxIdx, vtxIdx-img.cols-1, w, w );
            }
            if( p.y>0 )
            {
                double w = upW.at<double>(p);
                graph.addEdges( vtxIdx, vtxIdx-img.cols, w, w );
            }
            if( p.x<img.cols-1 && p.y>0 )
            {
                double w = uprightW.at<double>(p);
                graph.addEdges( vtxIdx, vtxIdx-img.cols+1, w, w );
            }
        }
    }
}

/*
  Estimate segmentation using MaxFlow algorithm
*/
static void estimateSegmentation(AdaptedGrpah& graph, Mat& mask )
{
    cout << "E1";
    graph.maxFlow();
    cout << "E2";
    Point p;
    for( p.y = 0; p.y < mask.rows; p.y++ )
    {
        for( p.x = 0; p.x < mask.cols; p.x++ )
        {
            if( mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD )
            {
                if( graph.inSourceSegment(p.y*mask.cols+p.x))
                    mask.at<uchar>(p) = GC_PR_FGD;
                else
                    mask.at<uchar>(p) = GC_PR_BGD;
            }
        }
    }
}

/*
    void GrabCut( InputArray _img, InputOutputArray _mask, Rect rect,
                      InputOutputArray _bgdModel, InputOutputArray _fgdModel,
                      int iterCount, int mode )

    Input parameters:
      cv::InputArray _img : The input image
      cv::Rect rect : The drawn rectangle in the image
      int iterCount : The times of iterations in each segmentation

    Output parameters:
      cv::InputOutputArray _mask : The segmentation result.

    Intermediate parameters:
      cv::InputOutputArray _bgdModel : Background model.
      cv::InputOutputArray _fgdModel : Foreground model.
*/
void GrabCut( InputArray _img, InputOutputArray _mask, Rect rect,
                  InputOutputArray _bgdModel, InputOutputArray _fgdModel,
                  int iterCount, int mode )
{
//    CV_INSTRUMENT_REGION();
    // 1. Load input image.
    Mat img = _img.getMat();
    Mat& mask = _mask.getMatRef();
    Mat& bgdModel = _bgdModel.getMatRef();
    Mat& fgdModel = _fgdModel.getMatRef();

    if( img.empty() )
        CV_Error( CV_StsBadArg, "image is empty" );
    if( img.type() != CV_8UC3 )
        CV_Error( CV_StsBadArg, "image must have CV_8UC3 type" );

    GMM bgdGMM( bgdModel ), fgdGMM( fgdModel );
    Mat compIdxs( img.size(), CV_32SC1 );

    if( mode == GC_INIT_WITH_RECT || mode == GC_INIT_WITH_MASK )
    {
        // 2. Initialize the mask. Labels: (background: 0, foreground: 1, possible background: 2, possible foreground: 3)
        //    Outside of the rectangle -> background            /  0
        //    Inside of the rectangle  -> possible foreground   /  3
        if( mode == GC_INIT_WITH_RECT )  initMaskWithRect( mask, img.size(), rect );
        else if (mode == GC_INIT_WITH_MASK)   checkMask( img, mask );
        // 3. Sample points and clustering with K-Means.
        initGMMs( img, mask, bgdGMM, fgdGMM );
    }

    if( iterCount <= 0)
        return;

    if( mode == GC_EVAL_FREEZE_MODEL )
        iterCount = 1;

    if( mode == GC_EVAL || mode == GC_EVAL_FREEZE_MODEL )
        checkMask( img, mask );

    // 4. Construct graph. Calculate the terminal-weight term and non-terminal-weight term.
    const double gamma = 50;
    const double lambda = 9*gamma;
    // 4.1.1 Calculate the beta term.
    const double beta = calcBeta( img );
    Mat leftW, upleftW, upW, uprightW;
    // 4.1.2 Calculate the non-terminal weights.
    calcNWeights( img, leftW, upleftW, upW, uprightW, beta, gamma );
    for( int i = 0; i < iterCount; i++ )
    {
        AdaptedGrpah graph;
        assignGMMsComponents( img, mask, bgdGMM, fgdGMM, compIdxs );
        if( mode != GC_EVAL_FREEZE_MODEL )
            learnGMMs( img, mask, compIdxs, bgdGMM, fgdGMM );
        constructGCGraph(img, mask, bgdGMM, fgdGMM, lambda, leftW, upleftW, upW, uprightW, graph );
        estimateSegmentation( graph, mask );
    }
}