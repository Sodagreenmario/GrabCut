//
// Created by Sodagreenmario on 2020-05-17.
//

#ifndef GRABCUT_GRABCUT_H
#define GRABCUT_GRABCUT_H

#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "AdaptedGraph.h"
#include "GMM.h"

static double calcBeta( const Mat& img );
static double calcNWeights( const Mat& img, Mat& leftW, Mat& upleftW, Mat& upW, Mat& uprightW, double beta, double gamma );
static void checkMask( const Mat& img, const Mat& mask );
static void initMaskWithRect( Mat& mask, Size imgSize, Rect rect );
static void initGMMs( const Mat& img, const Mat& mask, GMM& bgdGMM, GMM& fgdGMM );
static void assignGMMsComponents( const Mat& img, const Mat& mask, const GMM& bgdGMM, const GMM& fgdGMM, Mat& compIdxs );
static void learnGMMs( const Mat& img, const Mat& mask, const Mat& compIdxs, GMM& bgdGMM, GMM& fgdGMM );
static double constructGCGraph( const Mat& img, const Mat& mask, const GMM& bgdGMM, const GMM& fgdGMM, double lambda,
                              const Mat& leftW, const Mat& upleftW, const Mat& upW, const Mat& uprightW,
                              AdaptedGrpah& graph );
static int estimateSegmentation(AdaptedGrpah& graph, Mat& mask );

void GrabCut( InputArray _img, InputOutputArray _mask, Rect rect,
              InputOutputArray _bgdModel, InputOutputArray _fgdModel,
              int iterCount, int mode );

#endif //GRABCUT_GRABCUT_H
