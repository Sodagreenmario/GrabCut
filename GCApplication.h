//
// Created by Sodagreenmario on 2020-05-17.
// Reference : https://docs.opencv.org/trunk/d8/d34/samples_2cpp_2grabcut_8cpp-example.html#a36
//

#ifndef GRABCUT_GCAPPLICATION_H
#define GRABCUT_GCAPPLICATION_H


#include "opencv2/imgcodecs.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;

const Scalar RED = Scalar(0,0,255);
const Scalar PINK = Scalar(230,130,255);
const Scalar BLUE = Scalar(255,0,0);
const Scalar LIGHTBLUE = Scalar(255,255,160);
const Scalar GREEN = Scalar(0,255,0);
const int BGD_KEY = EVENT_FLAG_CTRLKEY;
const int FGD_KEY = EVENT_FLAG_SHIFTKEY;

template<typename T> inline T sqr(T x) { return x * x; } // out of range risk for T = byte, ...
template<class T, int D> inline T vecSqrDist(const Vec<T, D> &v1, const Vec<T, D> &v2) {T s = 0; for (int i=0; i<D; i++) s += sqr(v1[i] - v2[i]); return s;} // out of range risk for T = byte, ...
template<class T, int D> inline T    vecDist(const Vec<T, D> &v1, const Vec<T, D> &v2) { return sqrt(vecSqrDist(v1, v2)); } // out of range risk for T = byte, ...

typedef vector<double> vecD;
typedef pair<float, int> CostfIdx;

static void getBinMask( const Mat& comMask, Mat& binMask )
{
    if( comMask.empty() || comMask.type()!=CV_8UC1 )
        CV_Error( Error::StsBadArg, "comMask is empty or has incorrect type (not CV_8UC1)" );
    if( binMask.empty() || binMask.rows!=comMask.rows || binMask.cols!=comMask.cols )
        binMask.create( comMask.size(), CV_8UC1 );
    binMask = comMask & 1;
}

class GCApplication
{
public:
    enum{ NOT_SET = 0, IN_PROCESS = 1, SET = 2 };
    static const int radius = 2;
    static const int thickness = -1;
    void reset();
    void setImageAndWinName( const Mat& _image, const string& _winName );
    void showImage() const;
    void mouseClick( int event, int x, int y, int flags, void* param );
    int nextIter();
    int getIterCount() const { return iterCount; }
    int Quantize(Mat& img3f, Mat &idx1i, Mat &_color3f, Mat &_colorNum, double ratio = 0.95, const int clrNums[3] = DefaultNums);
    Mat GetHC(Mat &img3f);
    void GetHC(Mat &binColor3f, Mat &colorNums1i, Mat &_colorSal);
    void SmoothSaliency(Mat &sal1f, float delta, const vector<vector<CostfIdx>> &similar);
    void SmoothSaliency(Mat &colorNum1i, Mat &sal1f, float delta, const vector<vector<CostfIdx>> &similar);
    void HCResult();
private:
    void setRectInMask();
    void setLblsInMask( int flags, Point p, bool isPr );
    const string* winName;
    const Mat* image;
    Mat mask;
    Mat bgdModel, fgdModel;
    uchar rectState, lblsState, prLblsState;
    bool isInitialized;
    Rect rect;
    vector<Point> fgdPxls, bgdPxls, prFgdPxls, prBgdPxls;
    int iterCount;
    static const int DefaultNums[3];
};

#endif //GRABCUT_GCAPPLICATION_H
