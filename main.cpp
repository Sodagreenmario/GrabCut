//
// Created by Sodagreenmario on 2020-05-17.
//

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "GCApplication.h"
#include <iostream>

using namespace std;
using namespace cv;

static void help(char** argv)
{
    cout << "\nThis program demonstrates GrabCut segmentation -- select an object in a region\n"
            "and then grabcut will attempt to segment it out.\n"
            "Call:\n"
         <<  argv[0] << " <image_name>\n"
                        "\nSelect a rectangular area around the object you want to segment\n" <<
         "\nHot keys: \n"
         "\tq - quit the program\n"
         "\tr - restore the original image\n"
         "\tn - next iteration\n"
         "\n"
         "\tleft mouse button - set rectangle\n"
         "\n"
         "\tCTRL+left mouse button - set GC_BGD pixels\n"
         "\tSHIFT+left mouse button - set GC_FGD pixels\n"
         "\n"
         "\tCTRL+right mouse button - set GC_PR_BGD pixels\n"
         "\tSHIFT+right mouse button - set GC_PR_FGD pixels\n" << endl;
}

GCApplication gcapp;

static void on_mouse( int event, int x, int y, int flags, void* param )
{
    gcapp.mouseClick( event, x, y, flags, param );
}

int main( int argc, char** argv )
{
    cv::CommandLineParser parser(argc, argv, "{@input| messi5.jpg |}");
    help(argv);
    string path = "/Users/mario/Downloads/GrabCut/GrabCut/";
    string filename = path + "test.jpg";
//    string filename = parser.get<string>("@input");
    if( filename.empty() )
    {
        cout << "\nDurn, empty filename" << endl;
        return 1;
    }
    Mat image = imread(samples::findFile(filename), IMREAD_COLOR);
    if( image.empty() )
    {
        cout << "\n Durn, couldn't read image filename " << filename << endl;
        return 1;
    }
    const string winName = "image";
    namedWindow( winName, WINDOW_AUTOSIZE );
    setMouseCallback( winName, on_mouse, 0 );
    gcapp.setImageAndWinName( image, winName );
    gcapp.showImage();
    for(;;)
    {
        char c = (char)waitKey(0);
        switch( c )
        {
            case 'q':
                cout << "Exiting ..." << endl;
                goto exit_main;
            case 'r':
                cout << endl;
                gcapp.reset();
                gcapp.showImage();
                break;
            case 'n':
                int iterCount = gcapp.getIterCount();
                cout << "<" << iterCount << "... ";
                int newIterCount = gcapp.nextIter();
                if( newIterCount > iterCount )
                {
                    gcapp.showImage();
                    cout << iterCount << ">" << endl;
                }
                else
                    cout << "rect must be determined>" << endl;
                break;
        }
    }
    exit_main:
    destroyWindow( winName );
    return 0;
}