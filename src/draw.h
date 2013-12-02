#ifndef DRAW_H
#define DRAW_H

#include <opencv2/opencv.hpp>
using namespace cv;

const Scalar RED(0, 0, 255);
const Scalar GREEN(255, 255, 0);
const Scalar WHITE(255, 255, 255);


void cross(Mat& img, int x, int y, Scalar color);


#endif
