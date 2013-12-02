#ifndef UTIL_H
#define UTIL_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <stdio.h>
#include <string.h>

using namespace std;
using namespace cv;

void readInputList(FILE* ifile, vector<string>& nameArray, vector<Mat>& homoArray);

Point2d homoTransform(Point2d p, InputArray homo);

unsigned long getUnixTime();

#endif
