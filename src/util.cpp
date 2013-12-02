#include "util.h"

#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
using namespace cv;

void readInputList(FILE* ifile, vector<string>& nameArray, vector<Mat>& homoArray)
{
  Mat homo(3, 3, CV_64F);
  homo.at<double>(2, 2) = 1.0;
  char buf[128];
  while (fscanf(ifile, "%s %lf %lf %lf %lf %lf %lf %lf %lf", buf,
      &homo.at<double>(0, 0), &homo.at<double>(0, 1), &homo.at<double>(0, 2),
      &homo.at<double>(1, 0), &homo.at<double>(1, 1), &homo.at<double>(1, 2),
      &homo.at<double>(2, 0), &homo.at<double>(2, 1)) == 9)
  {
    nameArray.push_back(string(buf));
    homoArray.push_back(homo.clone());
  }
}

Point2d homoTransform(Point2d p, InputArray H)
{
  Mat homo = H.getMat();
  Mat s(3, 1, CV_64FC1);
  s.at<double>(0) = p.x;
  s.at<double>(1) = p.y;
  s.at<double>(2) = 1;
  Mat d = homo * s;
  double rx = d.at<double>(0);
  double ry = d.at<double>(1);
  double r = d.at<double>(2);
  rx /= r;
  ry /= r;
  return Point2d((int) round(rx), (int) round(ry));
}

unsigned long getUnixTime()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}
