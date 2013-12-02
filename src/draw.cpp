#include "draw.h"

void cross(Mat& img, int x, int y, Scalar color)
{
  line(img, Point(x-5, y-5), Point(x+5, y+5), color);
  line(img, Point(x+5, y-5), Point(x-5, y+5), color);
}

