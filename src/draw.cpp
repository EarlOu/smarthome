#include "draw.h"

#define CROSS_SIZE 20

void cross(Mat& img, int x, int y, Scalar color)
{
  line(img, Point(x-CROSS_SIZE, y-CROSS_SIZE), Point(x+CROSS_SIZE, y+CROSS_SIZE), color);
  line(img, Point(x+CROSS_SIZE, y-CROSS_SIZE), Point(x-CROSS_SIZE, y+CROSS_SIZE), color);
}

