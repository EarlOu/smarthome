#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <stdio.h>

using namespace cv;
using std::pair;

const Scalar RED(0, 0, 255);
const Scalar GREEN(255, 255, 0);

Mat H;
Mat frame;
Mat map;
vector<Point2f> srcPoints, dstPoints;

void cross(Mat& img, int x, int y, Scalar color)
{
  line(img, Point(x-5, y-5), Point(x+5, y+5), color);
  line(img, Point(x+5, y-5), Point(x-5, y+5), color);
}

static void onMouse(int event, int x, int y, int flag, void* param)
{
  Mat canvas = frame.clone();

  // Draw text
  char pose[128];
  sprintf(pose, "(%d, %d)", x, y);
  putText(canvas, pose, Point(0,230), FONT_HERSHEY_PLAIN, 1,
      RED, 1, 8, false);

  // Draw selected point
  for (int i=0, n = srcPoints.size(); i<n; ++i)
  {
    cross(canvas, srcPoints[i].x, srcPoints[i].y, GREEN);
  }
  // Draw cross
  cross(canvas, x, y, RED);

  imshow("img", canvas);

  if (!H.empty())
  {
    Mat s(3, 1, CV_64FC1);
    s.at<double>(0) = x;
    s.at<double>(1) = y;
    s.at<double>(2) = 1;
    Mat d = H*s;
    float rx = d.at<double>(0);
    float ry = d.at<double>(1);
    float r = d.at<double>(2);
    rx /= r;
    ry /= r;
    Mat mapCanvas = map.clone();
    cross(mapCanvas, ry, rx, RED);
    imshow("map", mapCanvas);
    return;
  }

  if (event != EVENT_LBUTTONDOWN) return;
  printf("Real location for (%d, %d):", x, y);
  int rx, ry;
  scanf("%d %d", &rx, &ry);
  srcPoints.push_back(Point2f(x, y));
  dstPoints.push_back(Point2f(rx, ry));
  if (srcPoints.size() == 4)
  {
    printf("Estimating homography transform...\n");
    H = findHomography(srcPoints, dstPoints);
    for (int y=0; y<3; ++y)
    {
      for (int x=0; x<3; ++x)
      {
        printf("%lf ", H.at<double>(y, x));
      }
      printf("\n");
    }
  }
}

int main(int argc, char *argv[])
{
  if (argc != 2)
  {
    printf("usage: %s <video>\n", argv[0]);
    return -1;
  }

  VideoCapture cap(argv[1]);
  cap.read(frame);
  map.create(Size(1000, 700), CV_8UC3);

  namedWindow("img");
  namedWindow("map");
  imshow("map", map);

  // Prepare parameters
  void** params = new void*[3];
  params[0] = &frame;
  params[1] = &srcPoints;
  params[2] = &dstPoints;
  setMouseCallback("img", onMouse, &frame);
  onMouse(CV_EVENT_MOUSEMOVE, 0, 0, 0, NULL);

  while(waitKey(0) != 'q');
}
