#include "util.h"
#include "draw.h"

#include <opencv2/opencv.hpp>
#include <opencv2/ocl/ocl.hpp>
#include <opencv2/nonfree/gpu.hpp>
#include <stdio.h>
#include <vector>

using namespace cv;
using namespace std;

//#define OCL

#define SMOOTH_LENGTH 15
#define MESSAGE_PERIOD 75

int main(int argc, char *argv[])
{
  if (argc != 3)
  {
    printf("usage: %s <video_list> <map_img>\n", argv[0]);
    return -1;
  }
  FILE* ifile = fopen(argv[1], "r");
  vector<Mat> homoArray;
  vector<string> nameArray;
  readInputList(ifile, nameArray, homoArray);

  vector<VideoCapture> capArray;
  int height = 0;
  for (int i=0, n=nameArray.size(); i<n; ++i)
  {
    VideoCapture cap(nameArray[i]);
    if (!cap.isOpened())
    {
      perror("Failed to open video stream");
      return -1;
    }
    capArray.push_back(cap);
    namedWindow(nameArray[i]);
    moveWindow(nameArray[i], 1200, height);
    height += 300;
  }

#ifdef OCL
  ocl::DevicesInfo devices;
  ocl::getOpenCLDevices(devices);
  ocl::setDevice(devices[0]);
#endif

#ifdef OCL
  ocl::HOGDescriptor hog;
  hog.setSVMDetector(ocl::HOGDescriptor::getPeopleDetector48x96());
#else
  HOGDescriptor hog;
  hog.winSize=Size(48, 96);
  hog.setSVMDetector(HOGDescriptor::getDaimlerPeopleDetector());
//  hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
#endif

  Mat map = imread(argv[2]);
  Mat mapCanvas;
  namedWindow("map");
  moveWindow("map", 0, 0);

  char key = 0;
  int counter = 0;
  int history_sum[6] = {0};
  while (key != 'q')
  {
    mapCanvas = map.clone();
    vector<Point2d> poseArray;
    for (int i=0, n=capArray.size(); i<n; ++i)
    {
      Mat frame;
      VideoCapture& cap = capArray[i];
      if (!cap.read(frame)) return 0;

      // detecting human
      Mat grayFrame;
      vector<Rect> found;
      cvtColor(frame, grayFrame, CV_BGR2GRAY);
#ifdef OCL
      ocl::oclMat calFrame(grayFrame);
#else
      Mat calFrame = grayFrame;
#endif
      hog.detectMultiScale(calFrame, found, 0);
      for (int j=0, n=found.size(); j<n; ++j)
      {
        rectangle(frame, found[j].tl(), found[j].br(), GRAY, 3);
        int x = (found[j].tl().x + found[j].br().x) / 2;
        int y = (found[j].tl().y +  2 * found[j].br().y) / 3;
        Point2d pose = homoTransform(Point2d(x, y), homoArray[i]);
        poseArray.push_back(pose);
      }
      imshow(nameArray[i], frame);
    }
    vector<Point2d> centerArray;
    vector<int> countArray;
    for (int i=0, n=poseArray.size(); i<n; ++i)
    {
      bool matched = false;
      for (int j=0, k=centerArray.size(); j<k; ++j)
      {
        int dx = poseArray[i].x - poseArray[j].x;
        int dy = poseArray[i].y - poseArray[j].y;
        int d = dx * dx + dy * dy;
        if (d < 10000)
        {
          matched = true;
          centerArray[j].x =
            (countArray[j] * centerArray[j].x + poseArray[i].x)
            / (countArray[j] + 1);
          countArray[j]++;
        }
      }
      if (!matched)
      {
        centerArray.push_back(poseArray[i]);
        countArray.push_back(1);
      }
    }

    // Count number in each block for demo
    int count[6] = {0};
    for (int i=0, n=centerArray.size(); i<n; ++i)
    {
      int y = centerArray[i].y;
      int x = centerArray[i].x;
      int idx = 2 * ((y - 258) / 121) + (x / 326);
      if (idx > 5) continue;
      count[idx]++;
    }

    if (counter % SMOOTH_LENGTH == 0)
    {
      for (int i=0, n=centerArray.size(); i<n; ++i)
      {
        cross(mapCanvas, centerArray[i].y, centerArray[i].x, WHITE);
      }
      imshow("map", mapCanvas);
    }
    if (counter % MESSAGE_PERIOD == 0)
    {
      printf("[");
      for (int i=0; i<6; ++i)
      {
        if (i != 5) printf("%d, ",
            (int) round(history_sum[i]/(double) MESSAGE_PERIOD));
        else printf("%d",
            (int) round(history_sum[i]/(double) MESSAGE_PERIOD));
      }
      printf("]\n");
      fflush(stdout);
      for (int i=0; i<6; ++i) history_sum[i] = 0;
    }
    else
    {
      for (int i=0; i<6; ++i) history_sum[i] += count[i];
    }
    key = waitKey(1);
    counter++;
  }
}
