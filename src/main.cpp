#include "util.h"
#include "draw.h"

#include <opencv2/opencv.hpp>
#include <opencv2/ocl/ocl.hpp>
#include <stdio.h>
#include <vector>

using namespace cv;
using namespace std;

//#define OCL

int main(int argc, char *argv[])
{
  if (argc != 2)
  {
    printf("usage: %s <video_list>\n", argv[0]);
    return -1;
  }
  FILE* ifile = fopen(argv[1], "r");
  vector<Mat> homoArray;
  vector<string> nameArray;
  readInputList(ifile, nameArray, homoArray);

  vector<VideoCapture> capArray;
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
  }

#ifdef OCL
  vector<ocl::Info> oclInfo;
  ocl::getDevice(oclInfo);
  ocl::setDevice(oclInfo[0]);
#endif

#ifdef OCL
  ocl::HOGDescriptor hog;
  hog.setSVMDetector(ocl::HOGDescriptor::getDefaultPeopleDetector());
#else
  HOGDescriptor hog;
  hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
#endif

  char key = 0;
  while (key != 'q')
  {
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
      for (int i=0, n=found.size(); i<n; ++i)
      {
        rectangle(frame, found[i].tl(), found[i].br(), RED, 3);
        int x = (found[i].tl().x + found[i].br().x) / 2;
        int y = (found[i].tl().y +  2 * found[i].br().y) / 3;
        Point2d pose = homoTransform(Point2d(x, y), homoArray[i]);
        //cross(mapCanvas, rx, ry, RED);
      }
      imshow(nameArray[i], frame);
    }
    key = waitKey(1);
  }
}
