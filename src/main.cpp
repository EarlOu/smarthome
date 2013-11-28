#include <opencv2/opencv.hpp>
#include <opencv2/ocl/ocl.hpp>
#include <sys/time.h>

using namespace cv;

//#define OCL

const Scalar RED(0, 0, 255);
const Scalar GREEN(255, 255, 0);

void cross(Mat& img, int x, int y, Scalar color)
{
  line(img, Point(x-5, y-5), Point(x+5, y+5), color);
  line(img, Point(x+5, y-5), Point(x-5, y+5), color);
}

long getTime()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

int main(int argc, char *argv[])
{

#ifdef OCL
  vector<ocl::Info> oclInfo;
  ocl::getDevice(oclInfo);
  ocl::setDevice(oclInfo[0]);
#endif

  VideoCapture cap(argv[1]);

#ifdef OCL
  ocl::HOGDescriptor hog;
  hog.setSVMDetector(ocl::HOGDescriptor::getDefaultPeopleDetector());
#else
  HOGDescriptor hog;
  hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
#endif

  Mat frame;
  Mat map(300, 800, CV_8UC3);

  Mat H(3, 3, CV_64FC1);
  H.at<double>(0, 0) = 3.798395;
  H.at<double>(0, 1) = 2.532263;
  H.at<double>(0, 2) = -355.783021;
  H.at<double>(1, 0) = 0.000000;
  H.at<double>(1, 1) = 7.283661;
  H.at<double>(1, 2) = -553.558245;
  H.at<double>(2, 0) = 0.000810;
  H.at<double>(2, 1) = 0.008059;
  H.at<double>(2, 2) = 1.000000;

  namedWindow("img");
  namedWindow("map");
  int index = 0;
  long prevTime = 0;
  while (cap.read(frame))
  {
    long time = getTime();
    if (prevTime != 0)
    {
      long diffTime = time - prevTime;
      printf("%f\n", 1000.0f / diffTime);
    }
    prevTime = time;
    Mat grayFrame;
    cvtColor(frame, grayFrame, CV_BGR2GRAY);
#ifdef OCL
    ocl::oclMat calFrame(grayFrame);
#else
    Mat calFrame = grayFrame;
#endif
    Mat mapCanvas = map.clone();
    vector<Rect> found;
    hog.detectMultiScale(calFrame, found, 0);
    for (int i=0, n=found.size(); i<n; ++i)
    {
      rectangle(frame, found[i].tl(), found[i].br(), RED, 3);
      int x = (found[i].tl().x + found[i].br().x) / 2;
      int y = (found[i].tl().y +  2 * found[i].br().y) / 3;
      Mat s(3, 1, CV_64FC1);
      s.at<double>(0) = x;
      s.at<double>(1) = y;
      s.at<double>(2) = 1;
      Mat d = H * s;
      double rx = d.at<double>(0);
      double ry = d.at<double>(1);
      double r = d.at<double>(2);
      rx /= r;
      ry /= r;
      cross(mapCanvas, rx, ry, RED);
    }

    imshow("img", frame);
    imshow("map", mapCanvas);
    waitKey(1);
  }
}
