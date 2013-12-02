#include <opencv2/opencv.hpp>
#include <sys/time.h>
#include <stdio.h>

using namespace cv;

unsigned long getUnixTime()
{
    struct timeval tv;
    unsigned long time;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

int main(int argc, char *argv[])
{
  if (argc != 2 && argc != 3)
  {
    printf("usage: %s <video> <calib>\n", argv[0]);
    return -1;
  }
  VideoCapture cap(argv[1]);
  bool isCalib = argc == 3;

  namedWindow("video");
  namedWindow("img");

  Mat camMat;
  Mat distCoef;

  if (isCalib)
  {
    namedWindow("undist");

    FileStorage fs(argv[2], FileStorage::READ);
    fs["camera_matrix"] >> camMat;
    fs["distortion_coefficients"] >> distCoef;
  }

  Mat frame, img;

  int idx = 0;
  while (cap.read(frame))
  {
    imshow("video", frame);
    if (isCalib)
    {
      Mat undistorted;
      undistort(frame, undistorted, camMat, distCoef);
      imshow("undist", undistorted);
    }
    char w = waitKey(1);
    if (w == 'q') return 0;
    if (w != -1)
    {
      img = frame.clone();
      unsigned long time = getUnixTime();
      char buf[128];
      //sprintf(buf, "%ld.jpg", time);
      sprintf(buf, "%d.jpg", idx++);
      imwrite(buf, img);
      imshow("img", img);
    }
  }
}
