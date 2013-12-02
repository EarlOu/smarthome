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
  VideoCapture cap(argv[1]);
  namedWindow("video");
  namedWindow("img");

  Mat frame, img;

  int idx = 0;
  while (cap.read(frame))
  {
    imshow("video", frame);
    char w = waitKey(30);
    if (w == 'q') return 0;
    if (w != -1)
    {
      img = frame.clone();
      unsigned long time = getUnixTime();
      char buf[128];
      //sprintf(buf, "%ld.jpg", time);
      sprintf(buf, "%d.jpg", idx);
      imwrite(buf, img);
      imshow("img", img);
    }
  }
}
