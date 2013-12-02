CFLAGS := `pkg-config --cflags opencv`
CLIBS := `pkg-config --libs opencv`

.PHONY: all

all: bin/camera bin/calibration bin/homography

bin/camera: camera.cpp
	$(CXX) $< $(CFLAGS) $(CLIBS) -o $@

bin/calibration: calibration.cpp
	$(CXX) $< $(CFLAGS) $(CLIBS) -o $@

bin/homography: homography.cpp
	$(CXX) $< $(CFLAGS) $(CLIBS) -o $@
