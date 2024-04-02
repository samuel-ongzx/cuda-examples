# IDIR=./
CXX = nvcc

CXXFLAGS += $(shell pkg-config --cflags --libs opencv4)
LDFLAGS += $(shell pkg-config --libs --static opencv4)

all: clean build

build: 
	$(CXX) convert_rgb_to_grayscale.cu --std c++17 `pkg-config opencv4 --cflags --libs` -o convert_rgb_to_grayscale.exe -Wno-deprecated-gpu-targets $(CXXFLAGS) -I/usr/local/cuda/include -lcuda

run:
	./convert_rgb_to_grayscale.exe $(ARGS)

clean:
	rm -f convert_rgb_to_grayscale.exe