# IDIR=./
CXX = nvcc

CXXFLAGS += $() # $(shell pkg-config --cflags --libs opencv4)
LDFLAGS += $() # $(shell pkg-config --libs --static opencv)

all: clean build

build: cublas_example.cu
	$(CXX) cublas_example.cu --std c++17 -o cublas_example.exe -Wno-deprecated-gpu-targets $(CXXFLAGS) -I/usr/local/cuda/include -lcuda -lcublas

run:
	./cublas_example.exe

clean:
	rm -f cublas_example.exe 