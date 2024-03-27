# IDIR=./
CXX = nvcc

CXXFLAGS += $() # $(shell pkg-config --cflags --libs opencv4)
LDFLAGS += $() # $(shell pkg-config --libs --static opencv)

all: clean build

build: cuda_gemm.cu
	$(CXX) cuda_gemm.cu --std c++17 -o cuda_gemm.exe -Wno-deprecated-gpu-targets $(CXXFLAGS) -I/usr/local/cuda/include -lcuda

run:
	./cuda_gemm.exe

clean:
	rm -f cuda_gemm.exe 