# IDIR=./
CXX = nvcc

# CXXFLAGS += $(shell pkg-config --cflags --libs opencv4)
# LDFLAGS += $(shell pkg-config --libs --static opencv)

all: clean build_hello build_vector_add build_vector_add_thread build_vector_add_grid

build_hello: hello.cu
	$(CXX) hello.cu --std c++17 -o hello.exe -Wno-deprecated-gpu-targets -I/usr/local/cuda/include -lcuda

build_vector_add: vector_add.cu # single thread
	$(CXX) vector_add.cu --std c++17 -o vector_add.exe -Wno-deprecated-gpu-targets -I/usr/local/cuda/include -lcuda

build_vector_add_thread: vector_add_thread.cu # multiple thread
	$(CXX) vector_add_thread.cu --std c++17 -o vector_add_thread.exe -Wno-deprecated-gpu-targets -I/usr/local/cuda/include -lcuda

build_vector_add_grid: vector_add_grid.cu # grid
	$(CXX) vector_add_grid.cu --std c++17 -o vector_add_grid.exe -Wno-deprecated-gpu-targets -I/usr/local/cuda/include -lcuda

clean:
	rm -f hello.exe vector_add.exe vector_add_thread.exe vector_add_grid.exe