#include "convert_rgb_to_grayscale.hpp"

/*
 * CUDA Kernel Device code
 */
__global__ void convert(uchar *d_r, uchar *d_g, uchar *d_b, uchar *d_gray)
{
  //To convert from RGB to grayscale, use the average of the values in d_r, d_g, d_b and place in d_gray
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= d_imgWidth || y >= d_imgHeight)
    return;

  int pxIdx = y * d_imgWidth + x;
  d_gray[pxIdx] = (d_r[pxIdx] + d_g[pxIdx] + d_b[pxIdx]) / 3;
}

__host__ std::tuple<uchar *, uchar *, uchar *, uchar *> allocateDeviceMemory(int imgHeight, int imgWidth)
{
  std::cout << "Allocating GPU device memory\n";
  int num_image_pixels = imgHeight * imgWidth;
  size_t size = num_image_pixels * sizeof(uchar);

  // Allocate the device input vector d_r
  uchar *d_r = NULL;
  cudaError_t err = cudaMalloc(&d_r, size);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to allocate device vector d_r (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Allocate the device input vector d_g
  uchar *d_g = NULL;
  err = cudaMalloc(&d_g, size);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to allocate device vector d_g (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Allocate the device input vector d_b
  uchar *d_b = NULL;
  err = cudaMalloc(&d_b, size);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to allocate device vector d_b (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Allocate the device input vector d_gray
  uchar *d_gray = NULL;
  err = cudaMalloc(&d_gray, size);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to allocate device vector d_gray (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  //Allocate device constant symbols for rows and columns
  cudaMemcpyToSymbol(d_imgHeight, &imgHeight, sizeof(int), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_imgWidth, &imgWidth, sizeof(int), 0, cudaMemcpyHostToDevice);

  return {d_r, d_g, d_b, d_gray};
}

/*
 * Memory Copy
 */
__host__ void copyFromHostToDevice(uchar *h_r, uchar *d_r, uchar *h_g, uchar *d_g, uchar *h_b, uchar *d_b, int rows, int columns)
{
  std::cout << "Copying from Host to Device\n";
  int num_image_pixels = rows * columns;
  size_t size = num_image_pixels * sizeof(uchar);

  cudaError_t err;
  err = cudaMemcpy(d_r, h_r, size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to copy vector r from host to device (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaMemcpy(d_g, h_g, size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to copy vector g from host to device (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to copy vector b from host to device (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

__host__ void copyFromDeviceToHost(uchar *d_gray, uchar *gray, int imgWidth, int imgHeight)
{
  std::cout << "Copying from Device to Host\n";
  // Copy the device result int array in device memory to the host result int array in host memory.
  size_t size = imgHeight * imgWidth * sizeof(uchar);

  cudaError_t err = cudaMemcpy(gray, d_gray, size, cudaMemcpyDeviceToHost);

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to copy array d_gray from device to host (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

/*
 * Entrypoint to Device call
 */
__host__ void executeKernel(uchar *d_r, uchar *d_g, uchar *d_b, uchar *d_gray, int imgWidth, int imgHeight, int threadsPerBlock)
{
  //Launch the convert CUDA Kernel
  // int blockZSize = 4; // Could consider making the block/grid and memory layout 3d mapped but for now just breaking up computation
  dim3 blockSize(threadsPerBlock, threadsPerBlock, 1);
  int gridCols = (imgWidth + blockSize.x - 1) / blockSize.x;
  int gridRows = (imgHeight + blockSize.y - 1) / blockSize.y;
  dim3 gridSize(gridRows, gridCols, 1);

  printf("Executing Kernel: (%d, %d, %d), (%d, %d, %d)\n", gridRows, gridCols, 1, threadsPerBlock, threadsPerBlock, 1); 
  convert<<<gridSize, blockSize>>>(d_r, d_g, d_b, d_gray);
  cudaError_t err = cudaGetLastError();

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to launch convert kernel (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

// Free host memory
__host__ void deallocateHostMemory(uchar *h_r, uchar *h_g, uchar *h_b, uchar *h_gray) {
  std::cout << "Deallocating host memory\n";
  free(h_r);
  free(h_g);
  free(h_b);
  free(h_gray);
}

// Free device global memory
__host__ void deallocateDeviceMemory(uchar *d_r, uchar *d_g, uchar *d_b, uchar *d_gray)
{
  std::cout << "Deallocating GPU device memory\n";
  cudaError_t err = cudaFree(d_r);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to free device vector d_r (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaFree(d_g);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to free device vector d_g (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaFree(d_b);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to free device vector d_b (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaFree(d_gray);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to free device int variable d_gray (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

// Reset the device and exit
__host__ void cleanUpDevice()
{
  std::cout << "Cleaning CUDA device\n";
  // cudaDeviceReset causes the driver to clean up all state. While
  // not mandatory in normal operation, it is good practice.  It is also
  // needed to ensure correct operation when the application is being
  // profiled. Calling cudaDeviceReset causes all profile data to be
  // flushed before the application exits
  cudaError_t err = cudaDeviceReset();

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

__host__ std::tuple<std::string, std::string, int> parseCommandLineArguments(int argc, char *argv[])
{
  std::cout << "Parsing CLI arguments\n";
  int threadsPerBlock = 32;
  std::string inputImage = "./dog.jpg";
  std::string outputImage = "./grey-dog.jpg";

  for (int i = 1; i < argc; i++)
  {
    std::string option(argv[i]);
    i++;
    std::string value(argv[i]);
    if (option.compare("-i") == 0)
    {
      inputImage = value;
    }
    else if (option.compare("-o") == 0)
    {
      outputImage = value;
    }
    else if (option.compare("-t") == 0)
    {
      threadsPerBlock = atoi(value.c_str());
    }
  }
  std::cout << "inputImage: " << inputImage << " outputImage: " << outputImage << " threadsPerBlock: " << threadsPerBlock << "\n";
  return {inputImage, outputImage, threadsPerBlock};
}

__host__ std::tuple<int, int, uchar *, uchar *, uchar *> readImageFromFile(std::string inputFile)
{
  std::cout << "Reading Image From File\n";
  cv::Mat img = cv::imread(inputFile, cv::IMREAD_COLOR);
  
  const int imgHeight = img.rows;
  const int imgWidth = img.cols;
  const int channels = img.channels();
  std::cout << "Height: " << imgHeight << " Width: " << imgWidth << "\n";
  
  // split into rgb channels
  std::vector<cv::Mat> rgbChannels(3);
  cv::split(img, rgbChannels);

  // remeber to allocate memory for the channel data!!
  uchar *h_b = (uchar *)malloc(imgHeight * imgWidth);
  uchar *h_g = (uchar *)malloc(imgHeight * imgWidth);
  uchar *h_r = (uchar *)malloc(imgHeight * imgWidth);
  
  memcpy(h_b, rgbChannels[0].data, imgHeight * imgWidth);
  memcpy(h_g, rgbChannels[1].data, imgHeight * imgWidth);
  memcpy(h_r, rgbChannels[2].data, imgHeight * imgWidth);

  return {imgHeight, imgWidth, h_r, h_g, h_b};
}

int main(int argc, char *argv[])
{
  std::tuple<std::string, std::string, int> parsedCommandLineArgsTuple = parseCommandLineArguments(argc, argv);
  std::string inputImage = std::get<0>(parsedCommandLineArgsTuple);
  std::string outputImage = std::get<1>(parsedCommandLineArgsTuple);
  int threadsPerBlock = std::get<2>(parsedCommandLineArgsTuple);

  auto [imgHeight, imgWidth, h_r, h_g, h_b] = readImageFromFile(inputImage);
  uchar *h_gray = (uchar *)malloc(sizeof(uchar) * imgHeight * imgWidth);

  auto [d_r, d_g, d_b, d_gray] = allocateDeviceMemory(imgHeight, imgWidth);
  copyFromHostToDevice(h_r, d_r, h_g, d_g, h_b, d_b, imgHeight, imgWidth);

  executeKernel(d_r, d_g, d_b, d_gray, imgHeight, imgWidth, threadsPerBlock);

  copyFromDeviceToHost(d_gray, h_gray, imgHeight, imgWidth);
  deallocateDeviceMemory(d_r, d_g, d_b, d_gray);
  cleanUpDevice();

  cv::Mat grayImageMat(imgHeight, imgWidth, CV_8UC1, h_gray);
  cv::imwrite(outputImage, grayImageMat);
  deallocateHostMemory(h_r, h_g, h_b, h_gray);

  return 0;
}