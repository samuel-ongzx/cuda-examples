#include <stdlib.h>
#include <stdio.h>
#include <cublas_v2.h> // use v2 of cublas
#include <random>

#define HA 2 // A-rows
#define WA 9 // dim
#define WB 2 // B-rows
#define HB WA 
#define WC WB   
#define HC HA  
#define index(i,j,ld) (((j)*(ld))+(i))

void printMat(float*P,int uWP,int uHP){
  int i,j;
  for(i=0;i<uHP;i++){
      printf("\n");
      for(j=0;j<uWP;j++)
          printf("%f ",P[index(i,j,uHP)]);
  }
}

__host__ float* initializeHostMemory(int height, int width, bool random, float nonRandomValue) {
  float* hostMatrix = (float*) malloc(height*width*sizeof(float));
  // fill float values into matrices
  std::default_random_engine engine(std::random_device{}());
  auto d = std::uniform_real_distribution<float>(-1000.0, 1000.0); // default for readablity
  if (random) {
    for (size_t i=0; i<width*height; i++) {
      hostMatrix[i] = d(engine);
    }
  } else {
    for (size_t i=0; i<width*height; i++) {
      hostMatrix[i] = nonRandomValue;
    }
  }
  return hostMatrix;
}

__host__ float *initializeDeviceMemoryFromHostMemory(int height, int width, float *hostMatrix) {
  // allocate
  float* deviceMatrix;
  size_t size = height*width*sizeof(float);
  cudaMalloc(&deviceMatrix, size);

  // copy values from host to device
  cudaMemcpy(deviceMatrix, hostMatrix, size, cudaMemcpyHostToDevice);
  return deviceMatrix;
}

__host__ float *retrieveDeviceMemory(int height, int width, float *deviceMatrix, float *hostMemory) {
  // copy out values to host memory
  cudaMemcpy(hostMemory, deviceMatrix, height*width*sizeof(float), cudaMemcpyDeviceToHost);
  return hostMemory;
}

__host__ void printMatrices(float *A, float *B, float *C){
  printf("\nMatrix A:\n");
  printMat(A,WA,HA);
  printf("\n");
  printf("\nMatrix B:\n");
  printMat(B,WB,HB);
  printf("\n");
  printf("\nMatrix C:\n");
  printMat(C,WC,HC);
  printf("\n");
}

__host__ int freeMatrices(float *A, float *B, float *C, float *AA, float *BB, float *CC){
  free( A );  free( B );  free ( C );
  cudaError_t status = cudaFree(AA);
  if (status != cudaSuccess) {
    fprintf (stderr, "!!!! memory free error (A)\n");
    return EXIT_FAILURE;
  }
  status = cudaFree(BB);
  if (status != cudaSuccess) {
    fprintf (stderr, "!!!! memory free error (B)\n");
    return EXIT_FAILURE;
  }
  status = cudaFree(CC);
  if (status != cudaSuccess) {
    fprintf (stderr, "!!!! memory free error (C)\n");
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

int  main (int argc, char** argv) {  
  // initialize matrix with values in host memory
  float* A = initializeHostMemory(HA, WA, false, 1.0);
  float* B = initializeHostMemory(HB, WB, false, 2.0);
  float* C = initializeHostMemory(HC, WC, true, 0.0);

  // initialize & copy matrix to device memory
  float* AA = initializeDeviceMemoryFromHostMemory(HA, WA, A);
  float* BB = initializeDeviceMemoryFromHostMemory(HB, WB, B);
  float* CC = initializeDeviceMemoryFromHostMemory(HC, WC, C); // AA*BB = CC, height = HA, width = WB.

  // create BLAS handle
  cublasHandle_t handle;
  cublasCreate_v2(&handle);

  // cublas matrix multiplication call
  float alpha = 1.f;
  float beta = 0.f;
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, // no transpose
          HA, HB, WA,                           // rows of (A = C), cols of (B = C), cols of A = rows of B, C = AB (HA = m, HB = n, WA = k)
          &alpha,                               // host or device memory, C = alpha*(AB) + beta*C
          AA, HA,                               // A matrix & leading dimension of mat A (height of A)
          BB, HB,                               // B matrix & leading dimension of mat B
          &beta,                                // 0
          CC, HC);                              // C matrix & leading dimension of mat C

  // copy out matrix to host memory, print & then free
  C = retrieveDeviceMemory(HC, WC, CC, C);
  printMatrices(A, B, C);
  freeMatrices(A, B, C, AA, BB, CC);
  
  // cleanup
  cublasStatus_t status = cublasDestroy_v2(handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! shutdown error (A)\n");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
