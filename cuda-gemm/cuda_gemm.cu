#include <random>
#include <tuple>
#include <string>

#include <stdlib.h>
#include <stdio.h>

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

__host__ std::tuple<dim3, dim3, float, float> parseCommandLineArguments(int argc, char** argv) 
{
    dim3 threadsPerBlock(32,32,1);
    dim3 blocksPerGrid(32,32,1);
    float alpha = 1.0f;
    float beta = 0.0f;

    for (int i = 1; i < argc; i++) {
        if (argv[i][0] == '-' && argv[i][1] && !argv[i][2]) {
            char arg = argv[i][1];
            unsigned int* toSet = 0;
            switch(arg) {
                case 'x':
                    toSet = &threadsPerBlock.x;
                    break;
                case 'y':
                    toSet = &threadsPerBlock.y;
                    break;
                case 'z':
                    toSet = &threadsPerBlock.z;
                    break;
                case 'X':
                    toSet = &blocksPerGrid.x;
                    break;
                case 'Y':
                    toSet = &blocksPerGrid.y;
                    break;
                case 'Z':
                    toSet = &blocksPerGrid.z;
                    break;
                case 'a':
                    i++;
                    alpha = std::stof(argv[i]);
                    break;
                case 'b':
                    i++;
                    beta = std::stof(argv[i]);
                    break;
            }
            if (toSet) {
                i++;
                *toSet = (unsigned int) strtol(argv[i], 0, 10);
            }
        }
    }
    return {threadsPerBlock, blocksPerGrid, alpha, beta};
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

__host__ int freeMatrices(float *A, float *B, float *C, float *AA, float *BB, float *CC) {
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

// overload kernel function
__global__ void customSgemm(int m, int n, int k, 
                            const float* alpha, 
                            const float* A, int lda,
                            const float* B, int ldb,
                            const float* beta,
                            float* C, int ldc)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m && col < n) {
    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
        sum += A[row * k + i] * B[i * n + col];
    }
    // C = alpha(AB) + beta(C)
    if ((int)*beta == 0) {
      C[row * n + col] = (*alpha) * sum; // C does not need to be valid
    } else {
      C[row * n + col] *= (*beta);
      C[row * n + col] += (*alpha) * sum; // C needs to be valid
    }
  }
}

// Custom single precision matrix multiplication, row-major only.
// C = alpha*(AB) + beta*(C)
void customSgemm(int m, int n, int k, 
                 const float* alpha, 
                 const float* A, int lda,
                 const float* B, int ldb,
                 const float* beta,
                 float* C, int ldc, 
                 dim3 blocksPerGrid, 
                 dim3 threadsPerBlock)
{
  float* d_alpha;
  float* d_beta;
  // copy alpha & beta to device memory
  cudaMalloc(&d_alpha, sizeof(float));
  cudaMalloc(&d_beta,  sizeof(float));
  cudaMemcpy(d_alpha, alpha, sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_beta,  beta,  sizeof(float), cudaMemcpyHostToDevice);

  // actual kernel call
  customSgemm<<<blocksPerGrid, threadsPerBlock>>>(m, n, k, 
                                                  d_alpha, 
                                                  A, lda,
                                                  B, ldb, 
                                                  d_beta, 
                                                  C, ldc);

  // free alpha & beta
  cudaFree(d_alpha);
  cudaFree(d_beta);
}

int  main (int argc, char** argv) {
  auto [threadsPerBlock, blocksPerGrid, alpha, beta] = parseCommandLineArguments(argc, argv);

  // initialize matrix with values in host memory
  float* A = initializeHostMemory(HA, WA, false, 1.0);
  float* B = initializeHostMemory(HB, WB, false, 2.0);
  float* C = initializeHostMemory(HC, WC, false, 3.0);

  // initialize & copy matrix to device memory
  float* AA = initializeDeviceMemoryFromHostMemory(HA, WA, A);
  float* BB = initializeDeviceMemoryFromHostMemory(HB, WB, B);
  float* CC = initializeDeviceMemoryFromHostMemory(HC, WC, C); // AA*BB = CC, height = HA, width = WB.

  // cublas matrix multiplication call
  customSgemm(HA, WB, WA, // rows of (A = C), cols of (B = C), cols of A = rows of B, C = AB (HA = m, HB = n, WA = k)
              &alpha,      // host or device memory, C = alpha*(AB) + beta*C
              AA, HA,      // A matrix & leading dimension of mat A (height of A)
              BB, HB,      // B matrix & leading dimension of mat B
              &beta,       // 0
              CC, HC,      // C matrix & leading dimension of mat C
              blocksPerGrid,
              threadsPerBlock);

  // copy out matrix to host memory, print & then free
  C = retrieveDeviceMemory(HC, WC, CC, C);
  printMatrices(A, B, C);
  freeMatrices(A, B, C, AA, BB, CC);

  return EXIT_SUCCESS;
}
