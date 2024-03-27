## Introduction
CUDA (re)implementation of GEMM (General Matrix Multiplication).
GEMM has been implemented in cuBLAS, but is not open sourced. 
It would be useful to implement GEMM at the lowest level in CUDA, since matrix multiplication is the building blocks of machine learning.
GEMM equation is `C = \alpha * AB + \beta *C`.

## Building & Running
Execute `make` and `./cuda_gemm.exe`.

## Arguments
`-x N` for setting the the x-dim of threads per block to `N`.
`-y N` for setting the the y-dim of threads per block to `N`.
`-z N` for setting the the z-dim of threads per block to `N`.
`-X N` for setting the the x-dim of blocks per grid to `N`.
`-Y N` for setting the the y-dim of blocks per grid to `N`.
`-Z N` for setting the the z-dim of blocks per grid to `N`.
`-n N` for setting the the number of elements to sort to `N`.
`-a N` for setting value of `alpha` that scales matrix (AB).
`-b N` for setting value of `beta` that scales matrix (C).