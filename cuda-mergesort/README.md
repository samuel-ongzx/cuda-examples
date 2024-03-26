## Introduction
CUDA implementation of mergesort. Heavily referenced from Coursera GPU Specialization course with some modifications.

## Building & Running
Execute `make` and `./merge_sort.exe`.

## Arguments
`-x N` for setting the the x-dim of threads per block to `N`.
`-y N` for setting the the y-dim of threads per block to `N`.
`-z N` for setting the the z-dim of threads per block to `N`.
`-X N` for setting the the x-dim of blocks per grid to `N`.
`-Y N` for setting the the y-dim of blocks per grid to `N`.
`-Z N` for setting the the z-dim of blocks per grid to `N`.
`-n N` for setting the the number of elements to sort to `N`.
`-l` for using the `max` and `min` of a `long` else it defaults to `{-1000, 1000}`.