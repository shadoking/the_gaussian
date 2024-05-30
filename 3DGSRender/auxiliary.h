#pragma once
#include <stdio.h>
#include <string.h>
#include <Eigen/Dense>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define BLOCK_X 16
#define BLOCK_Y 16

int max(int a, int b) {
	return (a < b) ? b : a;
}

int min(int a, int b) {
	return (a < b) ? a : b;
}

#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}