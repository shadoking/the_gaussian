#pragma once
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <Eigen/Dense>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define BLOCK_X 16
#define BLOCK_Y 16

//int max(int a, int b) {
//	return (a < b) ? b : a;
//}
//
//int min(int a, int b) {
//	return (a < b) ? a : b;
//}

//#define CHECK_CUDA(A, debug) \
//A; if(debug) { \
//auto ret = cudaDeviceSynchronize(); \
//if (ret != cudaSuccess) { \
//std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
//throw std::runtime_error(cudaGetErrorString(ret)); \
//} \
//}
#define CHECK_CUDA(call)                                                        \
    {                                                                           \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1);                                                            \
        }                                                                       \
    }

__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};