#include "forward.h"

__global__ void PreRenderKernel(
    const float* xyz,
    //const float* rotation,
    //const float* scaling,
    //const float* opacity,
    //const float* features,

    const float near,
    const int N,
    //const int width,
    //const int height,
    const float* viewMatrix,
    const float* viewProjMatrix
    
    ) {
    // 1.计算 pid    
    int pid = blockDim.x * blockIdx.x + threadIdx.x;
    if (pid >= N) {
        return;
    }
    // 2.近平面裁剪
    Eigen::Vector4f oriPoint(xyz[pid * 3], xyz[pid * 3 + 1], xyz[pid * 3 + 2], 1.f);
    Eigen::Map<const Eigen::Matrix<float, 4, 4, Eigen::RowMajor>> viewMatrixE(viewMatrix);
    Eigen::Vector4f viewPoint = viewMatrixE * oriPoint;
    if (viewPoint[2] <= near) {
        return;
    }

    // 3.中心投影
    Eigen::Map<const Eigen::Matrix<float, 4, 4, Eigen::RowMajor>> viewProjMatrixE(viewProjMatrix);
    Eigen::Vector4f projPoint = viewProjMatrixE * oriPoint;


   /* if (pid == 1) {
        printf("oriPoint: %f,%f,%f,%f\n", oriPoint[0], oriPoint[1], oriPoint[2], oriPoint[3]);
        printf("viewPoint: %f,%f,%f,%f\n", viewPoint[0], viewPoint[1], viewPoint[2], viewPoint[3]);
    }*/
    
    // 4.矩阵计算
}

void  PreRender(
    const float* xyz, 
    
    const float near, 
    const int N, 
    const float* viewMatrix,
    const float* viewProjMatrix
) {
    float *d_xyz, *d_viewMatrix, *d_viewProjMatrix;
    cudaMalloc(&d_xyz, 3 * N * sizeof(float));
    cudaMemcpy(d_xyz, xyz, 3 * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_viewMatrix, 16 * sizeof(float));
    cudaMemcpy(d_viewMatrix, viewMatrix, 16 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_viewProjMatrix, 16 * sizeof(float));
    cudaMemcpy(d_viewProjMatrix, viewProjMatrix, 16 * sizeof(float), cudaMemcpyHostToDevice);

    PreRenderKernel <<<(N + 255) / 256, 256 >>> (d_xyz, near, N, d_viewMatrix, d_viewProjMatrix);

    cudaFree(d_xyz);
    cudaFree(d_viewMatrix);
    cudaFree(d_viewProjMatrix);
}