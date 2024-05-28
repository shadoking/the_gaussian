#include "forward.h"

__device__ void strConcat(char* dest, const char* src) {
    while (*dest) {
        dest++;
    }
    while ((*dest++ = *src++)) { }
}

__device__ void printMatrix3(Eigen::Matrix3f matrix, char* name) {
    strConcat(name, ":\n%f, %f, %f\n%f, %f, %f\n%f, %f, %f\n");
    printf(name,
        matrix(0, 0), matrix(0, 1), matrix(0, 2),
        matrix(1, 0), matrix(1, 1), matrix(1, 2),
        matrix(2, 0), matrix(2, 1), matrix(2, 2));
}

__device__ void printMatrix2(Eigen::Matrix2f matrix, char* name) {
    strConcat(name, ":\n%f, %f\n%f, %f\n");
    printf(name,
        matrix(0, 0), matrix(0, 1),
        matrix(1, 0), matrix(1, 1));
}

__device__ Eigen::Matrix2f ComputeCov2D(
    Eigen::Vector4f viewPoint, 
    Eigen::Matrix4f viewMatrix,
    Eigen::Matrix3f cov3D,
    float focalX, float focalY,
    float tanFovX, float tanFovY) {

    float limx = 1.3f * tanFovX;
    float limy = 1.3f * tanFovY;
    float xz = viewPoint[0] / viewPoint[2];
    float yz = viewPoint[1] / viewPoint[2];
    float x = fminf(limx, fmaxf(-limx, xz)) * viewPoint[2];
    float y = fminf(limy, fmaxf(-limy, yz)) * viewPoint[2];
    float z = viewPoint[2];

    Eigen::Matrix<float, 2, 3> J;
    J << focalX / z, 0.f, -(focalX * x) / (z * z),
        0.f, focalY / z, -(focalY * y) / (z * z);
    /*printf("J:\n%f, %f, %f\n%f, %f, %f\n",
        J(0, 0), J(0, 1), J(0, 2),
        J(1, 0), J(1, 1), J(1, 2));*/
    
    Eigen::Matrix3f W = viewMatrix.block<3, 3>(0, 0);
    /*printMatrix3(W, "W");
    printMatrix3(cov3D, "cov3D");*/

    Eigen::Matrix2f cov2D = J * W * cov3D * W.transpose() * J.transpose();

    cov2D(0, 0) += 0.3f;
    cov2D(1, 1) += 0.3f;

    return cov2D;
}

__device__ Eigen::Matrix3f ComputeCov3D(Eigen::Vector4f quat, Eigen::Vector3f scal) {
    // 归一化
    Eigen::Vector4f quatNorm = quat.normalized();
    float w = quatNorm[0];
    float x = quatNorm[1];
    float y = quatNorm[2];
    float z = quatNorm[3];

    // 旋转矩阵
    Eigen::Matrix3f R;
    R << 1.f - 2.f * y * y - 2.f * z * z, 2.f * x * y - 2.f * w * z, 2.f * x * z + 2.f * w * y,
        2.f * x * y + 2.f * w * z, 1.f - 2.f * x * x - 2.f * z * z, 2.f * y * z - 2.f * w * x,
        2.f * x * z - 2.f * w * y, 2.f * y * z + 2.f * w * x, 1.f - 2.f * x * x - 2.f * y * y;
    
    Eigen::Matrix3f cov3D;
    cov3D << scal[0], 0, 0, 0, scal[1], 0, 0, 0, scal[2];
    // RS
    cov3D = R * cov3D;
    // RSSR
    cov3D = cov3D * cov3D.transpose();

    return cov3D;
}



__global__ void PreRenderKernel(
    const float* xyz,
    const float* rotation,
    const float* scaling,
    //const float* opacity,
    //const float* features,

    const float near,
    const int N,
    //const int width,
    //const int height,
    const float* viewMatrix,
    const float* viewProjMatrix,
    float focalX, float focalY,
    float tanFovX, float tanFovY
    
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
    Eigen::Vector4f projPointH = viewProjMatrixE * oriPoint;
    Eigen::Vector3f projPoint = { projPointH[0], projPointH[1], projPointH[2] };


    // 4.矩阵计算
    Eigen::Vector4f quat(rotation[pid * 4], rotation[pid * 4 + 1], rotation[pid * 4 + 2], rotation[pid * 4 + 3]);
    Eigen::Vector3f scal(scaling[pid * 3], scaling[pid * 3 + 1], scaling[pid * 3 + 2]);
    Eigen::Matrix3f cov3D = ComputeCov3D(quat, scal);

    Eigen::Matrix2f cov2D = ComputeCov2D(viewPoint, viewMatrixE, cov3D, focalX, focalY, tanFovX, tanFovY);

    float a = cov2D(0, 0);
    float b = cov2D(0, 1);
    float c = cov2D(1, 0);
    float d = cov2D(1, 1);

    float det = (a * d - b * c);
    if (det == 0.0f) {
        return;
    }

    // 5.求近似圆
    float lambda1 = 0.5f * (a + d) + 0.5f * sqrtf((a - d) * (a - d) + 4 * b * c);
    float radius = ceilf(3.f * sqrtf(lambda1));

    // 6.tile覆盖
    


    if (pid == 1) {
        printf("%f\n", radius);
        //printf("viewPoint1: %f,%f,%f,%f\n", viewPoint[0], viewPoint[1], viewPoint[2], viewPoint[3]);
        
        //printf("viewPoint2: %f,%f,%f,%f\n", viewPoint[0], viewPoint[1], viewPoint[2], viewPoint[3]);
        /*printf("quat: %f,%f,%f,%f\n", quat[0], quat[1], quat[2], quat[3]);
        printf("scal: %f,%f,%f,%f\n", scal[0], scal[1], scal[2]);
        printf("cov3D: %f,%f,%f  %f,%f,%f  %f,%f,%f", 
            cov3D(0, 0), cov3D(0, 1), cov3D(0, 2), 
            cov3D(1, 0), cov3D(1, 1), cov3D(1, 2), 
            cov3D(2, 0), cov3D(2, 1), cov3D(2, 2));*/
    }
    
   
}

__global__ void Render() {

}

void  PreRender(
    const float* xyz, 
    const float* rotation,
    const float* scaling,

    const float near, 
    const int N, 
    const float* viewMatrix,
    const float* viewProjMatrix,
    float focalX, float focalY,
    float tanFovX, float tanFovY
) {
    float *d_xyz, *d_rotation, *d_scaling, *d_viewMatrix, *d_viewProjMatrix;
    cudaMalloc(&d_xyz, 3 * N * sizeof(float));
    cudaMemcpy(d_xyz, xyz, 3 * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_rotation, 4 * N * sizeof(float));
    cudaMemcpy(d_rotation, rotation, 4 * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_scaling, 3 * N * sizeof(float));
    cudaMemcpy(d_scaling, scaling, 3 * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_viewMatrix, 16 * sizeof(float));
    cudaMemcpy(d_viewMatrix, viewMatrix, 16 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_viewProjMatrix, 16 * sizeof(float));
    cudaMemcpy(d_viewProjMatrix, viewProjMatrix, 16 * sizeof(float), cudaMemcpyHostToDevice);

    PreRenderKernel <<<(N + 255) / 256, 256 >>> (
        d_xyz, 
        d_rotation, 
        d_scaling, 
        near, N, 
        d_viewMatrix, 
        d_viewProjMatrix,
        focalX, focalY,
        tanFovX, tanFovY);

    cudaFree(d_xyz);
    cudaFree(d_rotation);
    cudaFree(d_scaling);
    cudaFree(d_viewMatrix);
    cudaFree(d_viewProjMatrix);
}