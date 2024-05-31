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

__device__ void GetTiles(
    Eigen::Vector2i imgPoint, 
    float radius, 
    Eigen::Vector2i &rectMin, 
    Eigen::Vector2i &rectMax, 
    dim3 grid) {
    rectMin[0] = min(grid.x, max(0, static_cast<int>((imgPoint[0] - radius) / BLOCK_X)));
    rectMin[1] = min(grid.y, max(0, static_cast<int>((imgPoint[1] - radius) / BLOCK_Y)));

    rectMax[0] = min(grid.x, max(0, static_cast<int>((imgPoint[0] + radius + BLOCK_X - 1) / BLOCK_X)));
    rectMax[1] = min(grid.y, max(0, static_cast<int>((imgPoint[1] + radius + BLOCK_Y - 1) / BLOCK_Y)));
}

__device__ void GetShs(Eigen::Vector3f* sh, const float* features, int idx, int M) {
    for (int j = 0; j < M; j++) {
        sh[j][0] = features[idx * M * 3 + j * 3];
        sh[j][1] = features[idx * M * 3 + j * 3 + 1];
        sh[j][2] = features[idx * M * 3 + j * 3 + 2];
    }
}

__device__ Eigen::Vector3f ComputeColor(
    int idx, int deg, 
    Eigen::Vector4f oriPoint,
    Eigen::Vector3f center, 
    const float* features, int M) {
    Eigen::Vector3f* sh = new Eigen::Vector3f[M];
    GetShs(sh, features, idx, M);

    Eigen::Vector3f dir = oriPoint.head<3>() - center;
    dir.normalize();

    Eigen::Vector3f rgb = SH_C0 * sh[0];

    if (deg > 0) {
        float x = dir[0];
        float y = dir[1];
        float z = dir[2];

        rgb = rgb - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

        if (deg > 1) {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;
            rgb = rgb + 
                SH_C2[0] * xy * sh[4] +
                SH_C2[1] * yz * sh[5] +
                SH_C2[2] * (2.f * zz - xx - yy) * sh[6] +
                SH_C2[3] * xz * sh[7] +
                SH_C2[4] * (xx - yy) * sh[8];

            if (deg > 2) {
                rgb = rgb +
                    SH_C3[0] * y * (3.f * xx - yy) * sh[9] +
                    SH_C3[1] * xy * z * sh[10] +
                    SH_C3[2] * y * (4.f * zz - xx - yy) * sh[11] +
                    SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy) * sh[12] +
                    SH_C3[4] * x * (4.f * zz - xx - yy) * sh[13] +
                    SH_C3[5] * z * (xx - yy) * sh[14] +
                    SH_C3[6] * x * (xx - 3.f * yy) * sh[15];
            }
        }
    }
    rgb = rgb.array() + 2.f;

    delete[] sh;
    //TODO: backward clamp
    return rgb.array().max(0.f);
}

__device__ Eigen::Vector2i Ndc2Pix(Eigen::Vector3f projPoint, int width, int height) {
    Eigen::Vector2i pixel;
    pixel[0] = static_cast<int>((projPoint[0] + 1) * 0.5 * width);
    pixel[1] = static_cast<int>((1 - projPoint[1]) * 0.5 * height);

    return pixel;
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
    
    Eigen::Matrix3f W = viewMatrix.block<3, 3>(0, 0);

    Eigen::Matrix2f cov2D = J * W * cov3D * W.transpose() * J.transpose();

    cov2D(0, 0) += 0.3f;
    cov2D(1, 1) += 0.3f;

    return cov2D;
}

__device__ Eigen::Matrix3f ComputeCov3D(Eigen::Vector4f quat, Eigen::Vector3f scal) {
    Eigen::Vector4f quatNorm = quat.normalized();
    float w = quatNorm[0];
    float x = quatNorm[1];
    float y = quatNorm[2];
    float z = quatNorm[3];

    Eigen::Matrix3f R;
    R << 1.f - 2.f * y * y - 2.f * z * z, 2.f * x * y - 2.f * w * z, 2.f * x * z + 2.f * w * y,
        2.f * x * y + 2.f * w * z, 1.f - 2.f * x * x - 2.f * z * z, 2.f * y * z - 2.f * w * x,
        2.f * x * z - 2.f * w * y, 2.f * y * z + 2.f * w * x, 1.f - 2.f * x * x - 2.f * y * y;
    
    Eigen::Matrix3f cov3D;
    cov3D << scal[0], 0, 0, 0, scal[1], 0, 0, 0, scal[2];
    cov3D = R * cov3D;
    cov3D = cov3D * cov3D.transpose();

    return cov3D;
}

__global__ void DuplicateWithKeys(
    int N, 
    Eigen::Vector2i *imgXYZ, 
    float *depth,
    float *radii, 
    uint32_t* tileTouched,
    dim3 grid,
    TileDepth* tileDepthList) {

    int pid = blockDim.x * blockIdx.x + threadIdx.x;
    if (pid >= N) {
        return;
    }
    
    uint32_t off = (pid == 0) ? 0 : tileTouched[pid - 1];
    Eigen::Vector2i rectMin, rectMax;
    GetTiles(imgXYZ[pid], radii[pid], rectMin, rectMax, grid);

    for (int y = rectMin[1]; y < rectMax[1]; y++) {
        for (int x = rectMin[0]; x < rectMax[0]; x++) {
            TileDepth tileDepth;
            tileDepth.tileId = y * grid.x + x;
            tileDepth.depth = depth[pid];
            tileDepth.gaussianId = pid;

            tileDepthList[off] = tileDepth;
            off++;
        }
    }
}

__global__ void PreRenderKernel(
    const float* xyz,
    const float* rotation,
    const float* scaling,
    const float* features,

    const float near,
    const int N,
    const int width,
    const int height,
    const float* colorsPrecomp,
    const float* viewMatrix,
    const float* viewProjMatrix,
    const float* cameraCenter,
    float focalX, float focalY,
    float tanFovX, float tanFovY,
    uint32_t* tileTouched,
    const dim3 grid,
    Eigen::Vector2i* imgXYZ,
    float* depth,
    float* radii,
    float* rgb) {
    // 1.计算 pid
    int pid = blockDim.x * blockIdx.x + threadIdx.x;
    if (pid >= N) {
        return;
    }

    tileTouched[pid] = 0;

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
    float w = 1.f / (projPointH[3] + 0.0000001f);
    Eigen::Vector3f projPoint(projPointH[0] * w, projPointH[1] * w, projPointH[2] * w);


    // 4.求协方差矩阵
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
    Eigen::Vector2i imgPoint = Ndc2Pix(projPoint, width, height);
    Eigen::Vector2i rectMin, rectMax;
    GetTiles(imgPoint, radius, rectMin, rectMax, grid);
    if ((rectMax[1] - rectMin[1]) * (rectMax[0] - rectMin[0]) == 0) {
        return;
    }

    if (colorsPrecomp == nullptr) {
        Eigen::Vector3f center(cameraCenter[0], cameraCenter[1], cameraCenter[2]);
        Eigen::Vector3f rgbV = ComputeColor(pid, 3, oriPoint, center, features, 16);
        rgb[pid * 3] = rgbV[0];
        rgb[pid * 3 + 1] = rgbV[1];
        rgb[pid * 3 + 2] = rgbV[2];
    }

    depth[pid] = viewPoint[2];
    radii[pid] = radius;
    imgXYZ[pid] = imgPoint;
    tileTouched[pid] = (rectMax[1] - rectMin[1]) * (rectMax[0] - rectMin[0]);
}

void printStructArray(TileDepth* array, int start, int end) {
    for (int i = start; i < end; ++i) {
        std::cout << "gaussianId: " << array[i].gaussianId << ", tileId: " << array[i].tileId << ", depth: " << array[i].depth << std::endl;
    }
}

void PreRender(
    const float* xyz, 
    const float* rotation,
    const float* scaling,
    const float* features,

    const float near, 
    const int N, 
    const int width,
    const int height,
    const float* viewMatrix,
    const float* viewProjMatrix,
    const float* cameraCenter,
    float focalX, float focalY,
    float tanFovX, float tanFovY) {

    float *d_xyz, *d_rotation, *d_scaling, *d_featrues, *d_viewMatrix, *d_viewProjMatrix, *d_cameraCenter;
    cudaMalloc(&d_xyz, 3 * N * sizeof(float));
    cudaMemcpy(d_xyz, xyz, 3 * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_rotation, 4 * N * sizeof(float));
    cudaMemcpy(d_rotation, rotation, 4 * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_scaling, 3 * N * sizeof(float));
    cudaMemcpy(d_scaling, scaling, 3 * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_featrues, 48 * N * sizeof(float));
    cudaMemcpy(d_featrues, features, 48 * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_viewMatrix, 16 * sizeof(float));
    cudaMemcpy(d_viewMatrix, viewMatrix, 16 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_viewProjMatrix, 16 * sizeof(float));
    cudaMemcpy(d_viewProjMatrix, viewProjMatrix, 16 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_cameraCenter, 3 * sizeof(float));
    cudaMemcpy(d_cameraCenter, cameraCenter, 3 * sizeof(float), cudaMemcpyHostToDevice);

    uint32_t* tileTouched;
    cudaMalloc(&tileTouched, N * sizeof(uint32_t));

    float* r_rgb, * r_depth, * r_radii;
    Eigen::Vector2i *r_imgXYZ;
    cudaMalloc(&r_rgb, 3 * N * sizeof(float));
    cudaMalloc(&r_depth, N * sizeof(float));
    cudaMalloc(&r_radii, N * sizeof(float));
    cudaMalloc(&r_imgXYZ, N * sizeof(Eigen::Vector2i));

    dim3 grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
    dim3 block(BLOCK_X, BLOCK_Y, 1);

    PreRenderKernel <<<(N + 255) / 256, 256>>> (
        d_xyz, 
        d_rotation, 
        d_scaling, 
        d_featrues,

        near, N, 
        width, height,
        nullptr,
        d_viewMatrix, 
        d_viewProjMatrix,
        d_cameraCenter,
        focalX, focalY,
        tanFovX, tanFovY,
        tileTouched,
        grid,
        r_imgXYZ,
        r_depth,
        r_radii,
        r_rgb);

    // cudaDeviceSynchronize();
    // 1记录每个tile关联几个GS 2求前缀和，得知最终组装的索引 3组装排序列表 4排序

    // 求和
    thrust::device_ptr<uint32_t> tileTouchedPtr(tileTouched);
    thrust::inclusive_scan(tileTouchedPtr, tileTouchedPtr + N, tileTouchedPtr);

    uint32_t numRendered;
    cudaMemcpy(&numRendered, tileTouched + N - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    TileDepth *tileDepthList;
    cudaMalloc(&tileDepthList, numRendered * sizeof(TileDepth));

    // 组装 depth tileId
    DuplicateWithKeys<<<(N + 255) / 256, 256>>>(N, r_imgXYZ, r_depth, r_radii, tileTouched, grid, tileDepthList);
    
    // 排序
    thrust::device_ptr<TileDepth> tileDepthListPtr(tileDepthList);
    thrust::sort(tileDepthListPtr, tileDepthListPtr + numRendered);

    /*TileDepth* test = new TileDepth[numRendered];
    cudaMemcpy(test, tileDepthList, numRendered * sizeof(TileDepth), cudaMemcpyDeviceToHost);
    printStructArray(test, N - 100, N);*/
 

    //delete[] test;
    cudaFree(d_xyz);
    cudaFree(d_rotation);
    cudaFree(d_scaling);
    cudaFree(d_featrues);
    cudaFree(d_viewMatrix);
    cudaFree(d_viewProjMatrix);
    cudaFree(d_cameraCenter);
    cudaFree(tileTouched);
    cudaFree(r_imgXYZ);
    cudaFree(r_depth);
    cudaFree(r_radii);
    cudaFree(r_rgb);
    cudaFree(tileDepthList);
}