#include "loadPLY.h"


std::vector<Point> LoadPLY(char* filepath) {
	std::ifstream file(filepath, std::ios::binary);
	std::string line;
	std::vector<Point> points;

    while (std::getline(file, line) && line != "end_header") { }

    while (!file.eof()) {
        Point p;
        file.read(reinterpret_cast<char*>(&p.x), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.y), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.z), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.nx), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.ny), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.nz), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.f_dc_0), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.f_dc_1), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.f_dc_2), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.f_rest_0), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.f_rest_1), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.f_rest_2), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.f_rest_3), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.f_rest_4), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.f_rest_5), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.f_rest_6), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.f_rest_7), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.f_rest_8), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.f_rest_9), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.f_rest_10), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.f_rest_11), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.f_rest_12), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.f_rest_13), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.f_rest_14), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.f_rest_15), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.f_rest_16), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.f_rest_17), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.f_rest_18), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.f_rest_19), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.f_rest_20), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.f_rest_21), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.f_rest_22), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.f_rest_23), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.f_rest_24), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.f_rest_25), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.f_rest_26), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.f_rest_27), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.f_rest_28), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.f_rest_29), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.f_rest_30), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.f_rest_31), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.f_rest_32), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.f_rest_33), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.f_rest_34), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.f_rest_35), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.f_rest_36), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.f_rest_37), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.f_rest_38), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.f_rest_39), sizeof(float)); 
        file.read(reinterpret_cast<char*>(&p.f_rest_40), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.f_rest_41), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.f_rest_42), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.f_rest_43), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.f_rest_44), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.opacity), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.scale_0), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.scale_1), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.scale_2), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.rot_0), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.rot_1), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.rot_2), sizeof(float));
        file.read(reinterpret_cast<char*>(&p.rot_3), sizeof(float));

        points.push_back(p);
    }
    points.pop_back();
    return points;
}

void GetPointList(std::vector<Point> points, 
    float* xyz,
    float* scaling,
    float* rotation,
    float* opacity
    ) {
    for (int i = 0; i < points.size(); i++) {
        xyz[i * 3] = points[i].x;
        xyz[i * 3 + 1] = points[i].y;
        xyz[i * 3 + 2] = points[i].z;

        rotation[i * 4] = points[i].rot_0;
        rotation[i * 4 + 1] = points[i].rot_1;
        rotation[i * 4 + 2] = points[i].rot_2;
        rotation[i * 4 + 3] = points[i].rot_3;

        opacity[i] = points[i].opacity;

        scaling[i * 3] = points[i].scale_0;
        scaling[i * 3 + 1] = points[i].scale_1;
        scaling[i * 3 + 2] = points[i].scale_2;
    }
}

//[[ 0.6091, 0.1589, -0.7770, -0.1297],
//[0.2259, 0.9044, 0.3621, -0.0375],
//[0.7602, -0.3961, 0.5150, 4.1264],
//[0.0000, 0.0000, 0.0000, 1.0000]]

float* GetViewMatrix() {
    float* viewMatrix = new float[16];
    viewMatrix[0] = 0.6091f;
    viewMatrix[1] = 0.1589f;
    viewMatrix[2] = -0.7770f;
    viewMatrix[3] = -0.1297f;

    viewMatrix[4] = 0.2259f;
    viewMatrix[5] = 0.9044f;
    viewMatrix[6] = 0.3621f;
    viewMatrix[7] = -0.0375f;

    viewMatrix[8] = 0.7602f;
    viewMatrix[9] = -0.3961f;
    viewMatrix[10] = 0.5150f;
    viewMatrix[11] = 4.1068f;

    viewMatrix[12] = 0.0f;
    viewMatrix[13] = 0.0f;
    viewMatrix[14] = 0.0f;
    viewMatrix[15] = 1.0f;

    return viewMatrix;
}

//[[1.1452, 0.2988, -1.4608, -0.2439],
//[0.6362, 2.5470, 1.0198, -0.1057],
//[0.7603, -0.3961, 0.5150, 4.1068],
//[0.7602, -0.3961, 0.5150, 4.1264]]

float* GetViewProjMatrix() {
    float* viewProjMatrix = new float[16];
    viewProjMatrix[0] = 1.1452f;
    viewProjMatrix[1] = 0.2988f;
    viewProjMatrix[2] = -1.4608f;
    viewProjMatrix[3] = -0.2439f;

    viewProjMatrix[4] = 0.6362f;
    viewProjMatrix[5] = 2.5470f;
    viewProjMatrix[6] = 1.0198f;
    viewProjMatrix[7] = -0.1057f;

    viewProjMatrix[8] = 0.7603f;
    viewProjMatrix[9] = -0.3961f;
    viewProjMatrix[10] = 0.5150f;
    viewProjMatrix[11] = 4.1068f;

    viewProjMatrix[12] = 0.7602f;
    viewProjMatrix[13] = -0.3961f;
    viewProjMatrix[14] = 0.5150f;
    viewProjMatrix[15] = 4.1264f;

    return viewProjMatrix;
}