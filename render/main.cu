#include "loadPLY.h"
#include "forward.h"

int main() {
    float* viewmatrix = GetViewMatrix();
    float* viewprojmatrix = GetViewProjMatrix();
    float* cameracenter = GetCameraCenter();
    float focalx = 4649.505977743847f;
    float focaly = 4627.300372546341f;
    float tanfovx = 0.5318844651104229f;
    float tanfovy = 0.3550666409615158f;
    int width = 4946;
    int height = 3286;
    
    std::vector<Point> points = LoadPLY("point_cloud.ply");
    float* xyz = new float[3 * points.size()];
    float* scaling = new float[3 * points.size()];
    float* rotation = new float[4 * points.size()];
    float* opacity = new float[points.size()];
    float* features = new float[48 * points.size()];
    GetPointList(points, xyz, scaling, rotation, opacity, features);

    PreRender(xyz, rotation, scaling, features, 0.2f, points.size(), width, height, viewmatrix, viewprojmatrix, cameracenter, focalx, focaly, tanfovx, tanfovy);

    delete[](xyz);
    delete[](scaling);
    delete[](rotation);
    delete[](opacity);
    delete[](features);
    delete[](viewmatrix);
    delete[](viewprojmatrix);
    delete[](cameracenter);

    return 0;
}



