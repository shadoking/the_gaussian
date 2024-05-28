#include "loadPLY.h"
#include "forward.h"


int main() {
    float* viewMatrix = GetViewMatrix();
    float* viewProjMatrix = GetViewProjMatrix();
    float focalX = 4649.505977743847f;
    float focalY = 4627.300372546341f;
    float tanFovX = 0.5318844651104229f;
    float tanFovY = 0.3550666409615158f;
    
    std::vector<Point> points = LoadPLY("point_cloud.ply");
    float* xyz = new float[3 * points.size()];
    float* scaling = new float[3 * points.size()];
    float* rotation = new float[4 * points.size()];
    float* opacity = new float[points.size()];
    GetPointList(points, xyz, scaling, rotation, opacity);
    
    //std::cout << "ply:" << points[1].x << ", " << points[1].y << ", " << points[1].z << std::endl;
    //std::cout << "xyz:" << xyz[3] << ", " << xyz[4] << ", " << xyz[5] << std::endl;

    PreRender(xyz, rotation, scaling, 0.2f, points.size(), viewMatrix, viewProjMatrix, focalX, focalY, tanFovX, tanFovY);

    delete[](xyz);
    delete[](scaling);
    delete[](rotation);
    delete[](opacity);
    delete[](viewMatrix);
    delete[](viewProjMatrix);

    return 0;
}
