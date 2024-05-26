#include "loadPLY.h"
#include "forward.h"


int main() {
    float* viewMatrix = GetViewMatrix();
    float* viewProjMatrix = GetViewProjMatrix();
    
    std::vector<Point> points = LoadPLY("point_cloud.ply");
    float* xyz = new float[3 * points.size()];
    float* scaling = new float[3 * points.size()];
    float* rotation = new float[4 * points.size()];
    float* opacity = new float[points.size()];
    GetPointList(points, xyz, scaling, rotation, opacity);
    
    //std::cout << "ply:" << points[1].x << ", " << points[1].y << ", " << points[1].z << std::endl;
    //std::cout << "xyz:" << xyz[3] << ", " << xyz[4] << ", " << xyz[5] << std::endl;

    PreRender(xyz, 0.2f, points.size(), viewMatrix, viewProjMatrix);

    delete[](xyz);
    delete[](scaling);
    delete[](rotation);
    delete[](opacity);
    delete[](viewMatrix);

    return 0;
}
