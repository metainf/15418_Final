#include "gpuVector3D.cu"
#include "gpuMatrix3x3.cu"
#include "gpuRay.cu"
#include "CMU462/CMU462.h"
#include "CMU462/matrix3x3.h"
using namespace CMU462;

class gpuCamera {
  public:
    
    __host__ gpuCamera(Matrix3x3 c2w, Vector3D p, size_t screenW, size_t screenH, double screenDist);
    
    __device__ gpuRay generate_ray(double x, double y) const;

  private:
    gpuVector3D pos;
    gpuMatrix3x3 c2w;
    size_t screenW, screenH;
    double screenDist;

};
gpuCamera::gpuCamera(Matrix3x3 cameraToWorld, Vector3D p, 
    size_t screenW, size_t screenH, double screenDist) 
  : screenW(screenW), screenH(screenH), screenDist(screenDist)
{
  pos = gpuVector3D(p.x, p.y, p.z);
  
  c2w = gpuMatrix3x3();
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++) {
      c2w(i, j) = cameraToWorld(i, j);
    }
  }
}

__device__ gpuRay gpuCamera::generate_ray(double x, double y) const {
  gpuVector3D sp_cam = gpuVector3D(-(x - 0.5) * screenW / screenDist,
                                   -(y - 0.5) * screenH / screenDist, 1);
  gpuVector3D dir_cam = -sp_cam;

  gpuVector3D sp_world = c2w * sp_cam + pos;
  gpuVector3D dir_world = c2w * dir_cam;
  dir_world.normalize();

  return gpuRay(sp_world, dir_world);
}
