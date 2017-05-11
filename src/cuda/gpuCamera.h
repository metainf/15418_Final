#ifndef GPU_CAMERA_H
#define GPU_CAMERA_H

#include "gpuVector3D.h"
#include "gpuMatrix3x3.h"
#include "gpuRay.h"
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

#endif
