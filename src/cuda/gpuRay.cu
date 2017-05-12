#ifndef GPU_RAY_H
#define GPU_RAY_H
#include "math_constants.h"
#include "gpuVector3D.cu"

struct gpuRay {
  size_t depth;

  gpuVector3D o;
  gpuVector3D d;
  double min_t;
  double max_t;

  gpuVector3D inv_d;
  int sign[3];

  /* Constructors */
  __device__
    gpuRay(gpuVector3D o, gpuVector3D d, int depth = 0)
      : o(o), d(d), min_t(0.0), depth(depth) {
    inv_d = gpuVector3D(1 / d.x, 1 / d.y, 1 / d.z);
    sign[0] = (inv_d.x < 0);
    sign[1] = (inv_d.y < 0);
    sign[2] = (inv_d.z < 0);
    max_t = CUDART_INF;
  }

  __device__
    inline gpuVector3D at_time(double t){ return o + t * d; }

  /*
  gpuRay transform_by(const gpuMatrix4x4& t) const {
    const gpuVector4D& new0 = t * gpuVector4D(o, 1.0);
    return gpuRay((new0 / new0.w).to3D(), (t * gpuVector4D(d, 0.0)).to3D());
    
  }*/
};

#endif
