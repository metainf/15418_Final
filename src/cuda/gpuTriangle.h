#ifndef GPU_TRIANGLE_H
#define GPU_TRIANGLE_H

#include "gpuBBox.h"
#include "gpuVector3D"
#include "gpuRay.h"
#include "gpuMesh.h"

class gpuTriangle {
  public:
    __host__ gpuTriangle(const gpuMesh* mesh, size_t v1, size_t v2, size_t v3);

    __device__ gpuBBox get_bbox();

    __device__ bool intersect(const gpuRay& r);

    __device__ gpuVector3D get_center();
  private:

    const gpuMesh* mesh;
    size_t v1;
    size_t v2;
    size_t v3;

    gpuVector3D centorid;
};

#endif
