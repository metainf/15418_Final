#ifndef GPU_TRIANGLE_H
#define GPU_TRIANGLE_H

#include "gpuBBox.h"
#include "gpuVector3D.h"
#include "gpuRay.h"
#include "gpuMesh.h"
#include "../static_scene/object.h"

using namespace CMU462;
using namespace StaticScene;

class gpuTriangle {
  public:
    __host__ gpuTriangle(){}

    __host__ gpuTriangle(const Mesh* mesh_cpu, const gpuMesh* mesh, 
        size_t v1, size_t v2, size_t v3);

    __device__ gpuBBox get_bbox() const;

    __device__ bool intersect(const gpuRay& r) const;

    __device__ gpuVector3D get_center();
  private:

    const gpuMesh* mesh;
    size_t v1;
    size_t v2;
    size_t v3;

    gpuVector3D centroid;
};
#endif
