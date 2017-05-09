#ifndef GPU_TRIANGLE_H
#define GPU_TRIANGLE_H

class gpuTriangle {
  public:
    __global__ gpuTriangle(const GPUMesh* mesh, size_t v1, size_t v2, size_t v3);

    __device__ gpuBBox get_bbox();

    __device__ bool intersect(const GPURay& r);

    __device__ Vector3G get_center();
  private:

    const GPUMesh* mesh;
    size_t v1;
    size_t v2;
    size_t v3;

    Vector3G centorid;
};

#endif
