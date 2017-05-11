#include "gpuPathtracer.h"
#include "../static_scene/triangle.h"
#include "../static_scene/object.h"

using namespace CMU462;
using namespace StaticScene;

__global__ void render()
{
}

gpuPathTracer::gpuPathTracer(PathTracer *__pathtracer)
{
  pathtracer = __pathtracer;
}

void gpuPathTracer::load_scene()
{
  // using the CPU's bvh, load the mesh information
  size_t num_tri = pathtracer->bvh->primitives.size();
  const Mesh* cpu_mesh = ((Triangle*)(pathtracer->bvh->primitives[0]))->mesh;
  size_t numVerts = cpu_mesh->numVerts;

  // Copy over the vertices and normals of the mesh
  gpuVector3D *pos_d;
  gpuVector3D *norm_d;

  cudaMalloc((void**)&pos_d,sizeof(gpuVector3D) * numVerts);
  cudaMalloc((void**)&norm_d,sizeof(gpuVector3D) * numVerts);

  cudaMemcpy(pos_d, cpu_mesh->positions, sizeof(gpuVector3D) * numVerts,cudaMemcpyHostToDevice);
  cudaMemcpy(norm_d, cpu_mesh->normals, sizeof(gpuVector3D) * numVerts,cudaMemcpyHostToDevice);

  // Group the mesh info into a gpuMesh
  gpuMesh gpu_mesh(pos_d,norm_d);

  cudaMalloc((void**)&mesh,sizeof(gpuMesh));

  cudaMemcpy(mesh,&gpu_mesh, sizeof(gpuMesh),cudaMemcpyHostToDevice);

  // Copy over the triangles
  gpuTriangle* temp_tri = new gpuTriangle[num_tri];
  for(int i = 0; i < num_tri; i++)
  {
    temp_tri[i] = gpuTriangle(mesh,
        ((Triangle*)(pathtracer->bvh->primitives[i]))->v1,
        ((Triangle*)(pathtracer->bvh->primitives[i]))->v2,
        ((Triangle*)(pathtracer->bvh->primitives[i]))->v3);
  }

  cudaMalloc((void**)&primitives,sizeof(gpuTriangle) * num_tri);
  cudaMemcpy(primitives,temp_tri,sizeof(gpuTriangle) * num_tri,cudaMemcpyHostToDevice);
}

void gpuPathTracer::load_camera()
{
}

void gpuPathTracer::set_frame_size(size_t width, size_t height)
{
  w = width;
  h = height;

  // reallocate the imagePixels buffer
  cudaFree(imagePixels);
  cudaMalloc((void**)&imagePixels,sizeof(bool) * w * h);
}

void gpuPathTracer::update_screen()
{
}

void gpuPathTracer::start_raytrace()
{
}

void gpuPathTracer::build_accel()
{
}

__device__ bool gpuPathTracer::trace_ray(const gpuRay& ray)
{
}

__device__ bool gpuPathTracer::raytrace_pixel(size_t x, size_t y)
{
}


