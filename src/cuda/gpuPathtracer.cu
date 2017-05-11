#include "gpuPathtracer.h"
#include "../static_scene/triangle.h"
#include "../static_scene/object.h"
#include "gpuRay.h"
#include "gpuMesh.h"
#include "gpuTriangle.h"
#include "gpuCamera.h"

using namespace CMU462;
using namespace StaticScene;

gpuMesh* mesh;
gpuTriangle* primitives;
gpuCamera* camera;

// returns the result of ray tracing intersection with the scene primitives
__device__ bool trace_ray(const gpuRay& ray)
{
  //bvh->intersect(r);
}

// Using the x and y position of the pixel, create a ray and use trace_ray
__device__ bool raytrace_pixel(size_t x, size_t y)
{
}

// kernel for doing raytracing
__global__ void render()
{
}


gpuPathTracer::gpuPathTracer(PathTracer *__pathtracer)
{
  pathtracer = __pathtracer;
}

gpuPathTracer::~gpuPathTracer() {
  cudaFree(mesh);
  cudaFree(primitives);
  cudaFree(imagePixels);
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
  gpuMesh gpu_mesh_tmp(pos_d,norm_d);

  cudaMalloc((void**)&mesh,sizeof(gpuMesh));

  cudaMemcpy(mesh,&gpu_mesh_tmp, sizeof(gpuMesh),cudaMemcpyHostToDevice);

  // Copy over the triangles
  gpuTriangle* temp_tri = new gpuTriangle[num_tri];
  for(int i = 0; i < num_tri; i++)
  {
    temp_tri[i] = gpuTriangle(cpu_mesh,mesh,
        ((Triangle*)(pathtracer->bvh->primitives[i]))->v1,
        ((Triangle*)(pathtracer->bvh->primitives[i]))->v2,
        ((Triangle*)(pathtracer->bvh->primitives[i]))->v3);
  }

  cudaMalloc((void**)&primitives,sizeof(gpuTriangle) * num_tri);
  cudaMemcpy(primitives,temp_tri,sizeof(gpuTriangle) * num_tri,cudaMemcpyHostToDevice);
  printf("[GPU Pathtracer]: finished loading scene\n");
}

void gpuPathTracer::load_camera(Camera *cam)
{
  gpuCamera temp = gpuCamera(cam->c2w, cam->position(), cam->screenW, cam->screenH, cam->screenDist);
  //free the device camera
  cudaFree(camera);
  cudaMalloc((void**)&camera,sizeof(gpuCamera));
  cudaMemcpy(camera,&temp,sizeof(gpuCamera),cudaMemcpyHostToDevice);
}

void gpuPathTracer::set_frame_size(size_t width, size_t height)
{
  w = width;
  h = height;

  // reallocate the imagePixels buffer
  cudaFree(imagePixels);
  cudaMalloc((void**)&imagePixels,sizeof(bool) * w * h);
}

// Takes the bool imagePixels and draws it on the screen as b/w pixels
void gpuPathTracer::update_screen()
{
  Color white(1, 1, 1, 1);
  Color black(0, 0, 0, 0);

  bool *tmp = new bool[w * h];
  cudaMemcpy(tmp, imagePixels, w * h * sizeof(bool),  cudaMemcpyDeviceToHost);
  //copy imagePixels into pathtracer->frameBuffer
  for(size_t i = 0; i < h; i++) {
    for(size_t j = 0; j < w; j++) {
      if(tmp[i * w + j]) {
        pathtracer->frameBuffer.update_pixel(black, j, i);
      }
      else {
        pathtracer->frameBuffer.update_pixel(white, j, i);
      }
    }
  }
  delete[] tmp;
  pathtracer->doneState();
}

// Wrapper for lanching the render() kernel
void gpuPathTracer::start_raytrace()
{
  
}

