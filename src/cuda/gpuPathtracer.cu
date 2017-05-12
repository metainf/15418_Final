#include "gpuPathtracer.h"
#include "../static_scene/triangle.h"
#include "../static_scene/object.h"
#include "gpuRay.cu"
#include "gpuMesh.cu"
#include "gpuTriangle.cu"
#include "gpuVector3D.cu"
#include "gpuCamera.cu"

#ifdef DEBUG
#define CHK(ans) {gpuAssert((ans), __FILE__, __LINE__);}
#define POSTKERNEL CHK(cudaPeekAtLastError())
#else
#define CHK(ans)
#endif
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %s\n",
        cudaGetErrorString(code),file,line);
    exit(code);
  }
}
using namespace CMU462;
using namespace StaticScene;

__constant__ gpuTriangle* primitives;
__constant__ gpuCamera* camera_const;
__constant__ bool* imagePixels_const;
__constant__ size_t w_d;
__constant__ size_t h_d;
__constant__ size_t numPrim;
__constant__ gpuVector3D* pos;

bool* imagePixels;
gpuCamera* camera;
gpuTriangle* gpu_primitives;
gpuVector3D *pos_d;

// returns the result of ray tracing intersection with the scene primitives
__device__ bool trace_ray(gpuRay ray)
{
  for(size_t i = 0; i < numPrim; i++)
  {
    if(primitives[i].intersect(ray,pos))
      return true;
  }
  return false;
}

// Using the x and y position of the pixel, create a ray and use trace_ray
__device__ bool raytrace_pixel(size_t x, size_t y)
{
  gpuVector3D p((x + 0.5)/w_d,(y + 0.5)/h_d,0);
  return trace_ray(camera_const->generate_ray(p.x,p.y));
}

// kernel for doing raytracing
__global__ void render()
{
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t x = index % w_d;
  size_t y = index / w_d;
  //printf("%u\n", index);
  if(x < w_d && y < h_d)
  {
    imagePixels_const[index] = raytrace_pixel(x,y);
  }
}


gpuPathTracer::gpuPathTracer(PathTracer *__pathtracer)
{
  pathtracer = __pathtracer;
}

gpuPathTracer::~gpuPathTracer() {
  cudaFree(camera);
  cudaFree(imagePixels);
  cudaFree(gpu_primitives);
  cudaFree(pos_d);
}

void gpuPathTracer::load_scene()
{
  // using the CPU's bvh, load the mesh information
  size_t num_tri = pathtracer->bvh->primitives.size();
  cudaMemcpyToSymbol(numPrim,&num_tri,sizeof(size_t));

  const Mesh* cpu_mesh = ((Triangle*)(pathtracer->bvh->primitives[0]))->mesh;
  size_t numVerts = cpu_mesh->numVerts;
  printf("numVerts: %d\n",numVerts);

  gpuVector3D* pos_temp = new gpuVector3D[numVerts];
  
  for(size_t i = 0; i < numVerts; i++)
  {
    pos_temp[i] = gpuVector3D(cpu_mesh->positions[i].x,
        cpu_mesh->positions[i].y,
        cpu_mesh->positions[i].z);
  }
  // Copy over the vertices and normals of the mesh

  cudaMalloc((void**)&pos_d,sizeof(gpuVector3D) * numVerts);

  cudaMemcpy(pos_d, pos_temp, sizeof(gpuVector3D) * numVerts,
      cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(pos,&pos_d,sizeof(gpuVector3D*));

  // Copy over the triangles
  gpuTriangle* temp_tri = new gpuTriangle[num_tri];
  for(int i = 0; i < num_tri; i++)
  {
    temp_tri[i] = gpuTriangle(cpu_mesh,
        ((Triangle*)(pathtracer->bvh->primitives[i]))->v1,
        ((Triangle*)(pathtracer->bvh->primitives[i]))->v2,
        ((Triangle*)(pathtracer->bvh->primitives[i]))->v3);
  }
  cudaMalloc((void**)&gpu_primitives,sizeof(gpuTriangle) * num_tri);
  cudaMemcpy(gpu_primitives,temp_tri,sizeof(gpuTriangle) * num_tri,
      cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(primitives,&gpu_primitives,sizeof(gpuTriangle*));

  printf("[GPU Pathtracer]: finished loading scene\n");
}

void gpuPathTracer::load_camera(Camera *cam)
{
  gpuCamera temp = gpuCamera(cam->c2w, cam->position(),
      cam->screenW, cam->screenH, cam->screenDist);
  cudaMalloc((void**)&camera,sizeof(gpuCamera));
  cudaMemcpy(camera,&temp,sizeof(gpuCamera),cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(camera_const,&camera,sizeof(gpuCamera*),cudaMemcpyHostToDevice);
}

void gpuPathTracer::set_frame_size(size_t width, size_t height)
{
  w = width;
  h = height;

  cudaMemcpyToSymbol(w_d,&w,sizeof(size_t));
  cudaMemcpyToSymbol(h_d,&h,sizeof(size_t));

  // reallocate the imagePixels buffer
  cudaMalloc((void**)&imagePixels,sizeof(bool) * w * h);
  cudaMemcpyToSymbol(imagePixels_const,&imagePixels,sizeof(bool*));
}

// Takes the bool imagePixels and draws it on the screen as b/w pixels
void gpuPathTracer::update_screen()
{
  Color white(1, 1, 1, 1);
  Color black(0, 0, 0, 0);

  bool *tmp = new bool[w * h];
  cudaMemcpy(tmp, imagePixels, w * h * sizeof(bool),
      cudaMemcpyDeviceToHost);
  //copy imagePixels into pathtracer->frameBuffer
  for(size_t i = 0; i < h; i++) {
    for(size_t j = 0; j < w; j++) {
      if(tmp[i * w + j]) {
        pathtracer->frameBuffer.update_pixel(white, j, i);
      }
      else {
        pathtracer->frameBuffer.update_pixel(black, j, i);
      }
    }
  }
  delete[] tmp;
  pathtracer->doneState();
}

// Wrapper for lanching the render() kernel
void gpuPathTracer::start_raytrace()
{
  size_t numBlocks = (w * h + 31 -1)/32;
  render<<<numBlocks,32>>>();
  cudaDeviceSynchronize();
  printf("[GPU Pathtracer]: finished rendering scene\n");
}
