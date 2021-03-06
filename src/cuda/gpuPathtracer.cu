#include "gpuPathtracer.h"
#include "../static_scene/triangle.h"
#include "../static_scene/object.h"
#include "gpuRay.cu"
#include "gpuTriangle.cu"
#include "gpuVector3D.cu"
#include "gpuCamera.cu"
//#include "gpuBvh.cu"
#include "gpuBBox.cu"

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
//__constant__ gpuCamera* camera_const;
__constant__ int* imagePixels_const;
__constant__ size_t w_d;
__constant__ size_t h_d;
__constant__ size_t numPrim;
__constant__ gpuVector3D* pos;

int* imagePixels;
gpuCamera* camera;
gpuTriangle* gpu_primitives;
gpuVector3D *pos_d;

// returns the result of ray tracing intersection with the scene primitives
__device__ int trace_ray(gpuRay ray)
{
  for(size_t i = 0; i < numPrim; i++)
  {
    if(primitives[i].intersect(ray))
      return 1;
  }
  return 0;
}

// Using the x and y position of the pixel, create a ray and use trace_ray
__device__ int raytrace_pixel(size_t x, size_t y,gpuCamera* cam)
{
  gpuVector3D p((x + 0.5)/w_d,(y + 0.5)/h_d,0);
  return trace_ray(cam->generate_ray(p.x,p.y));
}

// kernel for doing raytracing
__global__ void render(gpuCamera* cam)
{
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t x = index % w_d;
  size_t y = index / w_d;
  //printf("%u\n", index);
  if(x < w_d && y < h_d)
  {
    imagePixels_const[index] = raytrace_pixel(x,y,cam);
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
  timer.start();

  // using the CPU's bvh, load the mesh information
  size_t num_tri = pathtracer->bvh->primitives.size();
  printf("Triangles in scene: %u\n",num_tri);
  // Allocate the primitives/vertices
  cudaMalloc((void**)&pos_d,sizeof(gpuVector3D) * num_tri*3);
  cudaMalloc((void**)&gpu_primitives,sizeof(gpuTriangle) * num_tri);
  cudaMemcpyToSymbol(pos,&pos_d,sizeof(gpuVector3D*));
  cudaMemcpyToSymbol(primitives,&gpu_primitives,sizeof(gpuTriangle*));
  cudaMemcpyToSymbol(numPrim,&num_tri,sizeof(size_t));

  // Copy over the triangles and their vertices into gpu versions
  gpuTriangle* temp_tri = new gpuTriangle[num_tri];
  gpuVector3D* pos_temp = new gpuVector3D[num_tri*3];

  for(size_t i = 0; i < num_tri; i++)
  {
    // Get the triangle and its vectors
    size_t offset = i * 3;
    Triangle* tri = ((Triangle*)(pathtracer->bvh->primitives[i]));
    Vector3D p1 = tri->mesh->positions[tri->v1];
    Vector3D p2 = tri->mesh->positions[tri->v2];
    Vector3D p3 = tri->mesh->positions[tri->v3];

    // Copy over the vectors
    pos_temp[offset] = gpuVector3D(p1.x,p1.y,p1.z);
    pos_temp[offset+1] = gpuVector3D(p2.x,p2.y,p2.z);
    pos_temp[offset+2] = gpuVector3D(p3.x,p3.y,p3.z);

    // Create the gpuTriangle object
    temp_tri[i] = gpuTriangle(pos_d,offset,offset+1,offset+2);
  }

  // Copy over the vertices and normals of the triangles


  cudaMemcpy(pos_d, pos_temp, sizeof(gpuVector3D) * num_tri*3,
      cudaMemcpyHostToDevice);


  cudaMemcpy(gpu_primitives,temp_tri,sizeof(gpuTriangle) * num_tri,
      cudaMemcpyHostToDevice);
  timer.stop();

  printf("[GPU Pathtracer]: finished loading scene (%.4f sec)\n",timer.duration());
}

void gpuPathTracer::load_camera(Camera *cam)
{
  gpuCamera temp = gpuCamera(cam->c2w, cam->position(),
      cam->screenW, cam->screenH, cam->screenDist);
  cudaMalloc((void**)&camera,sizeof(gpuCamera));
  cudaMemcpy(camera,&temp,sizeof(gpuCamera),cudaMemcpyHostToDevice);
  cudaMemcpy(&temp,camera,sizeof(gpuCamera),cudaMemcpyDeviceToHost);
  printf("w %u, h %u, d %f\n",temp.screenW, temp.screenH, temp.screenDist);
  printf("w %u, h %u, d %f\n",cam->screenW, cam->screenH, cam->screenDist);
  //cudaMemcpyToSymbol(camera_const,camera,sizeof(gpuCamera*),cudaMemcpyHostToDevice);
}

void gpuPathTracer::set_frame_size(size_t width, size_t height)
{
  w = width;
  h = height;

  cudaMemcpyToSymbol(w_d,&w,sizeof(size_t));
  cudaMemcpyToSymbol(h_d,&h,sizeof(size_t));

  // reallocate the imagePixels buffer
  cudaMalloc((void**)&imagePixels,sizeof(int) * w * h);
  cudaMemcpyToSymbol(imagePixels_const,&imagePixels,sizeof(int*));
}

// Takes the int imagePixels and draws it on the screen as b/w pixels
void gpuPathTracer::update_screen()
{
  Color white(1, 1, 1, 1);
  Color black(0, 0, 0, 0);

  int *tmp = new int[w * h];
  cudaMemcpy(tmp, imagePixels, w * h * sizeof(int),
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
  timer.start();
  size_t numBlocks = (w * h + 512 -1)/512;
  render<<<numBlocks,512>>>(camera);
  cudaDeviceSynchronize();
  timer.stop();
  printf("[GPU Pathtracer]: finished rendering scene (%.4f sec)\n",timer.duration());
}

