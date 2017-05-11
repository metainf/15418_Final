#include <stdio.h>
#include "hello.h"
#include "gpuVector3D.h"

#define CHK(ans) gpuAssert((ans), __FILE__, __LINE__);
#define POSTKERNEL CHK(cudaPeekAtLastError())

using namespace CMU462;

inline void gpuAssert(cudaError_t code, const char *file, int line)
{
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %s\n",
        cudaGetErrorString(code),file,line);
  }
}

const int N = 16; 
const int blocksize = 16; 

__global__ void vector3DTest(gpuVector3D* v1, gpuVector3D* v2, gpuVector3D* v3)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index == 1)
  {
   *v3  = cross(*v1,*v2);
  }
}

int main_test()
{
  char a[N] = "Hello \0\0\0\0\0\0";
  int b[N] = {15, 10, 6, 0, -11, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  char *ad;
  int *bd;
  const int csize = N*sizeof(char);
  const int isize = N*sizeof(int);

  printf("%s", a);

  cudaMalloc( (void**)&ad, csize ); 
  cudaMalloc( (void**)&bd, isize ); 
  cudaMemcpy( ad, a, csize, cudaMemcpyHostToDevice ); 
  cudaMemcpy( bd, b, isize, cudaMemcpyHostToDevice ); 

  dim3 dimBlock( blocksize, 1 );
  dim3 dimGrid( 1, 1 );
  hello<<<dimGrid, dimBlock>>>(ad, bd);
  cudaMemcpy( a, ad, csize, cudaMemcpyDeviceToHost ); 
  cudaFree( ad );
  cudaFree( bd );

  printf("%s\n", a);
  Vector3D v1(1,1,3);
  gpuVector3D v2(1,1,1);
  gpuVector3D v3;
  
  gpuVector3D *v1d;
  gpuVector3D *v2d;
  gpuVector3D *v3d;
  
  cudaMalloc((void**)&v1d,sizeof(gpuVector3D));
  cudaMalloc((void**)&v2d,sizeof(gpuVector3D));
  cudaMalloc((void**)&v3d,sizeof(gpuVector3D));

  cudaMemcpy(v1d,&v1,sizeof(gpuVector3D),cudaMemcpyHostToDevice);
  cudaMemcpy(v2d,&v2,sizeof(gpuVector3D),cudaMemcpyHostToDevice);

  vector3DTest<<<1,32>>>(v1d,v2d,v3d);
  POSTKERNEL;
  cudaThreadSynchronize();
  
  cudaMemcpy(&v3,v3d,sizeof(gpuVector3D),cudaMemcpyDeviceToHost);
  printf("%f, %f ,%f\n",v3[0],v3[1],v3[2]);

  return EXIT_SUCCESS;
}
