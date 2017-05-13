#ifndef gpu_triangle
#define gpu_triangle

#include "gpuBBox.cu"
#include "gpuVector3D.cu"
#include "gpuRay.cu"
#include "gpuMesh.cu"
#include "../static_scene/object.h"

using namespace CMU462;
using namespace StaticScene;

class gpuTriangle {
  public:
    __host__ gpuTriangle(){}

    __host__ gpuTriangle(gpuVector3D* pos, 
        size_t v1, size_t v2, size_t v3);

    __device__ gpuBBox get_bbox();

    __device__ bool intersect(gpuRay r);

    __device__ gpuVector3D get_center();
    
    size_t v1;
    size_t v2;
    size_t v3;
  private:
    gpuVector3D* pos;
};

__host__
gpuTriangle::gpuTriangle(gpuVector3D* pos,
    size_t v1, size_t v2, size_t v3):
  pos(pos), v1(v1), v2(v2), v3(v3){}

__device__
gpuBBox gpuTriangle::get_bbox(){

  // TODO: 
  // compute the bounding box of the triangle
  gpuVector3D p1 = pos[v1];
  gpuVector3D p2 = pos[v2];
  gpuVector3D p3 = pos[v3];

  gpuVector3D min = gpuVector3D(fminf(p1.x,fminf(p2.x,p3.x)),
      fminf(p1.y,fminf(p2.y,p3.y)),
      fminf(p1.z,fminf(p2.z,p3.z)));

  gpuVector3D max = gpuVector3D(fmaxf(p1.x,fmaxf(p2.x,p3.x)),
      fmaxf(p1.y,fmaxf(p2.y,p3.y)),
      fmaxf(p1.z,fmaxf(p2.z,p3.z)));
  return gpuBBox(min,max);
}

__device__
bool gpuTriangle::intersect(gpuRay r){

  // TODO: implement ray-triangle intersection
  gpuVector3D e1 = pos[v2] - pos[v1];
  gpuVector3D e2 = pos[v3] - pos[v1];
  gpuVector3D s = r.o - pos[v1];

  gpuVector3D e1Xd = cross(e1,r.d);
  gpuVector3D sXe2 = cross(s,e2);

  float denom = dot(e1Xd, e2);

  if(denom == 0.0) {
    return false;
  }

  float u = -dot(sXe2, r.d);
  float v = dot(e1Xd, s);
  float t = -dot(sXe2, e1);

  gpuVector3D sol = 1/denom * gpuVector3D(u,v,t);

  if(0 <= sol[0] && 0 <= sol[1] && 
      sol[0] + sol[1] < 1 &&
      r.min_t <= sol[2] && sol[2] <= r.max_t)
  {
    r.max_t = sol[2];
    return true;
  }

  return false;
}

__device__
gpuVector3D gpuTriangle::get_center(){
    return (pos[v1] + pos[v2] + pos[v3])/3;
}
#endif
