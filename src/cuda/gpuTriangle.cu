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

    __host__ gpuTriangle(const Mesh* mesh_cpu, 
        size_t v1, size_t v2, size_t v3);

    __device__ gpuBBox get_bbox(gpuVector3D* pos);

    __device__ bool intersect(gpuRay r,gpuVector3D* pos);

    __device__ gpuVector3D get_center();
    
    size_t v1;
    size_t v2;
    size_t v3;
  private:
    gpuVector3D centroid;
};

__host__
gpuTriangle::gpuTriangle(const Mesh* mesh_cpu,
    size_t v1, size_t v2, size_t v3):
  v1(v1), v2(v2), v3(v3){
    Vector3D p1 = mesh_cpu->positions[v1];
    Vector3D p2 = mesh_cpu->positions[v2];
    Vector3D p3 = mesh_cpu->positions[v3];

    Vector3D c = (p1 + p2 + p3)/3;
    centroid = gpuVector3D(c.x,c.y,c.z);
  }

__device__
gpuBBox gpuTriangle::get_bbox(gpuVector3D* pos){

  // TODO: 
  // compute the bounding box of the triangle
  gpuVector3D p1 = pos[v1];
  gpuVector3D p2 = pos[v2];
  gpuVector3D p3 = pos[v3];

  gpuVector3D min = gpuVector3D(fmin(p1.x,fmin(p2.x,p3.x)),
      fmin(p1.y,fmin(p2.y,p3.y)),
      fmin(p1.z,fmin(p2.z,p3.z)));

  gpuVector3D max = gpuVector3D(fmax(p1.x,fmax(p2.x,p3.x)),
      fmax(p1.y,fmax(p2.y,p3.y)),
      fmax(p1.z,fmax(p2.z,p3.z)));
  return gpuBBox(min,max);
}

__device__
bool gpuTriangle::intersect(gpuRay r, gpuVector3D* pos){

  // TODO: implement ray-triangle intersection
  gpuVector3D e1 = pos[v2] - pos[v1];
  gpuVector3D e2 = pos[v3] - pos[v1];
  gpuVector3D s = r.o - pos[v1];

  gpuVector3D e1Xd = cross(e1,r.d);
  gpuVector3D sXe2 = cross(s,e2);

  double denom = s.x;//dot(e1Xd, e2);

  if(denom == 0.0) {
    return false;
  }

  return true;
  double u = -dot(sXe2, r.d);
  double v = dot(e1Xd, s);
  double t = -dot(sXe2, e1);

  gpuVector3D sol = 1/denom * gpuVector3D(u,v,t);
  printf("%f\n",sol[2]);

  if(0 <= sol[0] && sol[0] < 1 && 
      0 <= sol[1] && sol[1] < 1 &&
      r.min_t <= sol[2] && sol[2] <= r.max_t)
  {
    r.max_t = sol[2];
    return true;
  }

  return false;
}

__device__
gpuVector3D gpuTriangle::get_center(){
  return centroid;
}
