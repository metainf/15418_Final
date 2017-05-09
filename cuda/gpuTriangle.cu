#include "gpuTriangle.h"

__global__
gpuTriangle::gpuTriangle(const GPUMesh* mesh, size_t v1, size_t v2, size_t v3):
  mesh(mesh), v1(v1), v2(v2), v3(v3){
    Vector3G p1 = mesh->positions[v1];
    Vector3G p2 = mesh->positions[v2];
    Vector3G p3 = mesh->positions[v3];

    centroid = (p1 + p2 + p3)/3;
  }

gpuBBox gpuTriangle::get_bbox() const {

  // TODO: 
  // compute the bounding box of the triangle
  Vector3G p1 = mesh->positions[v1];
  Vector3G p2 = mesh->positions[v2];
  Vector3G p3 = mesh->positions[v3];

  Vector3G min = Vector3G(fmin(p1.x,fmin(p2.x,p3.x)),
      fmin(p1.y,fmin(p2.y,p3.y)),
      fmin(p1.z,fmin(p2.z,p3.z)));

  Vector3G max = Vector3G(fmax(p1.x,fmax(p2.x,p3.x)),
      fmax(p1.y,fmax(p2.y,p3.y)),
      fmax(p1.z,fmax(p2.z,p3.z)));
  return gpuBBox(min,max);
}

bool gpuTriangle::intersect(const gpuRay& r) const {

  // TODO: implement ray-triangle intersection
  Vector3G e1 = mesh->positions[v2] - mesh->positions[v1];
  Vector3G e2 = mesh->positions[v3] - mesh->positions[v1];
  Vector3G s = r.o - mesh->positions[v1];

  Vector3G e1Xd = cross(e1,r.d);
  Vector3G sXe2 = cross(s,e2);

  double denom = dot(e1Xd, e2);

  if(denom == 0)
    return false;

  double u = -dot(sXe2, r.d);
  double v = dot(e1Xd, s);
  double t = -dot(sXe2, e1);

  Vector3G sol = 1/denom * Vector3G(u,v,t);

  if(0 <= sol[0] && sol[0] < 1 && 
      0 <= sol[1] && sol[1] < 1 &&
      r.min_t <= sol[2] && sol[2] <= r.max_t)
  {
    r.max_t = sol[2];
    return true;
  }

  return false;
}

Vector3G gpuTriangle::get_center(){
  return centroid;
}
