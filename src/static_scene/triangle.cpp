#include "triangle.h"

#include "CMU462/CMU462.h"
#include "GL/glew.h"

namespace CMU462 { namespace StaticScene {

Triangle::Triangle(const Mesh* mesh, size_t v1, size_t v2, size_t v3) :
    mesh(mesh), v1(v1), v2(v2), v3(v3) { }

BBox Triangle::get_bbox() const {
  
  // TODO: 
  // compute the bounding box of the triangle
  
  return BBox();
}

bool Triangle::intersect(const Ray& r) const {
  
  // TODO: implement ray-triangle intersection
  Vector3D e1 = mesh->positions[v2] - mesh->positions[v1];
  Vector3D e2 = mesh->positions[v3] - mesh->positions[v1];
  Vector3D s = r.o - mesh->positions[v1];

  Vector3D e1Xd = cross(e1,r.d);
  Vector3D sXe2 = cross(s,e2);

  double denom = dot(e1Xd, e2);

  if(denom == 0)
    return false;

  double u = -dot(sXe2, r.d);
  double v = dot(e1Xd, s);
  double t = -dot(sXe2, e1);

  Vector3D sol = 1/denom * Vector3D(u,v,t);

  if(0 <= sol[0] && sol[0] < 1 && 
     0 <= sol[1] && sol[1] < 1 &&
     r.min_t <= sol[2] && sol[2] <= r.max_t)
  {
    r.max_t = sol[2];
    return true;
  }
  
  return false;
}

bool Triangle::intersect(const Ray& r, Intersection *isect) const {
  
  // TODO: 
  // implement ray-triangle intersection. When an intersection takes
  // place, the Intersection data should be updated accordingly
  
  Vector3D e1 = mesh->positions[v2] - mesh->positions[v1];
  Vector3D e2 = mesh->positions[v3] - mesh->positions[v1];
  Vector3D s = r.o - mesh->positions[v1];

  Vector3D e1Xd = cross(e1,r.d);
  Vector3D sXe2 = cross(s,e2);

  double denom = dot(e1Xd, e2);

  if(denom == 0)
    return false;

  double u = -dot(sXe2, r.d);
  double v = dot(e1Xd, s);
  double t = -dot(sXe2, e1);


  Vector3D sol = 1/denom * Vector3D(u,v,t);

  //printf("u:%f, v:%f, t:%f ",sol[0],sol[1],sol[2]);

  if(0 <= sol[0] && sol[0] + sol[1] < 1 && 
     0 <= sol[1] &&
     r.min_t <= sol[2] && sol[2] <= r.max_t)
  {
    //printf("intersect\n");
    Vector3D poi = r.o + sol[2] * r.d;
    isect->t = sol[2];
    r.max_t = sol[2];
    isect->primitive = this;
    isect->n = ((poi - mesh->positions[v1]).norm() * mesh->normals[v1] +
                (poi - mesh->positions[v2]).norm() * mesh->normals[v2] +
                (poi - mesh->positions[v3]).norm() * mesh->normals[v3]).unit();
    isect->bsdf = get_bsdf();
    return true;
  }
  //printf("miss\n");
  
  return false;
}

void Triangle::draw(const Color& c) const {
  glColor4f(c.r, c.g, c.b, c.a);
  glBegin(GL_TRIANGLES);
  glVertex3d(mesh->positions[v1].x,
             mesh->positions[v1].y,
             mesh->positions[v1].z);
  glVertex3d(mesh->positions[v2].x,
             mesh->positions[v2].y,
             mesh->positions[v2].z);
  glVertex3d(mesh->positions[v3].x,
             mesh->positions[v3].y,
             mesh->positions[v3].z);
  glEnd();
}

void Triangle::drawOutline(const Color& c) const {
  glColor4f(c.r, c.g, c.b, c.a);
  glBegin(GL_LINE_LOOP);
  glVertex3d(mesh->positions[v1].x,
             mesh->positions[v1].y,
             mesh->positions[v1].z);
  glVertex3d(mesh->positions[v2].x,
             mesh->positions[v2].y,
             mesh->positions[v2].z);
  glVertex3d(mesh->positions[v3].x,
             mesh->positions[v3].y,
             mesh->positions[v3].z);
  glEnd();
}



} // namespace StaticScene
} // namespace CMU462
