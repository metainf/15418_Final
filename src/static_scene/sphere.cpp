#include "sphere.h"

#include <cmath>

#include  "../bsdf.h"
#include "../misc/sphere_drawing.h"

namespace CMU462 { namespace StaticScene {

bool Sphere::test(const Ray& ray, double& t1, double& t2) const {

  // TODO:
  // Implement ray - sphere intersection test.
  // Return true if there are intersections and writing the
  // smaller of the two intersection times in t1 and the larger in t2.
  
  Vector3D L = o - ray.o;
  double t_ca = dot(L,ray.d);
  if(t_ca < 0)
    return false;

  double d2 = dot(L,L)-t_ca*t_ca;
  if(d2 > r2)
    return false;

  double t_hc = sqrt(r2 - d2);

  double sol1 = t_ca - t_hc;
  double sol2 = t_ca + t_hc;

  if(sol1 < sol2){
    t1 = sol1;
    t2 = sol2;
  }
  else{
    t1 = sol2;
    t2 = sol1;
  }

  return true;

}

bool Sphere::intersect(const Ray& r) const {

  // TODO:
  // Implement ray - sphere intersection.
  // Note that you might want to use the the Sphere::test helper here.
  double t1,t2;
  bool intersect = test(r,t1,t2);
  if(t1 < 0 || t2 < 0)
    return false;
  return true;
}

bool Sphere::intersect(const Ray& r, Intersection *i) const {

  // TODO:
  // Implement ray - sphere intersection.
  // Note again that you might want to use the the Sphere::test helper here.
  // When an intersection takes place, the Intersection data should be updated
  // correspondingly.

  double t1,t2;
  bool intersect = test(r,t1,t2);
  if(t1 < 0 || t2 < 0)
    return false;

  Vector3D poi = r.at_time(t1);
  i->t = t1;
  i->primitive = this;
  i->n = poi - o;
  i->n.normalize();
  i->bsdf = get_bsdf();
  return true;

}

void Sphere::draw(const Color& c) const {
  Misc::draw_sphere_opengl(o, r, c);
}

void Sphere::drawOutline(const Color& c) const {
    //Misc::draw_sphere_opengl(o, r, c);
}


} // namespace StaticScene
} // namespace CMU462
