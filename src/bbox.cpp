#include "bbox.h"

#include "GL/glew.h"

#include <algorithm>
#include <iostream>

#define MIN(X, Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X, Y) ((X) > (Y) ? (X) : (Y))

namespace CMU462 {

bool BBox::intersect(const Ray& r, double& t0, double& t1) const {

  // TODO:
  // Implement ray - bounding box intersection test
  // If the ray intersected the bouding box within the range given by
  // t0, t1, update t0 and t1 with the new intersection times.

  double tx1 = (min.x - r.o.x) / r.d.x;
  double tx2 = (max.x - r.o.x) / r.d.x;
  double txmin = MIN(tx1, tx2);
  double txmax = MAX(tx1, tx2);
  
  double ty1 = (min.y - r.o.y) / r.d.y;
  double ty2 = (max.y - r.o.y) / r.d.y;
  double tymin = MIN(tx1, tx2);
  double tymax = MAX(tx1, tx2);

  double tz1 = (min.z - r.o.z) / r.d.z; 
  double tz2 = (min.z - r.o.z) / r.d.z;
  double tzmin = MIN(tx1, tx2);
  double tzmax = MAX(tx1, tx2);

  double tmin = MAX(MAX(t0, txmin), MAX(tymin, tzmin));
  double tmax = MIN(MIN(t1, txmax), MIN(tymax, tzmax));

  if(tmin <= tmax && tmax >= 0) {
    t1 = tmax;
    t0 = tmin;
    return true;
  }

  return false;
  
}

void BBox::draw(Color c) const {

  glColor4f(c.r, c.g, c.b, c.a);

	// top
	glBegin(GL_LINE_STRIP);
	glVertex3d(max.x, max.y, max.z);
  glVertex3d(max.x, max.y, min.z);
  glVertex3d(min.x, max.y, min.z);
  glVertex3d(min.x, max.y, max.z);
  glVertex3d(max.x, max.y, max.z);
	glEnd();

	// bottom
	glBegin(GL_LINE_STRIP);
  glVertex3d(min.x, min.y, min.z);
  glVertex3d(min.x, min.y, max.z);
  glVertex3d(max.x, min.y, max.z);
  glVertex3d(max.x, min.y, min.z);
  glVertex3d(min.x, min.y, min.z);
	glEnd();

	// side
	glBegin(GL_LINES);
	glVertex3d(max.x, max.y, max.z);
  glVertex3d(max.x, min.y, max.z);
	glVertex3d(max.x, max.y, min.z);
  glVertex3d(max.x, min.y, min.z);
	glVertex3d(min.x, max.y, min.z);
  glVertex3d(min.x, min.y, min.z);
	glVertex3d(min.x, max.y, max.z);
  glVertex3d(min.x, min.y, max.z);
	glEnd();

}

std::ostream& operator<<(std::ostream& os, const BBox& b) {
  return os << "BBOX(" << b.min << ", " << b.max << ")";
}

} // namespace CMU462
