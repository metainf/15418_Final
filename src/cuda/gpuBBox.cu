#ifndef gpu_bbox
#define gpu_bbox

#include "gpuVector3D.cu"
#include "gpuRay.cu"
#include "math_constants.h"

#define MIN(X, Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X, Y) ((X) > (Y) ? (X) : (Y))


struct gpuBBox {

  gpuVector3D max;     ///< min corner of the bounding box
  gpuVector3D min;     ///< max corner of the bounding box
  gpuVector3D extent;  ///< extent of the bounding box (min -> max)

  /**
   * Constructor.
   * The default constructor creates a new bounding box which contains no
   * points.
   */
  __device__ __host__ gpuBBox() {
    max = gpuVector3D(0, 0, 0);
    min = gpuVector3D( 0,  0, 0);
    extent = max - min;
  }

  /**
   * Constructor.
   * Creates a bounding box that includes a single point.
   */
  __device__ __host__
    gpuBBox(const gpuVector3D p) : min(p), max(p) { extent = max - min; }

  /**
   * Constructor.
   * Creates a bounding box with given bounds.
   * \param min the min corner
   * \param max the max corner
   */
  __device__ __host__ 
    gpuBBox(const gpuVector3D min, const gpuVector3D max) :
    min(min), max(max) { extent = max - min; }

  /**
   * Constructor.
   * Creates a bounding box with given bounds (component wise).
   */
  __device__ __host__
    gpuBBox(const double minX, const double minY, const double minZ,
      const double maxX, const double maxY, const double maxZ) {
    min = gpuVector3D(minX, minY, minZ);
    max = gpuVector3D(maxX, maxY, maxZ);
    extent = max - min;
  }

  __device__ __host__
    gpuBBox(BBox bb) {
      min = gpuVector3D(bb.min.x, bb.min.y, bb.min.z);
      max = gpuVector3D(bb.max.x, bb.max.y, bb.max.z);
      extent = max - min;
    }
  
  /**
   * Expand the bounding box to include another (union).
   * If the given bounding box is contained within *this*, nothing happens.
   * Otherwise *this* is expanded to the minimum volume that contains the
   * given input.
   * \param bbox the bounding box to be included
   */
  __device__ void expand(const gpuBBox bbox) {
    min.x = fmin(min.x, bbox.min.x);
    min.y = fmin(min.y, bbox.min.y);
    min.z = fmin(min.z, bbox.min.z);
    max.x = fmax(max.x, bbox.max.x);
    max.y = fmax(max.y, bbox.max.y);
    max.z = fmax(max.z, bbox.max.z);
    extent = max - min;
  }

  /**
   * Expand the bounding box to include a new point in space.
   * If the given point is already inside *this*, nothing happens.
   * Otherwise *this* is expanded to a minimum volume that contains the given
   * point.
   * \param p the point to be included
   */
  __device__ void expand(const gpuVector3D p) {
    min.x = fmin(min.x, p.x);
    min.y = fmin(min.y, p.y);
    min.z = fmin(min.z, p.z);
    max.x = fmax(max.x, p.x);
    max.y = fmax(max.y, p.y);
    max.z = fmax(max.z, p.z);
    extent = max - min;
  }

  __device__ gpuVector3D centroid() const {
    return (min + max) / 2;
  }

  /**
   * Compute the surface area of the bounding box.
   * \return surface area of the bounding box.
   */
  __device__ double surface_area() const {
    if (empty()) return 0.0;
    return 2 * (extent.x * extent.z +
        extent.x * extent.y +
        extent.y * extent.z);
  }

  /**
   * Check if bounding box is empty.
   * Bounding box that has no size is considered empty. Note that since
   * bounding box are used for objects with positive volumes, a bounding
   * box of zero size (empty, or contains a single vertex) are considered
   * empty.
   */
  __device__ bool empty() const {
    return min.x > max.x || min.y > max.y || min.z > max.z;
  }

  /**
   * Ray - bbox intersection.
   * Intersects ray with bounding box, does not store shading information.
   * \param r the ray to intersect with
   * \param t0 lower bound of intersection time
   * \param t1 upper bound of intersection time
   */
 __device__ bool intersect(const gpuRay r, double t0, double t1) const {
  double tx1 = (min.x - r.o.x) / r.d.x;
  double tx2 = (max.x - r.o.x) / r.d.x;
  double txmin = MIN(tx1, tx2);
  double txmax = MAX(tx1, tx2);
  
  double ty1 = (min.y - r.o.y) / r.d.y;
  double ty2 = (max.y - r.o.y) / r.d.y;
  double tymin = MIN(ty1, ty2);
  double tymax = MAX(ty1, ty2);

  double tz1 = (min.z - r.o.z) / r.d.z; 
  double tz2 = (min.z - r.o.z) / r.d.z;
  double tzmin = MIN(tz1, tz2);
  double tzmax = MAX(tz1, tz2);

  double tmin = MAX(MAX(t0, txmin), MAX(tymin, tzmin));
  double tmax = MIN(MIN(t1, txmax), MIN(tymax, tzmax));

  if(tmin <= tmax && tmax >= 0) {
    t1 = tmax;
    t0 = tmin;
    return true;
  }

  return false;
 }

};
#endif
