#ifndef CMU462_STATICSCENE_PRIMITIVE_H
#define CMU462_STATICSCENE_PRIMITIVE_H

#include "../intersection.h"
#include "../bbox.h"

namespace CMU462 { namespace StaticScene {

/**
 * The abstract base class primitive is the bridge between geometry processing
 * and the shading subsystem. As such, its interface contains methods related
 * to both.
 */
class Primitive {
 public:

  /**
   * Get the world space bounding box of the primitive.
   * \return world space bounding box of the primitive
   */
  virtual BBox get_bbox() const = 0;

  /**
   * Ray - Primitive intersection.
   * Check if the given ray intersects with the primitive, no intersection
   * information is stored.
   * \param r ray to test intersection with
   * \return true if the given ray intersects with the primitive,
             false otherwise
   */
  virtual bool intersect(const Ray& r) const = 0;

  /**
   * Ray - Primitive intersection 2.
   * Check if the given ray intersects with the primitive, if so, the input
   * intersection data is updated to contain intersection information for the
   * point of intersection.
   * \param r ray to test intersection with
   * \param i address to store intersection info
   * \return true if the given ray intersects with the primitive,
             false otherwise
   */
  virtual bool intersect(const Ray& r, Intersection* i) const = 0;

  /**
   * Get BSDF.
   * Return the BSDF of the surface material of the primitive.
   * Note that the BSDFs are not stored in each primitive but in the
   * SceneObject the primitive belongs to.
   */
  virtual BSDF* get_bsdf() const = 0;

  /**
   * Draw with OpenGL (for visualization)
   * \param c desired highlight color
   */
  virtual void draw(const Color& c) const = 0;

  /**
   * Draw outline with OpenGL (for visualization)
   * \param c desired highlight color
   */
  virtual void drawOutline(const Color& c) const = 0;

  /**
   * Get the center/centroid of the primitive
   */
  virtual Vector3D get_center(){return Vector3D(0,0,0);};

  /**
   * Compares the center/centroid of the two primitives along the x axis
   */

  static bool comp_x(Primitive* i, Primitive* j)
  {
    return i->get_center().x > j->get_center().x;
  }
  
  /**
   * Compares the center/centroid of the two primitives along the y axis
   */

  static bool comp_y(Primitive* i, Primitive* j)
  {
    return i->get_center().y > j->get_center().y;
  }

  /**
   * Compares the center/centroid of the two primitives along the x axis
   */

  static bool comp_z(Primitive* i, Primitive* j)
  {
    return i->get_center().z > j->get_center().z;
  }
};

} // namespace StaticScene
} // namespace CMU462

#endif //CMU462_STATICSCENE_PRIMITIVE_H
