#ifndef GPU_BVH_H
#define GPU_BVH_H

#include "../bvh.h"
#include "gpuTriangle.cu"
#include "gpuBBox.cu"

#include <vector>

#define BUCKET_NUM 4

/**
 * A node in the BVH accelerator aggregate.
 * The accelerator uses a "flat tree" structure where all the primitives are
 * stored in one vector. A node in the data structure stores only the starting
 * index and the number of primitives in the node and uses this information to
 * index into the primitive vector for actual data. In this implementation all
 * primitives (index + range) are stored on leaf nodes. A leaf node has no child
 * node and its range should be no greater than the maximum leaf size used when
 * constructing the BVH.
 */
struct gpuBVHNode {

  __device__ __host__
    gpuBVHNode()
    : bb(gpuBBox()), start(0), range(0), l(-1), r(-1) {}

  __device__ __host__
    gpuBVHNode(BVHNode *n)
    : bb(gpuBBox(n->bb)), start(n->start), range(n->range), l(n->l), r(n->r) {}

  __device__
    inline bool isLeaf() const { return l == -1 && r == -1; }

  gpuBBox bb;        ///< bounding box of the node
  size_t start;   ///< start index into the primitive list
  size_t range;   ///< range of index into the primitive list
  int l;     ///< left child node
  int r;     ///< right child node
};

struct gpuP_bucket {

  __device__  gpuP_bucket()
    : bb(gpuBBox()), num_prim(0) {}
  gpuBBox bb;
  size_t num_prim;
};


/* BVH Traversal array stack */
struct gpuStack{

  __device__
    gpuStack() : size(0), pos(0) {}

  __device__ inline void push(gpuBVHNode *node) {
    if (pos == size) {
      return;
    }
    nodes[pos] = node;
    pos++;
  }

  __device__ inline gpuBVHNode *pop() {
    if (pos == 0) {
      return NULL;
    }
    pos--;
    return nodes[pos];
  }

  __device__ inline bool isEmpty() { return pos == 0; }

  size_t size = 64;
  size_t pos;
  gpuBVHNode nodes[64];
};

/**
 * Bounding Volume Hierarchy for fast Ray - Primitive intersection.
 * Note that the BVHAccel is an Aggregate (A Primitive itself) that contains
 * all the primitives it was built from. Therefore once a BVHAccel Aggregate
 * is created, the original input primitives can be ignored from the scene
 * during ray intersection tests as they are contained in the aggregate.
 */
class gpuBVHAccel {
  public:

    __host__
      gpuBVHAccel (vector<BVHNode *> t, gpuTriangle *_primitives, ) {
        tree = new gpuBVHNode *[t.size()];
        this->primitives = _primitives;
        for(int i = 0; i < t.size(); i++) {
          tree[i] = new gpuBVHNode(t[i]);
        }
      }

    /**
     * Parameterized Constructor.
     * Create BVH from a list of primitives. Note that the BVHAccel Aggregate
     * stores pointers to the primitives and thus the primitives need be kept
     * in memory for the aggregate to function properly.
     * \param primitives primitives to build from
     * \param max_leaf_size maximum number of primitives to be stored in leaves
     *
     gpuBVHAccel(const std::vector<gpuTriangle*>& primitives, size_t max_leaf_size = 4);
     */

    /**
     * Destructor.
     * The destructor only destroys the Aggregate itself, the primitives that
     * it contains are left untouched.
     */
    ~gpuBVHAccel();

    /**
     * Get the world space bounding box of the aggregate.
     * \return world space bounding box of the aggregate
     */
    __device__
      gpuBBox get_bbox() const;

    /**
     * Ray - Aggregate intersection.
     * Check if the given ray intersects with the aggregate (any primitive in
     * the aggregate), no intersection information is stored.
     * \param r ray to test intersection with
     * \return true if the given ray intersects with the aggregate,
     false otherwise
     */
    __device__
      bool intersect(const gpuRay& ray) ;

    /**
     * Ray - Aggregate intersection 2.
     * Check if the given ray intersects with the aggregate (any primitive in
     * the aggregate). If so, the input intersection data is updated to contain
     * intersection information for the point of intersection. Note that the
     * intersected primitive entry in the intersection should be updated to
     * the actual primitive in the aggregate that the ray intersected with and
     * not the aggregate itself.
     * \param r ray to test intersection with
     * \param i address to store intersection info
     * \return true if the given ray intersects with the aggregate,
     false otherwise
     */
    bool intersect(const gpuRay& r, Intersection* i) const;

    /**
     * Get entry point (root) - used in visualizer
     */
    __device__
      gpuBVHNode* get_root() const { return tree[0]; }


    /*
     * Returns the left child of the node
     */
    __device__
      gpuBVHNode* get_l_child(gpuBVHNode* node) const {return (node->l < 0) ? NULL: tree[node.l];}

    /*
     * Returns the left child of the node
     */
    __device__
      gpuBVHNode* get_r_child(gpuBVHNode * node) const {return(node->r < 0)? NULL: tree[node.r];}

    /**
     * Draw the BVH with OpenGL - used in visualizer
     */
    void draw(const Color& c) const { }

    /**
     * Draw the BVH outline with OpenGL - used in visualizer
     */
    void drawOutline(const Color& c) const { }

  private:
    gpuBVHNode *tree;
    gpuTriangle *primitives;
};



__device__
gpuBBox gpuBVHAccel::get_bbox() const {
  return (tree[0]).bb;
}

  __device__
bool gpuBVHAccel::intersect(const gpuRay& ray) 
{ 
  bool hit = false;
  //create a stack
  // stack<gpuBVHNode*> nodes;
  gpuStack stack;
  stack.push(get_root());
  while(!stack.isEmpty()) {
    gpuBVHNode *n = stack.pop();
    //if leaf
    gpuBVHNode *left = get_l_child(n);
    gpuBVHNode *right = get_r_child(n);
    if(n->isLeaf()) {
      //check if primitives intersect
      for(size_t p = 0; p < n->range; p++) {
        if(primitives[p + n->start].intersect(ray)) {
          hit = true;
        }
      }
    }
    else {
      if(left == NULL) {
        //check if right node bounding box intersects, add to stack
        double t0 = ray.min_t;
        double t1 = ray.max_t;
        if(right->bb.intersect(ray, t0, t1)) {
          stack.push(right);
        }
      }
      else if(right == NULL) {
        //check if left node intersetcs, add to stack
        double t0 = ray.min_t;
        double t1 = ray.max_t;
        if(left->bb.intersect(ray, t0, t1)) {
          stack.push(left);
        }
      }
      else {
        double t0 = ray.min_t;
        double t1 = ray.max_t;
        double t2 = ray.min_t;
        double t3 = ray.max_t;
        bool leftHit = left->bb.intersect(ray, t0, t1);
        bool rightHit = right->bb.intersect(ray, t2, t3);

        if(leftHit && rightHit) {
          gpuBVHNode *first = (t0 <= t2) ? left : right;
          gpuBVHNode *second = (t0 <= t2) ? right : left;
          stack.push(first);
          stack.push(second);
        }
        else if (leftHit) {
          stack.push(left);
        }
        else if (rightHit) {
          stack.push(right);
        }
      }
    }
  }
  return hit;

}
#endif // GPU_BVH_H
