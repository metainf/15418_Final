#include "bvh.h"

#include "CMU462/CMU462.h"
#include "static_scene/triangle.h"

#include <iostream>
#include <stack>
#include <algorithm>

using namespace std;

namespace CMU462 { namespace StaticScene {

BVHAccel::BVHAccel(const std::vector<Primitive *> &_primitives,
                   size_t max_leaf_size) {

  this->primitives = _primitives;

  // TODO:
  // Construct a BVH from the given vector of primitives and maximum leaf
  // size configuration. The starter code build a BVH aggregate with a
  // single leaf node (which is also the root) that encloses all the
  // primitives.

  BBox bb;
  for (size_t i = 0; i < primitives.size(); ++i) {
    bb.expand(primitives[i]->get_bbox());
  }

  root = new BVHNode(bb, 0, primitives.size());

  stack<BVHNode*> node_split;
  node_split.push(root);

  while(!node_split.empty())
  {
    BVHNode* sn = node_split.top();

    // Sort the range of the primitives of the node by an axis
    // For now, using only the x axis
    // 
    bucket = new p_bucket[BUCKET_NUM];
    sort(primitives.begin() + sn->start, primitives.begin() + sn->start + sn->range, comp_x);
    float step = ((primitives.begin() + sn->start + sn->range).x - (primitives.begin() + sn->start).x)/BUCKET_NUM;
    float init = (primitives.begin() + sn->start).x;
    Primitive *p = primitives.begin() + sn->start;
    for(size_t j = 0; j < sn->range; j++)
      int b = ((p + j).x - init) / step;
    }
    //calculate bucket information and store it in an array[bucketsize]
    //calculate the initial partition, 1:Bucketsize - 1 
    for(size_t j = 1; j < BUCKET_NUM; j++) {
      //for each bucket, calculate SAH
      //union partition into left partition and remove from right
    }
    sort(primitives.begin() + sn->start, primitives.begin() + sn->start + sn->range, comp_y);
    sort(primitives.begin() + sn->start, primitives.begin() + sn->start + sn->range, comp_z);

    node_split.pop();
  }
}

BVHAccel::~BVHAccel() {

  // TODO:
  // Implement a proper destructor for your BVH accelerator aggregate

}

BBox BVHAccel::get_bbox() const {
  return root->bb;
}

bool BVHAccel::intersect(const Ray &ray) const {

  // TODO:
  // Implement ray - bvh aggregate intersection test. A ray intersects
  // with a BVH aggregate if and only if it intersects a primitive in
  // the BVH that is not an aggregate.
  bool hit = false;
  for (size_t p = 0; p < primitives.size(); ++p) {
    if(primitives[p]->intersect(ray)) hit = true;
  }

  return hit;

}

bool BVHAccel::intersect(const Ray &ray, Intersection *i) const {

  // TODO:
  // Implement ray - bvh aggregate intersection test. A ray intersects
  // with a BVH aggregate if and only if it intersects a primitive in
  // the BVH that is not an aggregate. When an intersection does happen.
  // You should store the non-aggregate primitive in the intersection data
  // and not the BVH aggregate itself.

  bool hit = false;
  for (size_t p = 0; p < primitives.size(); ++p) {
    if(primitives[p]->intersect(ray, i)) hit = true;
  }

  return hit;

}

}  // namespace StaticScene
}  // namespace CMU462
