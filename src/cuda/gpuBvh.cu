#include "gpuBvh.h"

#include <stack>

__device__
gpuBBox gpuBVHAccel::get_bbox() const {
  return (tree[0])->bb;
}

__device__
bool gpuBVHAccel::intersect(const gpuRay ray) 
{ 
  bool hit = false;
  //create a stack
 // stack<gpuBVHNode*> nodes;
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
