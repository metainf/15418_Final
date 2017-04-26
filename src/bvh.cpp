#include "bvh.h"

#include "CMU462/CMU462.h"
#include "static_scene/triangle.h"

#include <iostream>
#include <stack>
#include <algorithm>

#define DEBUG 0

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
    node_split.pop();

    // Sort the range of the primitives of the node by an axis
    // For now, using only the x axis
    struct p_bucket *bucket = new struct p_bucket[BUCKET_NUM];
    
    double min_sah = INF_D;
    BBox partition1;
    BBox partition2;
    int axis = 0;
    int lp; //primitives in left node
    int rp; //primitives in right node

    //x-axis
    sort(primitives.begin() + sn->start, primitives.begin() + sn->start + sn->range, comp_x);

   // double step = ((primitives[sn->start + sn->range - 1])->get_bbox().max.x -
     //              (primitives[sn->start])->get_bbox().min.x)/BUCKET_NUM;
    //double init = (primitives[sn->start])->get_bbox().min.x;
    double step = sn->bb.extent.x / BUCKET_NUM;
    double init = sn->bb.min.x;
    
    Primitive **p = &(*primitives.begin()) + sn->start;


    //calculate bucket information and store it in an array[bucketsize]
    for(size_t j = 0; j < sn->range; j++) {
      int b = ((*(p + j))->get_center().x - init) / step;
      bucket[b].num_prim++;
      bucket[b].bb.expand((*(p+j))->get_bbox());
    }

    //calculate SAH for each partition
    for(size_t j = 1; j < BUCKET_NUM; j++) {
      BBox b1 = BBox();
      int b1_p = 0;
      BBox b2 = BBox();
      int b2_p = 0;
      for(size_t k = 0; k < j; k++) {
        b1.expand(bucket[k].bb);
        b1_p += bucket[k].num_prim;
      }
      for(size_t l = j; l < BUCKET_NUM; l++) {
        b2.expand(bucket[l].bb);
        b2_p += bucket[l].num_prim;
      }
      double sah = b1.surface_area() * b1_p + b2.surface_area() * b2_p;
      if(sah < min_sah){
        partition1 = BBox(b1.min, b1.max);
        partition2 = BBox(b2.min, b2.max);
        lp = b1_p;
        rp = b2_p;
        axis = 0;
        min_sah = sah;
      }
    }

/*
    //y-axis
    sort(primitives.begin() + sn->start, primitives.begin() + sn->start + sn->range, comp_y);
    step = ((primitives[sn->start + sn->range - 1])->get_center().y -
            (primitives[sn->start])->get_center().y)/BUCKET_NUM;
    init = (primitives[sn->start])->get_center().y;
    //calculate bucket information and store it in an array[bucketsize]
    for(size_t j = 0; j < sn->range; j++) {
      int b = ((*(p + j))->get_center().y - init) / step;
      bucket[b].num_prim++;
      bucket[b].bb.expand((*(p+j))->get_bbox());
    }

    //calculate SAH for each partition
    for(size_t j = 1; j < BUCKET_NUM; j++) {
      BBox b1 = BBox();
      int b1_p = 0;
      BBox b2 = BBox();
      int b2_p = 0;
      for(size_t k = 0; k < j; k++) {
        b1.expand(bucket[k].bb);
        b1_p += bucket[k].num_prim;
      }
      for(size_t l = j; l < BUCKET_NUM; l++) {
        b2.expand(bucket[l].bb);
        b2_p += bucket[l].num_prim;
      }
      double sah = b1.surface_area() * b1_p + b2.surface_area() * b2_p;
      if(sah < min_sah){
        partition1 = BBox(b1.min, b1.max);
        partition2 = BBox(b2.min, b2.max);
        lp = b1_p;
        rp = b2_p;
        axis = 1;
      }
    }


    std::cout << "starting\n"; 
    //z-axis 
    sort(primitives.begin() + sn->start, primitives.begin() + sn->start + sn->range, comp_z);
    step = ((primitives[sn->start + sn->range - 1])->get_center().z -
            (primitives[sn->start])->get_center().z)/BUCKET_NUM;
    init = (primitives[sn->start])->get_center().z;
    //calculate bucket iinformation and store it in an array[bucketsize]
    for(size_t j = 0; j < sn->range; j++) {
      int b = ((*(p + j))->get_center().z - init) / step;
      bucket[b].num_prim++;
      bucket[b].bb.expand((*(p+j))->get_bbox());
    }

    //calculate SAH for each partition
    for(size_t j = 1; j < BUCKET_NUM; j++) {
      BBox b1 = BBox();
      int b1_p = 0;
      BBox b2 = BBox();
      int b2_p = 0;
      for(size_t k = 0; k < j; k++) {
        b1.expand(bucket[k].bb);
        b1_p += bucket[k].num_prim;
      }
      for(size_t l = j; l < BUCKET_NUM; l++) {
        b2.expand(bucket[l].bb);
        b2_p += bucket[l].num_prim;
      }
      double sah = b1.surface_area() * b1_p + b2.surface_area() * b2_p;
      if(sah < min_sah){
        partition1 = BBox(b1.min, b1.max);
        partition2 = BBox(b2.min, b2.max);
        lp = b1_p;
        rp = b2_p;
        axis = 2;
      }
    } 
    if(axis == 0) {  
      sort(primitives.begin() + sn->start, primitives.begin() + sn->start + sn->range, comp_x);
    }
    else if(axis == 1) {  
      sort(primitives.begin() + sn->start, primitives.begin() + sn->start + sn->range, comp_y);
    }
    else {  
      sort(primitives.begin() + sn->start, primitives.begin() + sn->start + sn->range, comp_z);
    } */


    if(lp > 10 && lp != sn->range) {
      //    std::cout << "lp: " << lp << "\n";
      BVHNode *left = new BVHNode(partition1, sn->start, lp);
      node_split.push(left);
      sn->l = left;
    }
    if(rp > 10 && rp != sn->range) {
      BVHNode *right = new BVHNode(partition2, sn->start + lp, rp);
      node_split.push(right);
      sn->r = right;
      //    std::cout <<"rp: " << rp << "\n";
    }
    delete[] bucket;
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
