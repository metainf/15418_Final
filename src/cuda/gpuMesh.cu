#ifndef GPU_MESH_H
#define GPU_MESH_H
#include "gpuVector3D.cu"

class gpuMesh {
  public:

    /*
     * Constructor using list of primitives
     */
    __host__
    gpuMesh(gpuVector3D *pos, gpuVector3D *norm) : 
      positions(pos), normals(norm) {}


    gpuVector3D *positions;
    gpuVector3D *normals;

};

#endif
