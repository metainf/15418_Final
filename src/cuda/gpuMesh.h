#ifndef GPU_MESH_H
#define GPU_MESH_H

class gpuMesh {
  public:

    /*
     * Constructor using list of primitives
     */
    __host__
    gpuMesh(const gpuTriangle* prim, Vector3D *pos, Vector3D *norm) : 
      primitives(prim), positions(pos), normals(norm) {}


    const gpuTriangle* primitives;
    gpuVector3D *positions;
    gpuVector3D *normals;

}

#endif
