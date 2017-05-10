#ifndef GPU_MESH_H
#define GPU_MESH_H

class gpuMesh {
  public:

    /*
     * Constructor using list of primitives
     */
    __device__ __host__
    gpuMesh(const vector<Primitive*> prim, Vector3D *pos, Vector3D *norm) : primitives(prim), positions(pos), normals(norm) {}

    __device__
    vector<Primitive*> get_primitives() const { return primitives; }

    gpuVector3D *positions;
    gpuVector3D *normals;

  private:
    vector<Primitive*> primitives;

}

#endif
