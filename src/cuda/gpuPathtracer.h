#ifndef gpu_pathtracer_h
#define gpu_pathtracer_h

#include "../pathtracer.h"
#include "gpuRay.h"
#include "gpuMesh.h"
#include "gpuTriangle.h"

using namespace CMU462;

class gpuPathTracer{
  public:
    gpuPathTracer(PathTracer *__pathtracer);

    ~gpuPathTracer();

    /*
     * Loads a scene from the CPU to the GPU
     */
    void load_scene();

    /*
     * Loads a camera from the CPU to the GPU
     */
    void load_camera();

    /*
     * sets the frame size
     */
    void set_frame_size(size_t width, size_t height);

    /*
     * passes the result from the GPU to the CPU, and then renders the image
     */
    void update_screen();

    /*
     * starts the ray tracing kernel
     */
    void start_raytrace();

    /*
     * Passes the rendered result to the CPU, and then saves it to the png
     * void save_image();
     */

  private:
    
    void build_accel();

    __device__ bool trace_ray(const gpuRay& ray);

    __device__ bool raytrace_pixel(size_t x, size_t y);


    // Components //
    PathTracer * pathtracer;
    //gpuCamera* camera;
    bool* imagePixels;
    size_t w;
    size_t h;
    gpuMesh* mesh;
    gpuTriangle* primitives;
};

    

    


#endif
