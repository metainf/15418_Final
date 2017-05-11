#ifndef gpu_pathtracer_h
#define gpu_pathtracer_h

#include "../pathtracer.h"

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
    void load_camera(Camera *cam);

    /*
     * sets the frame size
     */
    void set_frame_size(size_t width, size_t height);

    /*
     * passes the result from the GPU to the CPU, and then renders the image
     */
    void update_screen();

    /*
     * Wrapper for the ray tracing kernel
     */
    void start_raytrace();

    /*
     * Passes the rendered result to the CPU, and then saves it to the png
     * void save_image();
     */

  private:

    // Components //
    PathTracer * pathtracer;
    bool* imagePixels;
    size_t w;
    size_t h;
};

    

    


#endif
