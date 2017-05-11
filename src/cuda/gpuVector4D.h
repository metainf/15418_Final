#ifndef GPU_VECTOR4D_H
#define GPU_VECTOR4D_H

#include "gpuVector3D.h"

/**
 * Defines 4D standard vectors.
 */
class gpuVector4D {
  public:

  // components
  double x, y, z, w;

  /**
   * Constructor.
   * Initializes tp vector (0,0,0, 0).
   */
  __device__ __host__
  gpuVector4D() : x( 0.0 ), y( 0.0 ), z( 0.0 ), w( 0.0 ) { }

  /**
   * Constructor.
   * Initializes to vector (x,y,z,w).
   */
  __device__ __host__
  gpuVector4D( double x, double y, double z, double w) : x( x ), y( y ), z( z ), w( w ) { }

  /**
   * Constructor.
   * Initializes to vector (x,y,z,0).
   */
  __device__ __host__
  gpuVector4D( double x, double y, double z) : x( x ), y( y ), z( z ), w( 0.0 ) { }


  /**
   * Constructor.
   * Initializes to vector (c,c,c,c)
   */
  __device__ __host__
  gpuVector4D( double c ) : x( c ), y( c ), z( c ), w ( c ) { }

  /**
   * Constructor.
   * Initializes from existing vector4D.
   */
  __device__ __host__
  gpuVector4D( const gpuVector4D& v ) : x( v.x ), y( v.y ), z( v.z ), w( v.w ) { }

  /**
   * Constructor.
   * Initializes from existing vector3D.
   */
  __device__ __host__
  gpuVector4D( const gpuVector3D& v ) : x( v.x ), y( v.y ), z( v.z ), w( 0.0 ) { }

  /**
   * Constructor.
   * Initializes from existing vector3D and w value.
   */
  __device__ __host__
  gpuVector4D( const gpuVector3D& v, double w ) : x( v.x ), y( v.y ), z( v.z ), w( w ) { }

  // returns reference to the specified component (0-based indexing: x, y, z)
  __device__
  inline double& operator[] ( const int& index ) {
    return ( &x )[ index ];
  }

  // returns const reference to the specified component (0-based indexing: x, y, z)
  __device__
  inline const double& operator[] ( const int& index ) const {
    return ( &x )[ index ];
  }

  // negation
  __device__
  inline gpuVector4D operator-( void ) const {
    return gpuVector4D( -x, -y, -z, -w);
  }

  // addition
  __device__
  inline gpuVector4D operator+( const gpuVector4D& v ) const {
    return gpuVector4D( x + v.x, y + v.y, z + v.z, w + v.w);
  }

  // subtraction
  __device__
  inline gpuVector4D operator-( const gpuVector4D& v ) const {
    return gpuVector4D( x - v.x, y - v.y, z - v.z, w - v.w );
  }

  // right scalar multiplication
  __device__
  inline gpuVector4D operator*( const double& c ) const {
    return gpuVector4D( x * c, y * c, z * c, w * c );
  }

  // scalar division
  __device__
  inline gpuVector4D operator/( const double& c ) const {
    const double rc = 1.0/c;
    return gpuVector4D( rc * x, rc * y, rc * z, rc * w );
  }

  // addition / assignment
  __device__
  inline void operator+=( const gpuVector4D& v ) {
    x += v.x; y += v.y; z += v.z; z += v.w;
  }

  // subtraction / assignment
  __device__
  inline void operator-=( const gpuVector4D& v ) {
    x -= v.x; y -= v.y; z -= v.z; w -= v.w;
  }

  // scalar multiplication / assignment
  __device__
  inline void operator*=( const double& c ) {
    x *= c; y *= c; z *= c; w *= c;
  }

  // scalar division / assignment
  __device__
  inline void operator/=( const double& c ) {
    (*this) *= ( 1./c );
  }

  /**
   * Returns Euclidean distance metric extended to 4 dimensions.
   */
  __device__
  inline double norm( void ) const {
    return sqrt( x*x + y*y + z*z + w*w );
  }

  /**
   * Returns Euclidean length squared.
   */
  __device__
  inline double norm2( void ) const {
    return x*x + y*y + z*z + w*w;
  }

  /**
   * Returns unit vector. (returns the normalized copy of this vector.)
   */
  __device__
  inline gpuVector4D unit( void ) const {
    double rNorm = 1. / sqrt( x*x + y*y + z*z + w*w);
    return gpuVector4D( rNorm*x, rNorm*y, rNorm*z );
  }

  /**
   * Divides by Euclidean length.
   * This vector will be of unit length i.e. "normalized" afterwards.
   */
  __device__
  inline void normalize( void ) {
    (*this) /= norm();
  }

  /**
   * Converts this vector to a 3D vector ignoring the w component.
   */
  __device__
  inline gpuVector3D to3D( void ) { return gpuVector3D(x, y, z); }

  
  /**
   * Converts this vector to a 3D vector by dividing x, y, and z by w.
   */
  __device__
  inline gpuVector3D projectTo3D( void ) {
    double invW = 1.0 / w;
    return gpuVector3D(x * invW, y * invW, z * invW);
  }

}; // class gpuVector4D

// left scalar multiplication
__device__
inline gpuVector4D operator* ( const double& c, const gpuVector4D& v ) {
  return gpuVector4D( c * v.x, c * v.y, c * v.z, c*v.w );
}

// dot product (a.k.a. inner or scalar product)
__device__
inline double dot( const gpuVector4D& u, const gpuVector4D& v ) {
  return u.x*v.x + u.y*v.y + u.z*v.z + u.w*v.w;;
}


#endif // CMU462_VECTOR4D_H
