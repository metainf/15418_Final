#ifndef VECTOR3G_H
#define VECTOR3G_H

class gpuVector3D {
  public:
    double x,y,z;

    __device__ __host__ gpuVector3D(): x(0.0), y(0.0), z(0.0){}

    __device__ __host__ 
      gpuVector3D(double x, double y, double z): x(x), y(y), z(z){}

    __device__ __host__ gpuVector3D(double c): x(c), y(c), z(c){}

    // returns reference to the specified component (0-based indexing: x, y, z)
    __device__ __host__ inline double& operator[] ( const int& index ) {
      return ( &x )[ index ];
    }

    // returns const reference to the specified component (0-based indexing: x, y, z)
    __device__ __host__ inline const double& operator[] ( const int& index ) const {
      return ( &x )[ index ];
    }
    __device__ inline bool operator==( const gpuVector3D& v) const {
      return v.x == x && v.y == y && v.z == z;
    }

    // negation
    __device__ inline gpuVector3D operator-( void ) const {
      return gpuVector3D( -x, -y, -z );
    }

    // addition
    __device__ inline gpuVector3D operator+( const gpuVector3D& v ) const {
      return gpuVector3D( x + v.x, y + v.y, z + v.z );
    }

    // subtraction
    __device__ inline gpuVector3D operator-( const gpuVector3D& v ) const {
      return gpuVector3D( x - v.x, y - v.y, z - v.z );
    }

    // right scalar multiplication
    __device__ inline gpuVector3D operator*( const double& c ) const {
      return gpuVector3D( x * c, y * c, z * c );
    }

    // scalar division
    __device__ inline gpuVector3D operator/( const double& c ) const {
      const double rc = 1.0/c;
      return gpuVector3D( rc * x, rc * y, rc * z );
    }

    // addition / assignment
    __device__ inline void operator+=( const gpuVector3D& v ) {
      x += v.x; y += v.y; z += v.z;
    }

    // subtraction / assignment
    __device__ inline void operator-=( const gpuVector3D& v ) {
      x -= v.x; y -= v.y; z -= v.z;
    }

    // scalar multiplication / assignment
    __device__ inline void operator*=( const double& c ) {
      x *= c; y *= c; z *= c;
    }

    // scalar division / assignment
    __device__ inline void operator/=( const double& c ) {
      (*this) *= ( 1./c );
    }

    /**
     * Returns Euclidean length.
     */
    __device__ inline double norm( void ) const {
      return norm3d( x*x, y*y, z*z );
    }

    /**
     * Returns Euclidean length squared.
     */
    __device__ inline double norm2( void ) const {
      return x*x + y*y + z*z;
    }

    /**
     * Returns unit vector.
     */
    __device__ inline gpuVector3D unit( void ) const {
      double rNorm = 1. / norm3d( x*x, y*y, z*z );
      return gpuVector3D( rNorm*x, rNorm*y, rNorm*z );
    }

    /**
     * Divides by Euclidean length.
     */
    __device__ inline void normalize( void ) {
      (*this) /= norm();
    }

}; // class gpuVector3D

// left scalar multiplication
__device__ inline gpuVector3D operator* ( const double& c, const gpuVector3D& v ) {
  return gpuVector3D( c * v.x, c * v.y, c * v.z );
}

// dot product (a.k.a. inner or scalar product)
__device__ inline double dot( const gpuVector3D& u, const gpuVector3D& v ) {
  return u.x*v.x + u.y*v.y + u.z*v.z ;
}

// cross product
__device__ inline gpuVector3D cross( const gpuVector3D& u, const gpuVector3D& v ) {
  return gpuVector3D( u.y*v.z - u.z*v.y,
      u.z*v.x - u.x*v.z,
      u.x*v.y - u.y*v.x );
}
#endif
