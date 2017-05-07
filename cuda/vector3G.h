#ifndef VECTOR3G_H
#define VECTOR3G_H

class Vector3G {
  public:
    double x,y,z;

    __device__ __global__ Vector3G(): x(0.0), y(0.0), z(0.0){}

    __device__ __global__ 
      Vector3G(double x, double y, double z): x(x), y(y), z(z){}

    __device__ __global__ Vector3G(double c): x(c), y(c), z(c){}

    __device__ inline bool operator==( const Vector3G& v) const {
      return v.x == x && v.y == y && v.z == z;
    }

    // negation
    __device__ inline Vector3G operator-( void ) const {
      return Vector3G( -x, -y, -z );
    }

    // addition
    __device__ inline Vector3G operator+( const Vector3G& v ) const {
      return Vector3G( x + v.x, y + v.y, z + v.z );
    }

    // subtraction
    __device__ inline Vector3G operator-( const Vector3G& v ) const {
      return Vector3G( x - v.x, y - v.y, z - v.z );
    }

    // right scalar multiplication
    __device__ inline Vector3G operator*( const double& c ) const {
      return Vector3G( x * c, y * c, z * c );
    }

    // scalar division
    __device__ inline Vector3G operator/( const double& c ) const {
      const double rc = 1.0/c;
      return Vector3G( rc * x, rc * y, rc * z );
    }

    // addition / assignment
    __device__ inline void operator+=( const Vector3G& v ) {
      x += v.x; y += v.y; z += v.z;
    }

    // subtraction / assignment
    __device__ inline void operator-=( const Vector3G& v ) {
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
    __device__ inline Vector3G unit( void ) const {
      double rNorm = 1. / norm3d( x*x, y*y, z*z );
      return Vector3G( rNorm*x, rNorm*y, rNorm*z );
    }

    /**
     * Divides by Euclidean length.
     */
    __device__ inline void normalize( void ) {
      (*this) /= norm();
    }

}; // class Vector3G

// left scalar multiplication
__device__ inline Vector3G operator* ( const double& c, const Vector3G& v ) {
  return Vector3G( c * v.x, c * v.y, c * v.z );
}

// dot product (a.k.a. inner or scalar product)
__device__ inline double dot( const Vector3G& u, const Vector3G& v ) {
  return u.x*v.x + u.y*v.y + u.z*v.z ;
}

// cross product
__device__ inline Vector3G cross( const Vector3G& u, const Vector3G& v ) {
  return Vector3G( u.y*v.z - u.z*v.y,
      u.z*v.x - u.x*v.z,
      u.x*v.y - u.y*v.x );
}
