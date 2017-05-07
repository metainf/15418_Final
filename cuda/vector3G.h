#ifndef VECTOR3G_H
#define VECTOR3G_H

class Vector3G {
  public:
    double x,y,z;

    __device__ __global__ Vector3G(): x(0.0), y(0.0), z(0.0){}

    __device__ __global__ 
      Vector3G(double x, double y, double z): x(x), y(y), z(z){}

    __device__ __global__ Vector3G(double c): x(c), y(c), z(c){}

    __device__ inline bool operator==( const Vector3D& v) const {
      return v.x == x && v.y == y && v.z == z;
    }

    // negation
    __device__ inline Vector3D operator-( void ) const {
      return Vector3D( -x, -y, -z );
    }

    // addition
    __device__ inline Vector3D operator+( const Vector3D& v ) const {
      return Vector3D( x + v.x, y + v.y, z + v.z );
    }

    // subtraction
    __device__ inline Vector3D operator-( const Vector3D& v ) const {
      return Vector3D( x - v.x, y - v.y, z - v.z );
    }

    // right scalar multiplication
    __device__ inline Vector3D operator*( const double& c ) const {
      return Vector3D( x * c, y * c, z * c );
    }

    // scalar division
    __device__ inline Vector3D operator/( const double& c ) const {
      const double rc = 1.0/c;
      return Vector3D( rc * x, rc * y, rc * z );
    }

    // addition / assignment
    __device__ inline void operator+=( const Vector3D& v ) {
      x += v.x; y += v.y; z += v.z;
    }

    // subtraction / assignment
    __device__ inline void operator-=( const Vector3D& v ) {
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
      return sqrt( x*x + y*y + z*z );
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
    __device__ inline Vector3D unit( void ) const {
      double rNorm = 1. / sqrt( x*x + y*y + z*z );
      return Vector3D( rNorm*x, rNorm*y, rNorm*z );
    }

    /**
     * Divides by Euclidean length.
     */
    __device__ inline void normalize( void ) {
      (*this) /= norm();
    }

}; // class Vector3G

// left scalar multiplication
__device__ inline Vector3D operator* ( const double& c, const Vector3D& v ) {
  return Vector3D( c * v.x, c * v.y, c * v.z );
}

// dot product (a.k.a. inner or scalar product)
__device__ inline double dot( const Vector3D& u, const Vector3D& v ) {
  return u.x*v.x + u.y*v.y + u.z*v.z ;
}

// cross product
__device__ inline Vector3D cross( const Vector3D& u, const Vector3D& v ) {
  return Vector3D( u.y*v.z - u.z*v.y,
      u.z*v.x - u.x*v.z,
      u.x*v.y - u.y*v.x );
}
