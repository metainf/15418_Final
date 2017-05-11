#ifndef GPU_MATRIX3X3_H
#define GPU_MATRIX3X3_H

#include "gpuVector3D.h"

class gpuMatrix3x3 {

  public:

  // The default constructor.
  __device__ __host__
  gpuMatrix3x3(void) { }

  // Constructor for row major form data.
  // Transposes to the internal column major form.
  // REQUIRES: data should be of size 9 for a 3 by 3 matrix..
  __device__ __host__
  gpuMatrix3x3(double * data)
  {
    for( int i = 0; i < 3; i++ ) {
      for( int j = 0; j < 3; j++ ) {
	        // Transpostion happens within the () query.
	        (*this)(i,j) = data[i*3 + j];
      }
    }
  }

  /**
   * Sets all elements to val.
   */
  __device__
  void zero(double val = 0.0 );

  /**
   * Returns the determinant of A.
   */
  __device__
  double det( void ) const;

  /**
   * Returns the Frobenius norm of A.
   */
  __device__
  double norm( void ) const;

  /**
   * Returns the 3x3 identity matrix.
   */
  __device__
  static gpuMatrix3x3 identity( void );

  /**
   * Returns a matrix representing the (left) cross product with u.
   */
  __device__
  static gpuMatrix3x3 crossProduct( const gpuVector3D& u );

  /**
   * Returns the ith column.
   */
  __device__
        gpuVector3D& column( int i );
  __device__
  const gpuVector3D& column( int i ) const;

  /**
   * Returns the transpose of A.
   */
  __device__
  gpuMatrix3x3 T( void ) const;

  /**
   * Returns the inverse of A.
   */
  __device__
  gpuMatrix3x3 inv( void ) const;

  // accesses element (i,j) of A using 0-based indexing
  __device__ __host__
        double& operator()( int i, int j );
  __device__ __host__
  const double& operator()( int i, int j ) const;

  // accesses the ith column of A
  __device__ __host__
        gpuVector3D& operator[]( int i );
  __device__ __host__
  const gpuVector3D& operator[]( int i ) const;

  // increments by B
  __device__
  void operator+=( const gpuMatrix3x3& B );

  // returns -A
  __device__
  gpuMatrix3x3 operator-( void ) const;

  // returns A-B
  __device__
  gpuMatrix3x3 operator-( const gpuMatrix3x3& B ) const;

  // returns c*A
  __device__
  gpuMatrix3x3 operator*( double c ) const;

  // returns A*B
  __device__
  gpuMatrix3x3 operator*( const gpuMatrix3x3& B ) const;

  // returns A*x
  __device__
  gpuVector3D operator*( const gpuVector3D& x ) const;

  // divides each element by x
  __device__
  void operator/=( double x );

  protected:

  // column vectors
  gpuVector3D entries[3];

}; // class gpuMatrix3x3

// returns the outer product of u and v
__device__
gpuMatrix3x3 outer( const gpuVector3D& u, const gpuVector3D& v );

// returns c*A
__device__
gpuMatrix3x3 operator*( double c, const gpuMatrix3x3& A );

#endif 
