#ifndef CMU462_MATRIX4X4_H
#define CMU462_MATRIX4X4_H

#include "gpuVector4D.h"

#include <iosfwd>

class gpuMatrix4x4 {

  public:


    // The default constructor.
    __device__ __host__ gpuMatrix4x4(void) { }

    // Constructor for row major form data.
    // Transposes to the internal column major form.
    // REQUIRES: data should be of size 16.
    __device__ __host__ gpuMatrix4x4(double * data)
    {
      for( int i = 0; i < 4; i++ )
        for( int j = 0; j < 4; j++ )
        {
          // Transpostion happens within the () query.
          (*this)(i,j) = data[i*4 + j];
        }

    }


    /**
     * Sets all elements to val.
     */
    __device__ void zero(double val = 0.0);

    /**
     * Returns the determinant of A.
     */
    __device__ double det( void ) const;

    /**
     * Returns the Frobenius norm of A.
     */
    __device__ double norm( void ) const;

    /**
     * Returns a fresh 4x4 identity matrix.
     */
    __device__ static gpuMatrix4x4 identity( void );

    // No Cross products for 4 by 4 matrix.

    /**
     * Returns the ith column.
     */
    __device__     gpuVector4D& column( int i );
    __device__ const gpuVector4D& column( int i ) const;

    /**
     * Returns the transpose of A.
     */
    __device__ gpuMatrix4x4 T( void ) const;

    /**
     * Returns the inverse of A.
     */
    __device__ gpuMatrix4x4 inv( void ) const;

    // accesses element (i,j) of A using 0-based indexing
    // where (i, j) is (row, column).
    __device__ __host__      double& operator()( int i, int j );
    __device__ __host__ const double& operator()( int i, int j ) const;

    // accesses the ith column of A
    __device__ __host__    gpuVector4D& operator[]( int i );
    __device__ __host__ const gpuVector4D& operator[]( int i ) const;

    // increments by B
    __device__ void operator+=( const gpuMatrix4x4& B );

    // returns -A
    __device__ gpuMatrix4x4 operator-( void ) const;

    // returns A-B
    __device__ gpuMatrix4x4 operator-( const gpuMatrix4x4& B ) const;

    // returns c*A
    __device__ gpuMatrix4x4 operator*( double c ) const;

    // returns A*B
    __device__ gpuMatrix4x4 operator*( const gpuMatrix4x4& B ) const;

    __device__ // returns A*x
      gpuVector4D operator*( const gpuVector4D& x ) const;

    // divides each element by x
    __device__ void operator/=( double x );

  protected:

    // 4 by 4 matrices are represented by an array of 4 column vectors.
    gpuVector4D entries[4];

}; // class Matrix3x3

// returns the outer product of u and v.
__device__ gpuMatrix4x4 outer( const gpuVector4D& u, const gpuVector4D& v );

// returns c*A
__device__ gpuMatrix4x4 operator*( double c, const gpuMatrix4x4& A );


#endif // CMU462_MATRIX4X4_H
