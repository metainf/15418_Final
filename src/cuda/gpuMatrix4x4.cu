#include "gpuVector4D.cu"

#include <iosfwd>

class gpuMatrix4x4 {

  public:


    // The default constructor.
    __device__ __host__ gpuMatrix4x4(void) { }

    // Constructor for row major form data.
    // Transposes to the internal column major form.
    // REQUIRES: data should be of size 16.
    __device__ __host__ gpuMatrix4x4(float * data)
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
    __device__ void zero(float val = 0.0);

    /**
     * Returns the determinant of A.
     */
    __device__ float det( void ) const;

    /**
     * Returns the Frobenius norm of A.
     */
    __device__ float norm( void ) const;

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
    __device__ __host__      float& operator()( int i, int j );
    __device__ __host__ const float& operator()( int i, int j ) const;

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
    __device__ gpuMatrix4x4 operator*( float c ) const;

    // returns A*B
    __device__ gpuMatrix4x4 operator*( const gpuMatrix4x4& B ) const;

    __device__ // returns A*x
      gpuVector4D operator*( const gpuVector4D& x ) const;

    // divides each element by x
    __device__ void operator/=( float x );

  protected:

    // 4 by 4 matrices are represented by an array of 4 column vectors.
    gpuVector4D entries[4];

}; // class Matrix3x3

// returns the outer product of u and v.
__device__ gpuMatrix4x4 outer( const gpuVector4D& u, const gpuVector4D& v );

// returns c*A
__device__ gpuMatrix4x4 operator*( float c, const gpuMatrix4x4& A );

__device__
void gpuMatrix4x4::zero( float val ) {
  // sets all elements to val
  entries[0] =
    entries[1] =
    entries[2] =
    entries[3] = gpuVector4D( val, val, val, val );
}

__device__
float gpuMatrix4x4::det( void ) const {
  const gpuMatrix4x4& A( *this );

  return
    A(0,3)*A(1,2)*A(2,1)*A(3,0) - A(0,2)*A(1,3)*A(2,1)*A(3,0) -
    A(0,3)*A(1,1)*A(2,2)*A(3,0) + A(0,1)*A(1,3)*A(2,2)*A(3,0) +
    A(0,2)*A(1,1)*A(2,3)*A(3,0) - A(0,1)*A(1,2)*A(2,3)*A(3,0) -
    A(0,3)*A(1,2)*A(2,0)*A(3,1) + A(0,2)*A(1,3)*A(2,0)*A(3,1) +
    A(0,3)*A(1,0)*A(2,2)*A(3,1) - A(0,0)*A(1,3)*A(2,2)*A(3,1) -
    A(0,2)*A(1,0)*A(2,3)*A(3,1) + A(0,0)*A(1,2)*A(2,3)*A(3,1) +
    A(0,3)*A(1,1)*A(2,0)*A(3,2) - A(0,1)*A(1,3)*A(2,0)*A(3,2) -
    A(0,3)*A(1,0)*A(2,1)*A(3,2) + A(0,0)*A(1,3)*A(2,1)*A(3,2) +
    A(0,1)*A(1,0)*A(2,3)*A(3,2) - A(0,0)*A(1,1)*A(2,3)*A(3,2) -
    A(0,2)*A(1,1)*A(2,0)*A(3,3) + A(0,1)*A(1,2)*A(2,0)*A(3,3) +
    A(0,2)*A(1,0)*A(2,1)*A(3,3) - A(0,0)*A(1,2)*A(2,1)*A(3,3) -
    A(0,1)*A(1,0)*A(2,2)*A(3,3) + A(0,0)*A(1,1)*A(2,2)*A(3,3);

}

__device__
float gpuMatrix4x4::norm( void ) const {
  return sqrt( entries[0].norm2() +
      entries[1].norm2() +
      entries[2].norm2() +
      entries[3].norm2());
}

__device__
gpuMatrix4x4 gpuMatrix4x4::operator-( void ) const {

  // returns -A (Negation).
  const gpuMatrix4x4& A( *this );
  gpuMatrix4x4 B;

  B(0,0) = -A(0,0); B(0,1) = -A(0,1); B(0,2) = -A(0,2); B(0,3) = -A(0,3);
  B(1,0) = -A(1,0); B(1,1) = -A(1,1); B(1,2) = -A(1,2); B(1,3) = -A(1,3);
  B(2,0) = -A(2,0); B(2,1) = -A(2,1); B(2,2) = -A(2,2); B(2,3) = -A(2,3);
  B(3,0) = -A(3,0); B(3,1) = -A(3,1); B(3,2) = -A(3,2); B(3,3) = -A(3,3);

  return B;
}

__device__
void gpuMatrix4x4::operator+=( const gpuMatrix4x4& B ) {

  gpuMatrix4x4& A( *this );
  float* Aij = (float*) &A;
  const float* Bij = (const float*) &B;

  // Add the 16 contigous vector packed float values.
  *Aij++ += *Bij++;//0
  *Aij++ += *Bij++;
  *Aij++ += *Bij++;
  *Aij++ += *Bij++;
  *Aij++ += *Bij++;//4
  *Aij++ += *Bij++;
  *Aij++ += *Bij++;
  *Aij++ += *Bij++;
  *Aij++ += *Bij++;//8
  *Aij++ += *Bij++;
  *Aij++ += *Bij++;
  *Aij++ += *Bij++;
  *Aij++ += *Bij++;//12
  *Aij++ += *Bij++;
  *Aij++ += *Bij++;
  *Aij++ += *Bij++;//15
  //16.

}

__device__
gpuMatrix4x4 gpuMatrix4x4::operator-( const gpuMatrix4x4& B ) const {
  const gpuMatrix4x4& A( *this );
  gpuMatrix4x4 C;

  for( int i = 0; i < 4; i++ )
    for( int j = 0; j < 4; j++ )
    {
      C(i,j) = A(i,j) - B(i,j);
    }

  return C;
}

__device__
gpuMatrix4x4 gpuMatrix4x4::operator*( float c ) const {
  const gpuMatrix4x4& A( *this );
  gpuMatrix4x4 B;

  for( int i = 0; i < 4; i++ )
    for( int j = 0; j < 4; j++ )
    {
      B(i,j) = c*A(i,j);
    }

  return B;
}

// Returns c*A.
__device__
gpuMatrix4x4 operator*( float c, const gpuMatrix4x4& A ) {

  gpuMatrix4x4 cA;
  const float* Aij = (const float*) &A;
  float* cAij = (float*) &cA;

  *cAij++ = c * (*Aij++);//0
  *cAij++ = c * (*Aij++);
  *cAij++ = c * (*Aij++);
  *cAij++ = c * (*Aij++);
  *cAij++ = c * (*Aij++);//4
  *cAij++ = c * (*Aij++);
  *cAij++ = c * (*Aij++);
  *cAij++ = c * (*Aij++);
  *cAij++ = c * (*Aij++);//8
  *cAij++ = c * (*Aij++);
  *cAij++ = c * (*Aij++);
  *cAij++ = c * (*Aij++);
  *cAij++ = c * (*Aij++);//12
  *cAij++ = c * (*Aij++);
  *cAij++ = c * (*Aij++);
  *cAij++ = c * (*Aij++);//15
  //16
  return cA;
}

// Tradiational Grade School Multiplication. N^3
__device__
gpuMatrix4x4 gpuMatrix4x4::operator*( const gpuMatrix4x4& B ) const {
  const gpuMatrix4x4& A( *this );
  gpuMatrix4x4 C;

  for( int i = 0; i < 4; i++ )
    for( int j = 0; j < 4; j++ )
    {
      C(i,j) = 0.;

      for( int k = 0; k < 4; k++ )
      {
        C(i,j) += A(i,k)*B(k,j);
      }
    }

  return C;
}


__device__
gpuVector4D gpuMatrix4x4::operator*( const gpuVector4D& x ) const {
  return x[0]*entries[0] + // Add up products for each matrix column.
    x[1]*entries[1] +
    x[2]*entries[2] +
    x[3]*entries[3];
}

// Naive Transposition.
__device__
gpuMatrix4x4 gpuMatrix4x4::T( void ) const {
  const gpuMatrix4x4& A( *this );
  gpuMatrix4x4 B;

  for( int i = 0; i < 4; i++ )
    for( int j = 0; j < 4; j++ )
    {
      B(i,j) = A(j,i);
    }

  return B;
}

__device__
gpuMatrix4x4 gpuMatrix4x4::inv( void ) const {
  const gpuMatrix4x4& A( *this );
  gpuMatrix4x4 B;

  // Hardcoded in Fully Symbolic computation.

  B(0,0) = A(1,2)*A(2,3)*A(3,1) - A(1,3)*A(2,2)*A(3,1) + A(1,3)*A(2,1)*A(3,2) - A(1,1)*A(2,3)*A(3,2) - A(1,2)*A(2,1)*A(3,3) + A(1,1)*A(2,2)*A(3,3);
  B(0,1) = A(0,3)*A(2,2)*A(3,1) - A(0,2)*A(2,3)*A(3,1) - A(0,3)*A(2,1)*A(3,2) + A(0,1)*A(2,3)*A(3,2) + A(0,2)*A(2,1)*A(3,3) - A(0,1)*A(2,2)*A(3,3);
  B(0,2) = A(0,2)*A(1,3)*A(3,1) - A(0,3)*A(1,2)*A(3,1) + A(0,3)*A(1,1)*A(3,2) - A(0,1)*A(1,3)*A(3,2) - A(0,2)*A(1,1)*A(3,3) + A(0,1)*A(1,2)*A(3,3);
  B(0,3) = A(0,3)*A(1,2)*A(2,1) - A(0,2)*A(1,3)*A(2,1) - A(0,3)*A(1,1)*A(2,2) + A(0,1)*A(1,3)*A(2,2) + A(0,2)*A(1,1)*A(2,3) - A(0,1)*A(1,2)*A(2,3);
  B(1,0) = A(1,3)*A(2,2)*A(3,0) - A(1,2)*A(2,3)*A(3,0) - A(1,3)*A(2,0)*A(3,2) + A(1,0)*A(2,3)*A(3,2) + A(1,2)*A(2,0)*A(3,3) - A(1,0)*A(2,2)*A(3,3);
  B(1,1) = A(0,2)*A(2,3)*A(3,0) - A(0,3)*A(2,2)*A(3,0) + A(0,3)*A(2,0)*A(3,2) - A(0,0)*A(2,3)*A(3,2) - A(0,2)*A(2,0)*A(3,3) + A(0,0)*A(2,2)*A(3,3);
  B(1,2) = A(0,3)*A(1,2)*A(3,0) - A(0,2)*A(1,3)*A(3,0) - A(0,3)*A(1,0)*A(3,2) + A(0,0)*A(1,3)*A(3,2) + A(0,2)*A(1,0)*A(3,3) - A(0,0)*A(1,2)*A(3,3);
  B(1,3) = A(0,2)*A(1,3)*A(2,0) - A(0,3)*A(1,2)*A(2,0) + A(0,3)*A(1,0)*A(2,2) - A(0,0)*A(1,3)*A(2,2) - A(0,2)*A(1,0)*A(2,3) + A(0,0)*A(1,2)*A(2,3);
  B(2,0) = A(1,1)*A(2,3)*A(3,0) - A(1,3)*A(2,1)*A(3,0) + A(1,3)*A(2,0)*A(3,1) - A(1,0)*A(2,3)*A(3,1) - A(1,1)*A(2,0)*A(3,3) + A(1,0)*A(2,1)*A(3,3);
  B(2,1) = A(0,3)*A(2,1)*A(3,0) - A(0,1)*A(2,3)*A(3,0) - A(0,3)*A(2,0)*A(3,1) + A(0,0)*A(2,3)*A(3,1) + A(0,1)*A(2,0)*A(3,3) - A(0,0)*A(2,1)*A(3,3);
  B(2,2) = A(0,1)*A(1,3)*A(3,0) - A(0,3)*A(1,1)*A(3,0) + A(0,3)*A(1,0)*A(3,1) - A(0,0)*A(1,3)*A(3,1) - A(0,1)*A(1,0)*A(3,3) + A(0,0)*A(1,1)*A(3,3);
  B(2,3) = A(0,3)*A(1,1)*A(2,0) - A(0,1)*A(1,3)*A(2,0) - A(0,3)*A(1,0)*A(2,1) + A(0,0)*A(1,3)*A(2,1) + A(0,1)*A(1,0)*A(2,3) - A(0,0)*A(1,1)*A(2,3);
  B(3,0) = A(1,2)*A(2,1)*A(3,0) - A(1,1)*A(2,2)*A(3,0) - A(1,2)*A(2,0)*A(3,1) + A(1,0)*A(2,2)*A(3,1) + A(1,1)*A(2,0)*A(3,2) - A(1,0)*A(2,1)*A(3,2);
  B(3,1) = A(0,1)*A(2,2)*A(3,0) - A(0,2)*A(2,1)*A(3,0) + A(0,2)*A(2,0)*A(3,1) - A(0,0)*A(2,2)*A(3,1) - A(0,1)*A(2,0)*A(3,2) + A(0,0)*A(2,1)*A(3,2);
  B(3,2) = A(0,2)*A(1,1)*A(3,0) - A(0,1)*A(1,2)*A(3,0) - A(0,2)*A(1,0)*A(3,1) + A(0,0)*A(1,2)*A(3,1) + A(0,1)*A(1,0)*A(3,2) - A(0,0)*A(1,1)*A(3,2);
  B(3,3) = A(0,1)*A(1,2)*A(2,0) - A(0,2)*A(1,1)*A(2,0) + A(0,2)*A(1,0)*A(2,1) - A(0,0)*A(1,2)*A(2,1) - A(0,1)*A(1,0)*A(2,2) + A(0,0)*A(1,1)*A(2,2);

  // Invertable iff the determinant is not equal to zero.
  B /= det();

  return B;
}

__device__
void gpuMatrix4x4::operator/=( float x ) {
  gpuMatrix4x4& A( *this );
  float rx = 1./x;

  for( int i = 0; i < 4; i++ )
    for( int j = 0; j < 4; j++ )
    {
      A( i, j ) *= rx;
    }
}

__device__
gpuMatrix4x4 gpuMatrix4x4::identity( void ) {
  gpuMatrix4x4 B;

  B(0,0) = 1.; B(0,1) = 0.; B(0,2) = 0.; B(0,3) = 0.;
  B(1,0) = 0.; B(1,1) = 1.; B(1,2) = 0.; B(1,3) = 0.;
  B(2,0) = 0.; B(2,1) = 0.; B(2,2) = 1.; B(2,3) = 0.;
  B(3,0) = 0.; B(3,1) = 0.; B(3,2) = 0.; B(3,3) = 1.;

  return B;
}

__device__
gpuMatrix4x4 outer( const gpuVector4D& u, const gpuVector4D& v ) {
  gpuMatrix4x4 B;

  // Opposite of an inner product.
  for( int i = 0; i < 4; i++ )
    for( int j = 0; j < 4; j++ )
    {
      B( i, j ) = u[i]*v[j];
    }

  return B;
}
