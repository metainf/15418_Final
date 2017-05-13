#include "gpuVector3D.cu"
class gpuMatrix3x3 {

  public:

  // The default constructor.
  __device__ __host__
  gpuMatrix3x3(void) { }

  // Constructor for row major form data.
  // Transposes to the internal column major form.
  // REQUIRES: data should be of size 9 for a 3 by 3 matrix..
  __device__ __host__
  gpuMatrix3x3(float * data)
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
  void zero(float val = 0.0 );

  /**
   * Returns the determinant of A.
   */
  __device__
  float det( void ) const;

  /**
   * Returns the Frobenius norm of A.
   */
  __device__
  float norm( void ) const;

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
        float& operator()( int i, int j );
  __device__ __host__
  const float& operator()( int i, int j ) const;

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
  gpuMatrix3x3 operator*( float c ) const;

  // returns A*B
  __device__
  gpuMatrix3x3 operator*( const gpuMatrix3x3& B ) const;

  // returns A*x
  __device__
  gpuVector3D operator*( const gpuVector3D& x ) const;

  // divides each element by x
  __device__
  void operator/=( float x );

  protected:

  // column vectors
  gpuVector3D entries[3];

}; // class gpuMatrix3x3

// returns the outer product of u and v
__device__
gpuMatrix3x3 outer( const gpuVector3D& u, const gpuVector3D& v );

// returns c*A
__device__
gpuMatrix3x3 operator*( float c, const gpuMatrix3x3& A );
__device__ __host__ float& gpuMatrix3x3::operator()( int i, int j ) {
  return entries[j][i];
}

__device__ __host__ const float& gpuMatrix3x3::operator()( int i, int j ) const {
  return entries[j][i];
}

__device__ __host__ gpuVector3D& gpuMatrix3x3::operator[]( int j ) {
    return entries[j];
}

__device__ __host__ const gpuVector3D& gpuMatrix3x3::operator[]( int j ) const {
  return entries[j];
}

__device__ void gpuMatrix3x3::zero( float val ) {
  // sets all elements to val
  entries[0] = entries[1] = entries[2] = gpuVector3D( val, val, val );
}

__device__ float gpuMatrix3x3::det( void ) const {
  const gpuMatrix3x3& A( *this );

  return -A(0,2)*A(1,1)*A(2,0) + A(0,1)*A(1,2)*A(2,0) +
          A(0,2)*A(1,0)*A(2,1) - A(0,0)*A(1,2)*A(2,1) -
          A(0,1)*A(1,0)*A(2,2) + A(0,0)*A(1,1)*A(2,2) ;
}

__device__ float gpuMatrix3x3::norm( void ) const {
  return sqrt( entries[0].norm2() +
               entries[1].norm2() +
               entries[2].norm2() );
}

__device__ gpuMatrix3x3 gpuMatrix3x3::operator-( void ) const {

 // returns -A
  const gpuMatrix3x3& A( *this );
  gpuMatrix3x3 B;

  B(0,0) = -A(0,0); B(0,1) = -A(0,1); B(0,2) = -A(0,2);
  B(1,0) = -A(1,0); B(1,1) = -A(1,1); B(1,2) = -A(1,2);
  B(2,0) = -A(2,0); B(2,1) = -A(2,1); B(2,2) = -A(2,2);

  return B;
}

__device__ void gpuMatrix3x3::operator+=( const gpuMatrix3x3& B ) {

  gpuMatrix3x3& A( *this );
  float* Aij = (float*) &A;
  const float* Bij = (const float*) &B;

  *Aij++ += *Bij++;
  *Aij++ += *Bij++;
  *Aij++ += *Bij++;
  *Aij++ += *Bij++;
  *Aij++ += *Bij++;
  *Aij++ += *Bij++;
  *Aij++ += *Bij++;
  *Aij++ += *Bij++;
  *Aij++ += *Bij++;
}

__device__ gpuMatrix3x3 gpuMatrix3x3::operator-( const gpuMatrix3x3& B ) const {
  const gpuMatrix3x3& A( *this );
  gpuMatrix3x3 C;

  for( int i = 0; i < 3; i++ )
  for( int j = 0; j < 3; j++ )
  {
     C(i,j) = A(i,j) - B(i,j);
  }

  return C;
}

__device__ gpuMatrix3x3 gpuMatrix3x3::operator*( float c ) const {
  const gpuMatrix3x3& A( *this );
  gpuMatrix3x3 B;

  for( int i = 0; i < 3; i++ )
  for( int j = 0; j < 3; j++ )
  {
     B(i,j) = c*A(i,j);
  }

  return B;
}

__device__ gpuMatrix3x3 operator*( float c, const gpuMatrix3x3& A ) {

  gpuMatrix3x3 cA;
  const float* Aij = (const float*) &A;
  float* cAij = (float*) &cA;

  *cAij++ = c * (*Aij++);
  *cAij++ = c * (*Aij++);
  *cAij++ = c * (*Aij++);
  *cAij++ = c * (*Aij++);
  *cAij++ = c * (*Aij++);
  *cAij++ = c * (*Aij++);
  *cAij++ = c * (*Aij++);
  *cAij++ = c * (*Aij++);
  *cAij++ = c * (*Aij++);

  return cA;
}

__device__ gpuMatrix3x3 gpuMatrix3x3::operator*( const gpuMatrix3x3& B ) const {
  const gpuMatrix3x3& A( *this );
  gpuMatrix3x3 C;

  for( int i = 0; i < 3; i++ )
  for( int j = 0; j < 3; j++ )
  {
     C(i,j) = 0.;

     for( int k = 0; k < 3; k++ )
     {
        C(i,j) += A(i,k)*B(k,j);
     }
  }

  return C;
}

__device__ gpuVector3D gpuMatrix3x3::operator*( const gpuVector3D& x ) const {
  return x[0]*entries[0] +
         x[1]*entries[1] +
         x[2]*entries[2] ;
}

__device__ gpuMatrix3x3 gpuMatrix3x3::T( void ) const {
  const gpuMatrix3x3& A( *this );
  gpuMatrix3x3 B;

  for( int i = 0; i < 3; i++ )
  for( int j = 0; j < 3; j++ )
  {
     B(i,j) = A(j,i);
  }

  return B;
}

__device__ gpuMatrix3x3 gpuMatrix3x3::inv( void ) const {
  const gpuMatrix3x3& A( *this );
  gpuMatrix3x3 B;

  B(0,0) = -A(1,2)*A(2,1) + A(1,1)*A(2,2); B(0,1) =  A(0,2)*A(2,1) - A(0,1)*A(2,2); B(0,2) = -A(0,2)*A(1,1) + A(0,1)*A(1,2);
  B(1,0) =  A(1,2)*A(2,0) - A(1,0)*A(2,2); B(1,1) = -A(0,2)*A(2,0) + A(0,0)*A(2,2); B(1,2) =  A(0,2)*A(1,0) - A(0,0)*A(1,2);
  B(2,0) = -A(1,1)*A(2,0) + A(1,0)*A(2,1); B(2,1) =  A(0,1)*A(2,0) - A(0,0)*A(2,1); B(2,2) = -A(0,1)*A(1,0) + A(0,0)*A(1,1);

  B /= det();

  return B;
}

__device__ void gpuMatrix3x3::operator/=( float x ) {
  gpuMatrix3x3& A( *this );
  float rx = 1./x;

  for( int i = 0; i < 3; i++ )
  for( int j = 0; j < 3; j++ )
  {
     A( i, j ) *= rx;
  }
}

__device__ gpuMatrix3x3 gpuMatrix3x3::identity( void ) {
  gpuMatrix3x3 B;

  B(0,0) = 1.; B(0,1) = 0.; B(0,2) = 0.;
  B(1,0) = 0.; B(1,1) = 1.; B(1,2) = 0.;
  B(2,0) = 0.; B(2,1) = 0.; B(2,2) = 1.;

  return B;
}

__device__ gpuMatrix3x3 gpuMatrix3x3::crossProduct( const gpuVector3D& u ) {
  gpuMatrix3x3 B;

  B(0,0) =   0.;  B(0,1) = -u.z;  B(0,2) =  u.y;
  B(1,0) =  u.z;  B(1,1) =   0.;  B(1,2) = -u.x;
  B(2,0) = -u.y;  B(2,1) =  u.x;  B(2,2) =   0.;

  return B;
}

__device__ gpuMatrix3x3 outer( const gpuVector3D& u, const gpuVector3D& v ) {
  gpuMatrix3x3 B;
  float* Bij = (float*) &B;

  *Bij++ = u.x*v.x;
  *Bij++ = u.y*v.x;
  *Bij++ = u.z*v.x;
  *Bij++ = u.x*v.y;
  *Bij++ = u.y*v.y;
  *Bij++ = u.z*v.y;
  *Bij++ = u.x*v.z;
  *Bij++ = u.y*v.z;
  *Bij++ = u.z*v.z;

  return B;
}

__device__ gpuVector3D& gpuMatrix3x3::column( int i ) {
  return entries[i];
}

__device__ const gpuVector3D& gpuMatrix3x3::column( int i ) const {
  return entries[i];
}
