#include <cstddef>
#include <iostream>
#include <stdio.h>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <sys/time.h>


#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

static void HandleError(cudaError_t err,
    const char* file,
    int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
            file, line);
        exit(EXIT_FAILURE);
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
#define I3D(z,y,x) (x + y*LX + z*LX*LY)
#define d_I3D(z,y,x) (x + d_lx * ( y + d_ly * (z)))
#define IDX(b,z,y,x,k) ( x + LX * ( y + LY * ( z + LZ * ( k + Q * (b)))))  //SOA
#define d_IDX(b,z,y,x,k) ( x + d_lx * ( y + d_ly * ( z + d_lz * ( k + d_Q * (b))))) // SOA
// #define d_IDX(b,z,y,x,k) ( k + d_Q * (x + d_lx * (y + d_ly * ( z + d_lz * b)))) // AOS
#define SQR(x)	((x)*(x))

using namespace std;  

//system size
const int LX = 256;     
const int LY = 64;
const int LZ = 32;

//creating cte on the device side
__device__ __constant__ int d_lx = LX;
__device__ __constant__ int d_ly = LY;
__device__ __constant__ int d_lz = LZ;
__device__ __constant__ float d_tau = 0.51f;        //kinematic viscosity = (tau-0.5)/3

//D3Q19 lattice
const int Q = 19;    
__device__ __constant__ int d_Q = 19;

const int cx[Q] = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0 };
__device__ __constant__ int d_cx[19] = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0 };

const int cy[Q] = { 0, 0, 0, 1,-1, 0, 0, 1, 1,-1,-1, 0, 0, 0, 0, 1,-1, 1,-1 };
__device__ __constant__ int d_cy[19] = { 0, 0, 0, 1,-1, 0, 0, 1, 1,-1,-1, 0, 0, 0, 0, 1,-1, 1,-1 };

const int cz[Q] = { 0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1, 1,-1,-1, 1, 1,-1,-1 };
__device__ __constant__ int d_cz[19] = { 0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1, 1,-1,-1, 1, 1,-1,-1 };

__device__ __constant__  int opposite[19] = { 0, 2, 1 ,4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15 };  //opposite vectors for the bounce-back boundary conditions
const float w0 = 1.0f / 3.0f;     //discrete weights
const float w1 = 1.0f / 18.0f;
const float w2 = 1.0f / 36.0f;
const float weight[19] = { w0, w1, w1, w1, w1, w1, w1, w2, w2, w2, w2, w2, w2, w2, w2, w2, w2, w2, w2 };

__device__ __constant__ float d_weight[19] = { 1.0f / 3.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f,
1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f };

//functions
__host__ float feq(int k, float density, float ux, float uy, float uz);
__host__ void init(float* h_f, float* h_Ux, float* h_Uy, float* h_Uz, float* h_Dens, float* h_Fx, float* h_Fy, float* h_Fz, int current, int buffer);
__host__ void setObst(bool* h_Obst);

__global__ void d_compute_velocity(float* d_Ux, float* d_Uy, float* d_Uz, float* d_Fx, float* d_Fy, float* d_Fz, float* d_OutputV);
__global__ void d_collisionGuo(float* d_f, float* d_Ux, float* d_Uy, float* d_Uz, float* d_Dens, float* d_Fx, float* d_Fy, float* d_Fz, bool* d_Obst);
__global__ void d_stream(float* d_f, bool* d_Obst);
__global__ void d_swap();
__global__ void d_macro(float* d_f, float* d_Ux, float* d_Uy, float* d_Uz, float* d_Dens, bool* d_Obst);
__global__ void g_BoundaryConditions(float* d_f, float* d_Ux, float* d_Uy, float* d_Uz, bool* d_Dens);
__device__ float d_guoForce(int k, float density, float ux, float uy, float uz, float fx, float fy, float fz, float tau);

void show_performance(ofstream& outFile, const struct timeval &t0, const struct timeval &t1,
                      int LX, int LY, int LZ, int ITERATIONS);

__device__ float d_feq(int k, float density, float ux, float uy, float uz);

__host__ void saveBinaryVtk(char* strAppend, float* h_OutputV);
__host__ void convertBintoVtk(int iterations, int step, float* dataout);

__device__ int d_current = 0;
__device__ int d_buffer = 1;