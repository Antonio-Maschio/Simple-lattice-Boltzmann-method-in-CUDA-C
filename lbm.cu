#include "header.cuh"

//Initialize the fields
__host__ void init(float* h_f, float* h_Ux, float* h_Uy, float* h_Uz,
    float* h_Dens, float* h_Fx, float* h_Fy, float* h_Fz, int current, int buffer)
{
    float density = 1.0f;
    float ux = 0.0f, uy = 0.0f, uz = 0.0f;
    float fx = 0.000001f, fy = 0.0f, fz = 0.0f;     //external force on fluid

    for (int z = 0; z < LZ; z++){
        for (int y = 0; y < LY; y++){
            for (int x = 0; x < LX; x++){
                density = 1.0f;
                h_Ux[I3D(z, y, x)] = ux;
                h_Uy[I3D(z, y, x)] = uy;
                h_Uz[I3D(z, y, x)] = uz;
                h_Fx[I3D(z, y, x)] = fx;
                h_Fy[I3D(z, y, x)] = fy;
                h_Fz[I3D(z, y, x)] = fz;
                h_Dens[I3D(z, y, x)] = density;
                for (int k = 0; k < Q; k++){
                    h_f[IDX(current, z, y, x, k)] = feq(k, density, ux, uy, uz);
                    h_f[IDX(buffer, z, y, x, k)] = 0.0;
                }
            }
        }
    }
}

//set the obstacles
__host__ void setObst(bool* h_Obst)
{
    for (int z = 0; z < LZ; z++){
        for (int y = 0; y < LY; y++){
            for (int x = 0; x < LX; x++){
                int xc = LX / 4, yc = LY / 2, zc = LZ / 2;
                float R = LY/8.0f;
                if (SQR(x - xc) + SQR(y - yc) + SQR(z - zc) < SQR(R)) {    //1 -> obstacle // 0 -> fluid
                    h_Obst[I3D(z, y, x)] = 1; }
                else {
                    h_Obst[I3D(z, y, x)] = 0; }
            }
        }
    }
}

//collision step
__global__ void d_collisionGuo(float* d_f, float* d_Ux, float* d_Uy, float* d_Uz, float* d_Dens, 
        float* d_Fx, float* d_Fy, float* d_Fz, bool* d_Obst)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;  //compute the position in the physical system
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;
    int ID = d_I3D(z, y, x);

    if (d_Obst[ID] == 0) {
        float f_old, heq, forceterm;
        float density = d_Dens[ID];
        float ux = d_Ux[ID], uy = d_Uy[ID], uz = d_Uz[ID];
        float Fx = d_Fx[ID], Fy = d_Fy[ID], Fz = d_Fz[ID];

        ux += 0.5f * Fx;    //compute the actual velocity
        uy += 0.5f * Fy;
        uz += 0.5f * Fz;
        for (int k = 0; k < d_Q; k++)  {
            f_old = d_f[d_IDX(d_current, z, y, x, k)];
            forceterm = d_guoForce(k, density, ux, uy, uz, Fx, Fy, Fz, d_tau);
            heq = d_feq(k, density, ux, uy, uz);
            d_f[d_IDX(d_current, z, y, x, k)] = f_old + (heq - f_old) / d_tau + forceterm;
        }
    }
}

//streaming step. Stream gather.
__global__ void d_stream(float* d_f, bool* d_Obst)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;

    int x_prev, y_prev, z_prev;
    for (int k = 0; k < d_Q; k++) {
        x_prev = (x - d_cx[k] + d_lx) % d_lx;
        y_prev = (y - d_cy[k] + d_ly) % d_ly;
        z_prev = (z - d_cz[k] + d_lz) % d_lz;
        if (d_Obst[d_I3D(z_prev, y_prev, x_prev)] == 0) {
            d_f[d_IDX(d_buffer, z, y, x, k)] = d_f[d_IDX(d_current, z_prev, y_prev, x_prev, k)]; }
        else {  //bounce-back
            d_f[d_IDX(d_buffer, z, y, x, k)] = d_f[d_IDX(d_current, z, y, x, opposite[k])]; }
    }
}

__global__ void d_swap()
{
    d_buffer = d_current;
    d_current = 1 - d_buffer;
}

//compute the macroscopic fields
__global__ void d_macro(float* d_f, float* d_Ux, float* d_Uy, float* d_Uz, float* d_Dens, bool* d_Obst)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;

    int ID = d_I3D(z, y, x);
    if (d_Obst[ID] == 0){
        float density = 0.0f, ux = 0.0f, uy = 0.0f, uz = 0.0f;
        for (int k = 0; k < d_Q; k++){
            density += d_f[d_IDX(d_current, z, y, x, k)];
            ux += d_f[d_IDX(d_current, z, y, x, k)] * d_cx[k];
            uy += d_f[d_IDX(d_current, z, y, x, k)] * d_cy[k];
            uz += d_f[d_IDX(d_current, z, y, x, k)] * d_cz[k];
        }
        d_Dens[ID] = density;
        d_Ux[ID] = ux / density;
        d_Uy[ID] = uy / density;
        d_Uz[ID] = uz / density;
    }
}

//Guo forcing term
__device__ float d_guoForce(int k, float density, float ux, float uy, float uz,
    float fx, float fy, float fz, float tau)
{
    float edu = (ux * d_cx[k] + uy * d_cy[k] + uz * d_cz[k]) * 3.0f;
    float edF = (d_cx[k] * fx + d_cy[k] * fy + d_cz[k] * fz) * 3.0f;
    float udF = (ux * fx + uy * fy + uz * fz) * 3.0f;

    return d_weight[k] * density * (1.0f - 0.5f / tau) * (edF + edu * edF - udF);
}

//equilibrium distribution
__host__ float feq(int k, float density, float ux, float uy, float uz)
{
    float u2 = ux * ux + uy * uy + uz * uz;
    float edu = (ux * cx[k] + uy * cy[k] + uz * cz[k]);

    return weight[k] * density * (1.0f + 3.0f * edu + 4.5f * edu * edu - 1.5f * u2);
}

__device__ float d_feq(int k, float density, float ux, float uy, float uz)
{

    float u2 = ux * ux + uy * uy + uz * uz;
    float edu = (ux * d_cx[k] + uy * d_cy[k] + uz * d_cz[k]);

    return d_weight[k] * density * (1.0f + 3.0f * edu + 4.5f * edu * edu - 1.5f * u2);
}//compute the actual velocity
__global__ void d_compute_velocity(float* d_Ux, float* d_Uy, float* d_Uz,
    float* d_Fx, float* d_Fy, float* d_Fz, float* d_OutputV)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;
    int ID = d_I3D(z, y, x);
    int size = d_lx * d_ly * d_lz;

    d_OutputV[ID] = d_Ux[ID] + 0.5f * d_Fx[ID];
    d_OutputV[ID + size] = d_Uy[ID] + 0.5f * d_Fy[ID];
    d_OutputV[ID + 2 * size] = d_Uz[ID] + 0.5f * d_Fz[ID];
}
