/**********************************
AUTHORS          : Antonio Maschio, Mahmoud Sedahmed, Rodrigo Coelho
CREATE DATE      : June 2022
PURPOSE          : Basic LBM code in CUDA C. Flow through a sphere.

Compilation: nvcc main.cu lbm.cu io.cu -o main
***********************************/

#include "header.cuh"

//host side arrays 
float *h_Ux, *h_Uy, *h_Uz;      //velocity
float *h_Fx, *h_Fy, *h_Fz;      //forces
float *h_f;                     //distribution function
float *h_Dens;                  //density
bool *h_Obst;                   //solid obstacles
float *h_OutputV;               //for the output
float *dataout;                 

//device side arrays
float *d_Ux, *d_Uy, *d_Uz;
float *d_Fx, *d_Fy, *d_Fz;
float *d_f;
float *d_Dens;
bool *d_Obst;
float *d_OutputV;


//simulation
int main(int argc, char** argv)
{
    clock_t cpu_start, cpu_end;
    cpu_start=clock();

    const int ITERATIONS = 200000;      //maximum number of LBM iterations

    char name[20];

    //allocate memory on the host side 
    size_t size = LX * LY * LZ;
    h_Ux = (float*)calloc(size, sizeof(float));
    h_Uy = (float*)calloc(size, sizeof(float));
    h_Uz = (float*)calloc(size, sizeof(float));
    h_Fx = (float*)calloc(size, sizeof(float));
    h_Fy = (float*)calloc(size, sizeof(float));
    h_Fz = (float*)calloc(size, sizeof(float));
    h_f = (float*)calloc(size * Q * 2, sizeof(float)); 
    h_Dens = (float*)calloc(size, sizeof(float));
    h_Obst = (bool*)malloc(size * sizeof(bool));
    h_OutputV = (float*)calloc(3 * size, sizeof(float));

    //allocate memory on the device side 
    HANDLE_ERROR(cudaMalloc(&d_Ux, size * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&d_Uy, size * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&d_Uz, size * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&d_Fx, size * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&d_Fy, size * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&d_Fz, size * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&d_f, size * sizeof(float) * Q * 2)); 
    HANDLE_ERROR(cudaMalloc(&d_Dens, size * sizeof(float))); 
    HANDLE_ERROR(cudaMalloc(&d_Obst, size * sizeof(bool))); 
    HANDLE_ERROR(cudaMalloc(&d_OutputV, 3 * size * sizeof(float))); //output array

    int systemRet = system("mkdir outputbin");
    if (systemRet == -1) { cout << "System could not create a folder outputbin" << endl; }
    int systemRet2 = system("mkdir outputvtk");
    if (systemRet2 == -1) { cout << "System could not create a folder outputvtk" << endl; }

    setObst(h_Obst);
    HANDLE_ERROR(cudaMemcpy(d_Obst, h_Obst, size * sizeof(bool), cudaMemcpyHostToDevice));// send obstacles do device.

    init(h_f, h_Ux, h_Uy, h_Uz, h_Dens, h_Fx, h_Fy, h_Fz, 0, 1);        //an external force is imposed

    //cudaMemcpy ( void* destination, const void* source, size_t count, cudaMemcpyKind type )
    HANDLE_ERROR(cudaMemcpy(d_f, h_f, size * sizeof(float) * Q * 2, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_Ux, h_Ux, size * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_Uy, h_Uy, size * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_Uz, h_Uz, size * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_Fx, h_Fx, size * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_Fy, h_Fy, size * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_Fz, h_Fz, size * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_Dens, h_Dens, size * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_OutputV, h_OutputV, 3 * size * sizeof(float), cudaMemcpyHostToDevice));

    sprintf(name, "vel-%d", 0);
    saveBinaryVtk(name, h_OutputV);

    dim3 block(256, 1, 1);     //check cuda occupancy size for your GPU
    dim3 grid(LX/block.x, LY/block.y, LZ/block.z); // the result of each division must be an integer
    int savestep = 20000;


    ofstream outFile;
    outFile.open ("performance.txt");

    struct timeval t0;
    gettimeofday(&t0,0);


    cout << "\nSimulation Started \n";
    for (int t = 1; t <= ITERATIONS; t++){
        //LBM cycle
        d_collisionGuo <<<grid, block >>> (d_f, d_Ux, d_Uy, d_Uz, d_Dens, d_Fx, d_Fy, d_Fz, d_Obst);
        HANDLE_ERROR(cudaDeviceSynchronize());
        d_stream <<<grid, block >>> (d_f, d_Obst);
        HANDLE_ERROR(cudaDeviceSynchronize());
        d_swap <<<1, 1 >>> ();
        HANDLE_ERROR(cudaDeviceSynchronize());
        d_macro <<<grid, block >>> (d_f, d_Ux, d_Uy, d_Uz, d_Dens, d_Obst);
        HANDLE_ERROR(cudaDeviceSynchronize());

        if (t % savestep == 0){
            d_compute_velocity <<<grid, block >>> (d_Ux, d_Uy, d_Uz, d_Fx, d_Fy, d_Fz, d_OutputV);
            HANDLE_ERROR(cudaMemcpy(h_OutputV, d_OutputV, 3 * size * sizeof(float), cudaMemcpyDeviceToHost));
            sprintf(name, "vel-%d", t);
            saveBinaryVtk(name, h_OutputV);
        }
    }

    // free memory 
    cudaFree(d_f);
    cudaFree(d_Ux); cudaFree(d_Uy); cudaFree(d_Uz);
    cudaFree(d_Fx); cudaFree(d_Fy); cudaFree(d_Fz);
    cudaFree(d_Dens);
    cudaFree(d_Obst);
    cudaFree(d_OutputV);

    free(h_f);
    free(h_Ux); free(h_Uy); free(h_Uz);
    free(h_Fx); free(h_Fy); free(h_Fz);
    free(h_Dens);
    free(h_Obst);
    free(h_OutputV);

    cudaDeviceReset();

    struct timeval t1;
    gettimeofday(&t1,0);
    show_performance(outFile, t0,t1, LX, LY, LZ, ITERATIONS);
    outFile.close();

    cout << "\nSimulation ended" << endl;
    
    cpu_end=clock();
    printf("Execution time : %4.6f \n", (double)((double)(cpu_end - cpu_start)/CLOCKS_PER_SEC));

    cout<<", converting output to VTK"<<endl;

    float * dataout;
    dataout = (float*)malloc(3 * size * sizeof(float));

    convertBintoVtk(ITERATIONS, savestep, dataout);

    free(dataout);
    return 0;
}