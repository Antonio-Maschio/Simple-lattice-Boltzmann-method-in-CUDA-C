#include "header.cuh"


//output
void saveBinaryVtk(char* strAppend, float* h_OutputV)
{
    char name[40];
    sprintf(name, "outputbin/%s.bin", strAppend);

    int file_size = 3 * LX * LY * LZ * sizeof(float);

    FILE* fp;
    fp = fopen(name, "wb");
    if (fp == NULL) {
        printf("\nError : cannot open file");
        exit(1);
    }
    fwrite(h_OutputV, file_size, 1, fp);

    fclose(fp);
}

//to visualize data in Paraview
void convertBintoVtk(int iterations, int step, float* dataout) // optional
{
    int systemsize = LX * LY * LZ;
    size_t size = 3 * LX * LY * LZ;

    for (int i = 0; i <= iterations; i += step)
    {
        //Reading and creating an array with bin data
        char name2[40]; 
        sprintf(name2, "outputbin/vel-%i.bin", i);

        FILE* fp;
        fp = fopen(name2, "rb");
        if (fp == NULL) {
            printf("\nError : cannot open file to read");
            exit(1);
        }
        fread(dataout, sizeof(float), size, fp);

        //writing the data in vtk
        char namevtk[40];

        sprintf(namevtk, "outputvtk/vel-%i.vtk", i);
        ofstream file(namevtk);
        file.setf(std::ios::fixed);
        file.precision(16);

        file << "# vtk DataFile Version 3.0" << endl;
        file << "Cavity" << endl;
        file << "ASCII" << endl;
        file << "DATASET STRUCTURED_POINTS" << endl;
        file << "DIMENSIONS " << LX << " " << LY << " " << LZ << endl;
        file << "ORIGIN " << 0 << " " << 0 << " " << 0 << endl;
        file << "SPACING " << 1 << " " << 1 << " " << 1 << endl;
        file << "POINT_DATA " << LX * LY * LZ << endl;
        file << "VECTORS " << "vector" << " double" << endl;

        for (size_t j = 0; j < systemsize; j++)        {
            file << dataout[j] << " " << dataout[j + systemsize] << " " << dataout[j + systemsize * 2] << endl;
        }

        file.close();
        fclose(fp);
    }
}


void show_performance(ofstream& outFile, const struct timeval &t0, const struct timeval &t1,
                      int LX, int LY, int LZ, int ITERATIONS)
{
    outFile << std::endl;

    long long elapsed = (t1.tv_sec - t0.tv_sec) * 1000000LL + t1.tv_usec - t0.tv_usec;

    outFile.precision(4);
    outFile.setf(std::ios::fixed, std::ios::floatfield);
    outFile << "time elapsed:   " << (double) elapsed / 1000000 << " seconds";
    outFile << " MLUPS = ";
    outFile << (ITERATIONS * (double) (LX*LY*LZ) / ((double) elapsed / 1000000)) / 1000000.0;
    outFile << std::endl;
}