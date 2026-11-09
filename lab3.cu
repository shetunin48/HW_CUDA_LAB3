
#include <iostream>
#include <cmath>
#include <iomanip>
#include <vector>
#include <string>
#include <cstring>
#include <fstream>
#include <cstdlib>

#define CSC(call) \
do { \
    cudaError_t res = call; \
    if (res != cudaSuccess) { \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n", \
            __FILE__, __LINE__, cudaGetErrorString(res)); \
        exit(0); \
    } \
} while(0)


__constant__ float avg_norm[32][3];

__global__ void kernel(uchar4 *data, int w, int h, int nc) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;

    for (int i = idx; i < w * h; i += offset) {
        uchar4 p = data[i];

        int best = 0;
        float best_val = p.x * avg_norm[0][0] + p.y * avg_norm[0][1] + p.z * avg_norm[0][2];
        for (int j = 1; j < nc; ++j) {
            float val = p.x * avg_norm[j][0] + p.y * avg_norm[j][1] + p.z * avg_norm[j][2];
            if (val > best_val) {
                best_val = val;
                best = j;
            }
        }

        data[i].w = (unsigned char)best;
    }
}


int main(int argc, char* argv[]) {
    int blocksPerGrid = 1024; // База
    int threadsPerBlock = 1024;
    std::string filename = "";

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-f") == 0 && i + 1 < argc) {
            filename = argv[++i];
        } else if (std::strcmp(argv[i], "-grid") == 0 && i + 1 < argc) {
            blocksPerGrid = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-block") == 0 && i + 1 < argc) {
            threadsPerBlock = std::atoi(argv[++i]);
        }
    }




    char in_path[256], out_path[256];
    scanf("%s %s", in_path, out_path);

    int nc;
    scanf("%d", &nc);

    int w, h;
    FILE *fp = fopen(in_path, "rb");
    fread(&w, sizeof(int), 1, fp);
    fread(&h, sizeof(int), 1, fp);
    uchar4 *data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
    fread(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    float avg_host[32][3];
    for (int j = 0; j < nc; ++j) {
        int np;
        scanf("%d", &np);
        double sr = 0.0, sg = 0.0, sb = 0.0;
        for (int k = 0; k < np; ++k) {
            int px, py;
            scanf("%d %d", &px, &py);
            uchar4 p = data[py * w + px];
            sr += p.x;
            sg += p.y;
            sb += p.z;
        }
        double ar = sr / np, ag = sg / np, ab = sb / np;
        double len = sqrt(ar * ar + ag * ag + ab * ab);
        
        avg_host[j][0] = (float)(ar / len);
        avg_host[j][1] = (float)(ag / len);
        avg_host[j][2] = (float)(ab / len);
    }
    CSC(cudaMemcpyToSymbol(avg_norm, avg_host, sizeof(avg_host)));

    uchar4 *dev_data;
    CSC(cudaMalloc(&dev_data, sizeof(uchar4) * w * h));
    CSC(cudaMemcpy(dev_data, data, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CSC(cudaEventCreate(&start));
    CSC(cudaEventCreate(&stop));

    CSC(cudaEventRecord(start));

    kernel<<< blocksPerGrid, threadsPerBlock >>>(dev_data, w, h, nc);

    CSC(cudaEventRecord(stop));
    CSC(cudaEventSynchronize(stop));
    CSC(cudaGetLastError());

    float time_ms = 0.0f;
    CSC(cudaEventElapsedTime(&time_ms, start, stop));

    CSC(cudaEventDestroy(start));
    CSC(cudaEventDestroy(stop));

    fprintf(stderr, "TIME_MS: %f\n", time_ms);

    CSC(cudaMemcpy(data, dev_data, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

    CSC(cudaFree(dev_data));

    fp = fopen(out_path, "wb");
    fwrite(&w, sizeof(int), 1, fp);
    fwrite(&h, sizeof(int), 1, fp);
    fwrite(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    free(data);
    return 0;
}
