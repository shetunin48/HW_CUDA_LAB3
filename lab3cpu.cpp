
#include <iostream>
#include <cmath>
#include <iomanip>
#include <vector>
#include <string>
#include <cstring>
#include <fstream>
#include <cstdlib>
#include <chrono>


struct uchar4 {
    uint8_t x, y, z, w;
};

int main(int argc, char* argv[]) {
    int blocksPerGrid = 1024;
    int threadsPerBlock = 1024;
    std::string filename = "";

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-f") == 0 && i + 1 < argc) {
            filename = argv[++i];
        } 
        else if (std::strcmp(argv[i], "-grid") == 0 && i + 1 < argc) {
            blocksPerGrid = std::atoi(argv[++i]);
        } 
        else if (std::strcmp(argv[i], "-block") == 0 && i + 1 < argc) {
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

    float avg_norm[32][3];
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
        
        avg_norm[j][0] = (float)(ar / len);
        avg_norm[j][1] = (float)(ag / len);
        avg_norm[j][2] = (float)(ab / len);
    }

    uchar4 *out = (uchar4 *)malloc(sizeof(uchar4) * w * h);

    auto start = std::chrono::high_resolution_clock::now();

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            uchar4 p = data[y * w + x];

            int best = 0;
            float best_val = p.x * avg_norm[0][0] + p.y * avg_norm[0][1] + p.z * avg_norm[0][2];
            for (int j = 1; j < nc; ++j) {
                float val = p.x * avg_norm[j][0] + p.y * avg_norm[j][1] + p.z * avg_norm[j][2];
                
                if (val > best_val) {
                    best_val = val;
                    best = j;
                }
            }

            out[y * w + x] = uchar4{p.x, p.y, p.z, (uint8_t)best};
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration elapsed = end - start;
    double time_ms = elapsed.count();

    std::cerr << "TIME_MS: " << time_ms << "\n";

    fp = fopen(out_path, "wb");
    fwrite(&w, sizeof(int), 1, fp);
    fwrite(&h, sizeof(int), 1, fp);
    fwrite(out, sizeof(uchar4), w * h, fp);
    fclose(fp);

    free(data);
    free(out);
    return 0;
}
