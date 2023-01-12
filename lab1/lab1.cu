#include <iostream>
#include <assert.h>

using namespace std;

#define CSC(call) \
do { \
    cudaError_t status = call; \
    if (status != cudaSuccess) { \
        fprintf(stderr, "ERROR is %s:%d. Message: %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
        exit(0); \
    } \
} while(0)

#define CONFLICT_FREE_OFFSET(n) ((n) >> 5)

const int MAX = 16777216;
const int THREADS = 1024;

__global__ void global_hist(int *data, int *hist, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;

    while (idx < n) {

        atomicAdd(hist + data[idx], 1);

        idx += offset;
    }
}


__global__ void blelloch_scan(int *hist, int *hist_sum, int n) {

    extern __shared__ int sdata[]; 

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;

    sdata[tid + CONFLICT_FREE_OFFSET(tid)] = hist[i];

    __syncthreads();

    unsigned int s = 1;
    unsigned int d = blockDim.x >> 1;

    for (s = 1; s < blockDim.x; s <<= 1) {

        int index1 = (2 * tid + 2) * s - 1;
        int index2 = (2 * tid + 1) * s - 1;

        index1 += CONFLICT_FREE_OFFSET(index1);
        index2 += CONFLICT_FREE_OFFSET(index2);


        if (tid < d) {
            sdata[index1] += sdata[index2];
        }

        d >>= 1;
        __syncthreads();
    }


    if (tid == 0) {
        hist_sum[blockIdx.x] = sdata[blockDim.x - 1 + CONFLICT_FREE_OFFSET(blockDim.x - 1)];
        sdata[blockDim.x - 1 + CONFLICT_FREE_OFFSET(blockDim.x - 1)] = 0;
    }

    d = 1;
    s = s / 2;

    for (; s >= 1; s /= 2) {

        int index2 = (2 * tid + 2) * s - 1;
        int index1 = (2 * tid + 1) * s - 1;

        index1 += CONFLICT_FREE_OFFSET(index1);
        index2 += CONFLICT_FREE_OFFSET(index2);

        if (tid < d) {
            int tmp = sdata[index1];
            sdata[index1] = sdata[index2];
            sdata[index2] += tmp;
        }

        d <<= 1;
        __syncthreads();
    }

    hist[i] = sdata[tid + CONFLICT_FREE_OFFSET(tid)];
}


__global__ void block_sum_hist(int *hist, int *hist_sum, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;
    int blc = blockIdx.x;

    while (idx < n) {

        hist[idx] += hist_sum[blc];

        idx += offset;
        blc += gridDim.x;
    }
}


__global__ void res_sort(int *res, int *data, int *hist, int n) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;

    while (idx < n) {

        res[atomicAdd(hist + data[idx], 1)] = data[idx];
       

        idx += offset;
    }
}

void recursion(int *dev_hist, int n) {

    if (n <= THREADS) {

        int *hist_block = (int *)malloc(sizeof(int) * n);

        for (int i = 0; i < n; ++i) {
            hist_block[i] = 0;
        }

        int *dev_hist_block;
        CSC(cudaMalloc(&dev_hist_block, sizeof(int) * n));
        CSC(cudaMemcpy(dev_hist_block, hist_block, sizeof(int) * n, cudaMemcpyHostToDevice));

        blelloch_scan<<<1, n, sizeof(int) * THREADS * 2>>>(dev_hist, dev_hist_block, n);
        CSC(cudaGetLastError());

        return;
    }
    
    int newn = n / THREADS;

    int *hist_block = (int *)malloc(sizeof(int) * newn);

    for (int i = 0; i < newn; ++i) {
        hist_block[i] = 0;
    }

    int *dev_hist_block;
    CSC(cudaMalloc(&dev_hist_block, sizeof(int) * newn));
    CSC(cudaMemcpy(dev_hist_block, hist_block, sizeof(int) * newn, cudaMemcpyHostToDevice));

    blelloch_scan<<<newn, THREADS, sizeof(int) * THREADS * 2>>>(dev_hist, dev_hist_block, newn);
    CSC(cudaGetLastError());

    recursion(dev_hist_block, newn);

    block_sum_hist<<<THREADS, THREADS>>>(dev_hist, dev_hist_block, n);
    CSC(cudaGetLastError());


    CSC(cudaFree(dev_hist_block));
    free(hist_block);
}



int main() {

    int n;
    //cin >> n;

    fread(&n, sizeof(int), 1, stdin);

    if (n == 0) {
        return 0;
    }

    int *data = (int *)malloc(sizeof(int) * n);


    fread(data, sizeof(int), n, stdin);

    int *res = (int *)malloc(sizeof(int) * n);

    for (int i = 0; i < n; ++i) {
        //cin >> data[i];
        res[i] = 0;
    }

    int *hist = (int *)malloc(sizeof(int) * MAX);

    for (int i = 0; i < MAX; ++i) {
        hist[i] = 0;
    }

    int *dev_data;
    CSC(cudaMalloc(&dev_data, sizeof(int) * n));
    CSC(cudaMemcpy(dev_data, data, sizeof(int) * n, cudaMemcpyHostToDevice));

    int *dev_res;
    CSC(cudaMalloc(&dev_res, sizeof(int) * n));
    CSC(cudaMemcpy(dev_res, res, sizeof(int) * n, cudaMemcpyHostToDevice));

    int *dev_hist;
    CSC(cudaMalloc(&dev_hist, sizeof(int) * MAX));
    CSC(cudaMemcpy(dev_hist, hist, sizeof(int) * MAX, cudaMemcpyHostToDevice));

    global_hist<<<1024, 1024>>>(dev_data, dev_hist, n);
    CSC(cudaGetLastError());

    recursion(dev_hist, MAX);

    res_sort<<<1024, 1024>>>(dev_res, dev_data, dev_hist, n);
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(res, dev_res, sizeof(int) * n, cudaMemcpyDeviceToHost));

    fwrite(res, sizeof(int), n, stdout);

    // for (int i = 0; i < n; ++i) {
    //     cout << res[i] << " ";
    //    // cin >> data[i];
    //     //res[i] = 0;
    // }
    // cout << endl;

    CSC(cudaFree(dev_data));
    CSC(cudaFree(dev_res));
    CSC(cudaFree(dev_hist));
    free(data);
    free(hist);
    free(res);
    return 0;
}
