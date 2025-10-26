#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>

struct float3_ { float x, y, z; };

__global__ void density_kernel(
    const float3_* predicted_positions,
    const unsigned int* keys,
    const unsigned int* Offsets,
    const unsigned int* CellParticleCounts,
    float* rhos,
    float* rhos_near,
    int N,
    int grid_size_x, int grid_size_y, int grid_size_z,
    float R,
    int totalCells)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= N) return;

    float3_ pos = predicted_positions[p];
    float local_rho = 0.0f;
    float local_rho_near = 0.0f;

    unsigned int myKey = keys[p];
    if (myKey >= (unsigned int)totalCells) {
        rhos[p] = 0.0f;
        rhos_near[p] = 0.0f;
        return;
    }

    int cellIdx = (int)myKey;
    int cx = cellIdx % grid_size_x;
    int cy = (cellIdx / grid_size_x) % grid_size_y;
    int cz = cellIdx / (grid_size_x * grid_size_y);


    rhos[p] = local_rho;
    rhos_near[p] = local_rho_near;
}

extern "C" {

    __declspec(dllexport) int calculate_density_cuda(
        const float3_* positions_host,
        const unsigned int* keys_host,
        const unsigned int* Offsets_host,
        const unsigned int* CellCounts_host,
        float* rhos_host,
        float* rhos_near_host,
        int N,
        int grid_size_x, int grid_size_y, int grid_size_z,
        float R)
    {
        int totalCells = grid_size_x * grid_size_y * grid_size_z;

        // Device arrays
        float3_* d_positions = nullptr;
        unsigned int* d_keys = nullptr;
        unsigned int* d_offsets = nullptr;
        unsigned int* d_counts = nullptr;
        float* d_rhos = nullptr;
        float* d_rhos_near = nullptr;

        cudaMalloc(&d_positions, sizeof(float3_) * N);
        cudaMalloc(&d_keys, sizeof(unsigned int) * N);
        cudaMalloc(&d_offsets, sizeof(unsigned int) * totalCells);
        cudaMalloc(&d_counts, sizeof(unsigned int) * totalCells);
        cudaMalloc(&d_rhos, sizeof(float) * N);
        cudaMalloc(&d_rhos_near, sizeof(float) * N);

        cudaMemcpy(d_positions, positions_host, sizeof(float3_) * N, cudaMemcpyHostToDevice);
        cudaMemcpy(d_keys, keys_host, sizeof(unsigned int) * N, cudaMemcpyHostToDevice);
        cudaMemcpy(d_offsets, Offsets_host, sizeof(unsigned int) * totalCells, cudaMemcpyHostToDevice);
        cudaMemcpy(d_counts, CellCounts_host, sizeof(unsigned int) * totalCells, cudaMemcpyHostToDevice);

        int threads = 128;
        int blocks = (N + threads - 1) / threads;

        density_kernel<<<blocks, threads >>>(d_positions, d_keys, d_offsets, d_counts,
            d_rhos, d_rhos_near, N,
            grid_size_x, grid_size_y, grid_size_z,
            R, totalCells);

        cudaDeviceSynchronize();

        cudaMemcpy(rhos_host, d_rhos, sizeof(float) * N, cudaMemcpyDeviceToHost);
        cudaMemcpy(rhos_near_host, d_rhos_near, sizeof(float) * N, cudaMemcpyDeviceToHost);

        cudaFree(d_positions);
        cudaFree(d_keys);
        cudaFree(d_offsets);
        cudaFree(d_counts);
        cudaFree(d_rhos);
        cudaFree(d_rhos_near);

        return 0;
    }

}
