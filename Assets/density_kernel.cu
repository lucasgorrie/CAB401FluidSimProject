#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

struct float3_ { float x,y,z; };

// kernel: one thread per particle
extern "C" __global__
void density_kernel(
    const float3_* predicted_positions, 
    const unsigned int* keys,
    const unsigned int* Offsets, 
    const unsigned int* CellParticleCounts,
    float* rhos, 
    float* rhos_near,  
    int N,
    int grid_size_x, int grid_size_y, int grid_size_z,
    float R,
    int totalCells
)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= N) return;

    // Load position for particle p
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

    for (int dx = -1; dx <= 1; dx++) {
        int nx = cx + dx;
        if (nx < 0 || nx >= grid_size_x) continue;
        for (int dy = -1; dy <= 1; dy++) {
            int ny = cy + dy;
            if (ny < 0 || ny >= grid_size_y) continue;
            for (int dz = -1; dz <= 1; dz++) {
                int nz = cz + dz;
                if (nz < 0 || nz >= grid_size_z) continue;

                int neighborCellIdx = nx + ny * grid_size_x + nz * grid_size_x * grid_size_y;
                if (neighborCellIdx < 0 || neighborCellIdx >= totalCells) continue;

                unsigned int offset = Offsets[neighborCellIdx];
                unsigned int count = CellParticleCounts[neighborCellIdx];
                if (count == 0u) continue;

                if (offset >= (unsigned int)N) continue;
                if (offset + count > (unsigned int)N) count = (unsigned int)N - offset;

                for (unsigned int qi = 0; qi < count; ++qi) {
                    unsigned int q = offset + qi;

                    float3_ posQ = predicted_positions[q];

                    float dxp = pos.x - posQ.x;
                    float dyp = pos.y - posQ.y;
                    float dzp = pos.z - posQ.z;
                    float dist = sqrtf(dxp*dxp + dyp*dyp + dzp*dzp);
                    if (dist < R && dist > 0.0f) {
                        float normal_distance = 1.0f - dist / R;
                        local_rho += normal_distance * normal_distance * 2.0f;
                        local_rho_near += normal_distance * normal_distance * normal_distance * 2.0f;
                    }
                }
            }
        }
    }

    rhos[p] = local_rho;
    rhos_near[p] = local_rho_near;
}
