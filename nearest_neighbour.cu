// Implement a GPU program that, for `N` three-dimensional points stored on the device, 
// fills `indices[i]` with the index `j ≠ i` of the point closest to `points[i]`. 
// Comparing _squared_ Euclidean distance is sufficient—you do **not** need to compute square-roots.

#include <cuda_runtime.h>
#include <cfloat>

#define TILE_SIZE 256
#define BLOCK_SIZE 256

constexpr int POINTS_PER_THREAD = 4;
constexpr int TILE_POINTS = TILE_SIZE * POINTS_PER_THREAD;
constexpr int VECS_PER_BLOCK = (TILE_POINTS * 3) / 4;

__global__ void nearest_neighbour(const float* __restrict__ points, int* __restrict__ indices, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    bool active = idx < N;
    float min_dst = FLT_MAX;
    int min_index = -1;

    float3 cur_p {0.0f, 0.0f, 0.0f};
    if(active)
    {
        cur_p = make_float3(
            points[3 * idx],
            points[3 * idx + 1],
            points[3 * idx + 2]
        );
    }
    __shared__ float3 shared_tile_points[TILE_POINTS];

    float4* shared_vec = reinterpret_cast<float4*>(shared_tile_points);

    int tile_start = 0;
    
    #pragma unroll
    for(; tile_start + TILE_POINTS <= N; tile_start += TILE_POINTS)
    {
        const float4* tile_vec = reinterpret_cast<const float4*>(&points[3 * tile_start]);

        for(int v = threadIdx.x; v < VECS_PER_BLOCK; v += BLOCK_SIZE)
        {
            shared_vec[v] = tile_vec[v];
        }
        __syncthreads();

        if(active)
        {
            #pragma unroll 4
            for(int t = 0; t < TILE_POINTS; ++t)
            {
                float3 p = shared_tile_points[t];

                float dx = p.x - cur_p.x;
                float dy = p.y - cur_p.y;
                float dz = p.z - cur_p.z;

                float dist = t + tile_start == idx ? FLT_MAX : fmaf(dx, dx, fmaf(dy, dy, dz * dz));

                if(dist < min_dst)
                {
                    min_dst = dist;
                    min_index = t + tile_start;
                }
            }
        }
        
        __syncthreads();
    } 

    if(tile_start < N)
    {
        int cur_tile_size = N - tile_start;
        //1 point per thread
        if(active)
        {
            for(int t = 0; t < cur_tile_size; ++t)
            {
                float3 p = make_float3(
                    points[3 * (tile_start + t)],
                    points[3 * (tile_start + t) + 1],
                    points[3 * (tile_start + t) + 2]
                );

                float dx = p.x - cur_p.x;
                float dy = p.y - cur_p.y;
                float dz = p.z - cur_p.z;

                float dist = tile_start + t == idx ? FLT_MAX : fmaf(dx, dx, fmaf(dy, dy, dz * dz));

                if(dist < min_dst)
                {
                    min_dst = dist;
                    min_index = tile_start + t;
                }
            }
        }
    }

    if(active)
    {
        indices[idx] = min_index;
    }
}

// points and indices are device pointers
extern "C" void solve(const float* points, int* indices, int N) {
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    nearest_neighbour<<<numBlocks, BLOCK_SIZE>>>(points, indices, N);
}
