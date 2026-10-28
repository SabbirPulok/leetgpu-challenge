// ================================================================
// Single-pass top-K with a monotone concurrent accumulator
//
// Signature (same as top_k_selection.cu):
//   extern "C" void solve(const float* input, float* output, int N, int k)
//
// Lemma: the k-th largest of ANY subset of the input is <= the true k-th
// largest v_k. So partial counts give a safe threshold that only rises.
//
// Shared state (all small enough to live in L2):
//   tau        max-register (only atomicMax); invariant: #(key >= tau) >= k
//   c8 / c16   grow-only counters over the top 8 / 16 bits of the key
//   z8 / z16   "zoom" counters over the low 16 bits, only for the 16-bit
//              bucket tau currently sits in, so tau can land on an exact key
//   log        append-only list of survivors (keys with key > tau on arrival)
//
// Pipeline (no host sync for k <= FAST_K):
//   1) init:       zero counters; for k <= SEED_MAX_K seed tau with a lower bound
//                  of the k-th largest of a strided sample (safe by the lemma)
//   2) warm-up:    for larger k, accumulate the first few tiles (~8k keys) with
//                  tau = 0, then refresh tau from their counts
//   3) accumulate: one pass over the remaining tiles in a random order; each key
//                  is compared with a cached tau, survivors go to the log and the
//                  counters; one warp at a time refreshes tau from the counters
//   4) finalize:   all blocks compact log entries > tau; the last block to finish
//                  refreshes tau, selects, sorts, and pads with tau (ties)
//
// Correctness: every update commutes (add, max, append) and a stale read can
// only make tau lower, so tau <= v_k at all times. Every element > v_k passes
// the filter whenever it arrives; strict '>' plus padding handles ties.
// ================================================================

// nvcc -std=c++17 -O3 -arch=sm_75 \
//    -Ithird_party/cccl/cub \
//    -Ithird_party/cccl/libcudacxx/include \
//    -Ithird_party/cccl/thrust \
//    top_k_monotone.cu -o top_k_monotone && ./top_k_monotone

#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <cstring>
#include <string>
#include <cuda_runtime.h>

#include <cub/device/device_topk.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cuda/std/execution>

#define BLOCK_DIM 256
#define VEC_PER_THREAD 4                          // float4 loads per thread per tile
#define TILE (BLOCK_DIM * VEC_PER_THREAD * 4)     // 4096 floats = 16 KB
#define ZOOM_SLOTS 4
#define SAMPLE_SIZE 8192                          // strided sample for the seed
#define SEED_MAX_K 1024                           // larger k uses the warm-up instead
#define WARMUP_KEYS_PER_K 8                       // warm-up covers ~8k keys (at least 8192)
#define FIN_THREADS 1024
#define FIN_CAP 8192                              // candidates sorted in shared memory
#define FAST_K 2048                               // k above this uses the host-side sort path
#define CACHE_SLOTS 32                            // per-block cache of hot 16-bit buckets
#define FLUSH_TILES 2                             // flush the cache to global every 2 tiles
#define LOG_CHUNK 32                              // max log slots a warp reserves at once
#define NONE 0xFFFFFFFFu
#define FULL_MASK 0xFFFFFFFFu

#ifndef TOPK_RANDOM_TILES
#define TOPK_RANDOM_TILES 1
#endif

__device__ __forceinline__ unsigned float_to_uint(float f)
{
    unsigned u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}

__device__ __forceinline__ float uint_to_float(unsigned u)
{
    u = (u & 0x80000000u) ? (u ^ 0x80000000u) : ~u;
    return __uint_as_float(u);
}

struct State
{
    unsigned tau;        // current threshold key
    unsigned cursor;     // number of survivors appended to the log
    unsigned last_ref;   // cursor value at the last refresh
    unsigned lock;       // one refresher at a time (efficiency only, not correctness)
    unsigned zoom_word;  // (slot << 16) | bucket16, or NONE
    unsigned next_slot;  // next unused zoom slot
    unsigned ccount;     // compacted candidates (finalize)
    unsigned done;       // finalize blocks finished
};

struct Acc
{
    State* st;
    unsigned* c8;
    unsigned* c16;
    unsigned* z8;
    unsigned* z16;
    unsigned* log;
};

// Largest bin b in cnt[0..255] with sum(cnt[b..255]) >= need (one warp).
// above = sum(cnt[b+1..255]), bin_cnt = cnt[b]. Returns false if total < need.
// GLOBAL = true reads through L2 (counters updated by other SMs).
template <bool GLOBAL = true>
__device__ bool warp_find(const unsigned* cnt, unsigned need,
                          unsigned& bin, unsigned& above, unsigned& bin_cnt)
{
    int lane = threadIdx.x & 31;
    unsigned v[8];
    unsigned s = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        v[i] = GLOBAL ? __ldcg(cnt + lane * 8 + i) : cnt[lane * 8 + i];
        s += v[i];
    }

    // Inclusive suffix sum over lanes (lane l gets sum of lanes l..31)
    unsigned incl = s;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1)
    {
        unsigned t = __shfl_down_sync(FULL_MASK, incl, off);
        if (lane + off < 32) incl += t;
    }

    unsigned ballot = __ballot_sync(FULL_MASK, incl >= need);
    if (ballot == 0) return false;
    int L = 31 - __clz(ballot);

    unsigned b = 0, a = 0, c = 0;
    if (lane == L)
    {
        unsigned acc = incl - s;
        for (int i = 7; i >= 0; --i)
        {
            if (acc + v[i] >= need)
            {
                b = lane * 8 + i;
                a = acc;
                c = v[i];
                break;
            }
            acc += v[i];
        }
    }
    bin = __shfl_sync(FULL_MASK, b, L);
    above = __shfl_sync(FULL_MASK, a, L);
    bin_cnt = __shfl_sync(FULL_MASK, c, L);
    return true;
}

// Recompute a lower bound for v_k from the counters and publish it (one warp).
// Each counter read counts real, distinct elements, so any interleaving with
// concurrent increments only under-counts: the bound stays safe.
__device__ void refresh_tau(const Acc& a, unsigned k, bool allow_zoom)
{
    int lane = threadIdx.x & 31;
    unsigned b0, a0, n0;
    if (!warp_find(a.c8, k, b0, a0, n0)) return;

    unsigned cand = b0 << 24;
    unsigned b1, a1, n1;
    if (warp_find(a.c16 + b0 * 256, k - a0, b1, a1, n1))
    {
        unsigned B = (b0 << 8) | b1;
        unsigned above = a0 + a1;
        cand = B << 16;

        unsigned zw = __ldcg(&a.st->zoom_word);
        if (zw != NONE && (zw & 0xFFFFu) == B)
        {
            unsigned slot = zw >> 16;
            unsigned z0, a2, n2;
            if (warp_find(a.z8 + slot * 256, k - above, z0, a2, n2))
            {
                cand = (B << 16) | (z0 << 8);
                unsigned z1, a3, n3;
                if (warp_find(a.z16 + slot * 65536 + z0 * 256, k - above - a2, z1, a3, n3))
                {
                    cand |= z1;
                }
            }
        }
        // Zoom only into a "fat" bucket (>= 2k counted); thin ones cost little.
        // Slots are zeroed by init_kernel and used once per call, never recycled.
        else if (allow_zoom && n1 >= 2 * k && (zw == NONE || B > (zw & 0xFFFFu)))
        {
            if (lane == 0)
            {
                unsigned slot = __ldcg(&a.st->next_slot);
                if (slot < ZOOM_SLOTS)
                {
                    atomicExch(&a.st->next_slot, slot + 1);
                    atomicExch(&a.st->zoom_word, (slot << 16) | B);
                }
            }
        }
    }
    if (lane == 0) atomicMax(&a.st->tau, cand);
}

// k-th largest key among the keys accepted by get(i, u), i < n (whole block,
// blockDim a multiple of 32, at least k accepted keys). greater = #accepted keys above it.
// With passes < 4 it returns the lowest key of the bucket holding the k-th largest:
// a lower bound, which is all a seed needs.
template <typename Get>
__device__ unsigned block_radix_select(unsigned n, unsigned k, Get get, unsigned* hist,
                                       unsigned* sel, unsigned& greater, int passes = 4)
{
    const int lane = threadIdx.x & 31;
    if (threadIdx.x == 0)
    {
        sel[0] = 0; // prefix
        sel[1] = 0; // mask
        sel[2] = k; // remaining rank
    }
    __syncthreads();

    for (int pass = 0; pass < passes; ++pass)
    {
        int shift = 24 - pass * 8;
        if (threadIdx.x < 256) hist[threadIdx.x] = 0;
        __syncthreads();
        unsigned prefix = sel[0], mask = sel[1];
        for (unsigned base = 0; base < n; base += blockDim.x)
        {
            unsigned i = base + threadIdx.x;
            unsigned u = 0;
            bool ok = i < n && get(i, u) && (u & mask) == prefix;
            unsigned bin = ok ? (u >> shift) & 0xFFu : 256u;
            // Warp-aggregate: skewed data puts most keys in a few bins
            unsigned peers = __match_any_sync(FULL_MASK, bin);
            if (ok && lane == __ffs(peers) - 1) atomicAdd(&hist[bin], (unsigned)__popc(peers));
        }
        __syncthreads();
        if (threadIdx.x < 32)
        {
            unsigned b, above, c;
            if (warp_find<false>(hist, sel[2], b, above, c) && lane == 0)
            {
                sel[0] |= b << shift;
                sel[1] |= 0xFFu << shift;
                sel[2] -= above;
            }
        }
        __syncthreads();
    }
    greater = k - sel[2];
    return sel[0];
}

// Blocks 1.. zero the counters; block 0 resets the state and, for k <= SEED_MAX_K,
// seeds tau with a lower bound of the k-th largest of a strided sample
__global__ void init_kernel(const float* __restrict__ input, int N, unsigned k,
                            Acc a, unsigned zero_words)
{
    if (blockIdx.x != 0)
    {
        uint4* counters4 = reinterpret_cast<uint4*>(a.c8); // c8, c16, z8, z16 are contiguous
        for (unsigned i = (blockIdx.x - 1) * blockDim.x + threadIdx.x; i < zero_words / 4;
             i += (gridDim.x - 1) * blockDim.x)
        {
            counters4[i] = make_uint4(0, 0, 0, 0);
        }
        return;
    }

    __shared__ unsigned sample[SAMPLE_SIZE];
    __shared__ unsigned hist[256];
    __shared__ unsigned sel[3];

    // blockDim.x == 1024 here
    unsigned s = min((unsigned)N, (unsigned)SAMPLE_SIZE);
    unsigned n = 0;
    if (s == SAMPLE_SIZE && k <= 256)
    {
        // Each thread keeps the max of 8 sampled keys. The maxima are still a subset of
        // the input, and for small k their k-th largest sits near the full sample's.
        unsigned stride = (unsigned)N / SAMPLE_SIZE;
        unsigned m = 0;
#pragma unroll
        for (int j = 0; j < SAMPLE_SIZE / 1024; ++j)
        {
            m = max(m, float_to_uint(__ldg(input + (size_t)(j * 1024 + threadIdx.x) * stride)));
        }
        sample[threadIdx.x] = m;
        n = 1024;
    }
    else if (k <= min(s, (unsigned)SEED_MAX_K))
    {
        for (unsigned i = threadIdx.x; i < s; i += blockDim.x)
        {
            sample[i] = float_to_uint(input[(size_t)i * N / s]);
        }
        n = s;
    }

    unsigned seed = 0;
    if (n > 0)
    {
        __syncthreads();
        unsigned greater;
        seed = block_radix_select(n, k, [&](unsigned i, unsigned& u) { u = sample[i]; return true; },
                                  hist, sel, greater, 2);
    }

    if (threadIdx.x == 0)
    {
        a.st->tau = seed;
        a.st->cursor = 0;
        a.st->last_ref = 0;
        a.st->lock = 0;
        a.st->zoom_word = NONE;
        a.st->next_slot = 0;
        a.st->ccount = 0;
        a.st->done = 0;
    }
}

// The single pass over the input: tiles [tile_begin, tile_end) of a permutation of num_tiles
template <bool VEC>
__global__ void __launch_bounds__(BLOCK_DIM)
accumulate_kernel(const float* __restrict__ input, int N, unsigned k, unsigned delta,
                  unsigned num_tiles, unsigned tile_begin, unsigned tile_end,
                  unsigned perm_mul, unsigned log_chunk, Acc a)
{
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const float4* in4 = reinterpret_cast<const float4*>(input);

    // Survivors cluster in one or two 16-bit buckets just above tau, so counting
    // each one with a global atomic serializes on a single L2 address. Count hot
    // buckets in shared memory instead and flush them every FLUSH_TILES tiles.
    __shared__ unsigned cache_tag[CACHE_SLOTS];
    __shared__ unsigned cache_cnt[CACHE_SLOTS];
    if (tid < CACHE_SLOTS)
    {
        cache_tag[tid] = NONE;
        cache_cnt[tid] = 0;
    }
    __syncthreads();

    // Per-warp log chunk [chunk_pos, chunk_end): one cursor atomic per log_chunk survivors
    unsigned chunk_pos = 0, chunk_end = 0;

    unsigned iter = 0;
    for (unsigned t = tile_begin + blockIdx.x; t < tile_end; t += gridDim.x, ++iter)
    {
#if TOPK_RANDOM_TILES
        unsigned tile = (unsigned)(((unsigned long long)t * perm_mul) % num_tiles);
#else
        unsigned tile = t;
#endif
        unsigned tau = __ldcg(&a.st->tau);
        unsigned zw = __ldcg(&a.st->zoom_word);

        // Load 16 keys per thread; out-of-range keys are 0 (never > tau)
        unsigned keys[VEC_PER_THREAD * 4];
        size_t base = (size_t)tile * TILE;
#pragma unroll
        for (int j = 0; j < VEC_PER_THREAD; ++j)
        {
            size_t i = base + (size_t)(j * BLOCK_DIM + tid) * 4;
            if (VEC && i + 4 <= (size_t)N)
            {
                float4 v = __ldcs(in4 + i / 4);
                keys[j * 4 + 0] = float_to_uint(v.x);
                keys[j * 4 + 1] = float_to_uint(v.y);
                keys[j * 4 + 2] = float_to_uint(v.z);
                keys[j * 4 + 3] = float_to_uint(v.w);
            }
            else
            {
#pragma unroll
                for (int c = 0; c < 4; ++c)
                {
                    keys[j * 4 + c] = (i + c < (size_t)N) ? float_to_uint(__ldcs(input + i + c)) : 0u;
                }
            }
        }

        unsigned cnt = 0;
#pragma unroll
        for (int q = 0; q < VEC_PER_THREAD * 4; ++q) cnt += (keys[q] > tau);

        // Rare path: warp-aggregated append + counter updates
        if (__any_sync(FULL_MASK, cnt > 0))
        {
            unsigned incl = cnt;
#pragma unroll
            for (int off = 1; off < 32; off <<= 1)
            {
                unsigned t2 = __shfl_up_sync(FULL_MASK, incl, off);
                if (lane >= off) incl += t2;
            }
            // Survivors j < room go to the current chunk, the rest to a new one
            unsigned total = __shfl_sync(FULL_MASK, incl, 31);
            unsigned room = chunk_end - chunk_pos;
            unsigned new_base = 0, new_size = 0;
            if (total > room)
            {
                new_size = max(log_chunk, total - room);
                if (lane == 0) new_base = atomicAdd(&a.st->cursor, new_size);
                new_base = __shfl_sync(FULL_MASK, new_base, 0);
            }
            unsigned j = incl - cnt;

            if (cnt > 0)
            {
                bool zoom_on = (zw != NONE);
                unsigned zb = zw & 0xFFFFu;
                unsigned zs = zw >> 16;
#pragma unroll
                for (int q = 0; q < VEC_PER_THREAD * 4; ++q)
                {
                    unsigned u = keys[q];
                    if (u > tau)
                    {
                        a.log[j < room ? chunk_pos + j : new_base + (j - room)] = u;
                        ++j;

                        unsigned b = u >> 16;
                        unsigned cs = b % CACHE_SLOTS;
                        unsigned tag = cache_tag[cs];
                        if (tag == NONE)
                        {
                            unsigned old = atomicCAS(&cache_tag[cs], NONE, b);
                            tag = (old == NONE) ? b : old;
                        }
                        if (tag == b)
                        {
                            atomicAdd(&cache_cnt[cs], 1u);
                        }
                        else
                        {
                            atomicAdd(&a.c16[b], 1u);
                            atomicAdd(&a.c8[u >> 24], 1u);
                        }

                        if (zoom_on && b == zb)
                        {
                            atomicAdd(&a.z8[zs * 256 + ((u >> 8) & 0xFFu)], 1u);
                            atomicAdd(&a.z16[zs * 65536 + (u & 0xFFFFu)], 1u);
                        }
                    }
                }
            }

            if (total > room)
            {
                chunk_pos = new_base + (total - room);
                chunk_end = new_base + new_size;
            }
            else
            {
                chunk_pos += total;
            }
        }

        // Flush cached bucket counts (block-uniform point); free slots that went cold
        if (iter % FLUSH_TILES == FLUSH_TILES - 1)
        {
            __syncthreads();
            if (tid < CACHE_SLOTS)
            {
                unsigned c = cache_cnt[tid];
                if (c > 0)
                {
                    unsigned b = cache_tag[tid];
                    atomicAdd(&a.c16[b], c);
                    atomicAdd(&a.c8[b >> 8], c);
                    cache_cnt[tid] = 0;
                }
                else
                {
                    cache_tag[tid] = NONE;
                }
            }
            __syncthreads();
        }

        // Warp 0 of any block may refresh tau once delta new survivors arrived
        if (warp == 0)
        {
            int go = 0;
            unsigned s = 0;
            if (lane == 0)
            {
                s = __ldcg(&a.st->cursor);
                unsigned last = __ldcg(&a.st->last_ref);
                if (s >= k && s - last >= delta && atomicCAS(&a.st->lock, 0u, 1u) == 0u) go = 1;
            }
            go = __shfl_sync(FULL_MASK, go, 0);
            if (go)
            {
                refresh_tau(a, k, true);
                if (lane == 0)
                {
                    atomicExch(&a.st->last_ref, s);
                    __threadfence();
                    atomicExch(&a.st->lock, 0u);
                }
            }
        }
    }

    // Final flush, and fill the unused end of this warp's chunk with key 0 (never > tau)
    __syncthreads();
    if (tid < CACHE_SLOTS && cache_cnt[tid] > 0)
    {
        unsigned b = cache_tag[tid];
        atomicAdd(&a.c16[b], cache_cnt[tid]);
        atomicAdd(&a.c8[b >> 8], cache_cnt[tid]);
    }
    for (unsigned p = chunk_pos + lane; p < chunk_end; p += 32) a.log[p] = 0;
}

// Descending bitonic sort of keys[0..n) in shared memory (n is a power of 2)
__device__ void block_bitonic_desc(unsigned* keys, unsigned n)
{
    for (unsigned size = 2; size <= n; size <<= 1)
    {
        for (unsigned stride = size >> 1; stride > 0; stride >>= 1)
        {
            for (unsigned i = threadIdx.x; i < n; i += blockDim.x)
            {
                unsigned j = i ^ stride;
                if (j > i)
                {
                    unsigned x = keys[i], y = keys[j];
                    bool desc = (i & size) == 0;
                    if ((x < y) == desc)
                    {
                        keys[i] = y;
                        keys[j] = x;
                    }
                }
            }
            __syncthreads();
        }
    }
}

// k <= FAST_K. Every block compacts its slice of the log (keys > current tau) into
// cbuf; the last block to finish refreshes tau, selects, sorts, and pads with tau.
__global__ void __launch_bounds__(FIN_THREADS)
finalize_kernel(Acc a, unsigned* __restrict__ cbuf, float* __restrict__ output, unsigned k)
{
    __shared__ unsigned cand[FIN_CAP];
    __shared__ unsigned top[FAST_K];
    __shared__ unsigned hist[256];
    __shared__ unsigned sel[3];
    __shared__ unsigned count;
    __shared__ bool is_last;
    const int lane = threadIdx.x & 31;

    unsigned tau0 = __ldcg(&a.st->tau);
    unsigned S = __ldcg(&a.st->cursor);

    // Log is cudaMalloc'd (16-byte aligned) and padded, so read it 4 keys at a time
    const uint4* log4 = reinterpret_cast<const uint4*>(a.log);
    unsigned n4 = (S + 3) / 4;
    for (unsigned base = blockIdx.x * blockDim.x; base < n4; base += gridDim.x * blockDim.x)
    {
        unsigned i = base + threadIdx.x;
        unsigned w[4] = {0, 0, 0, 0};
        if (i < n4)
        {
            uint4 v = __ldcg(log4 + i);
            w[0] = v.x; w[1] = v.y; w[2] = v.z; w[3] = v.w;
        }
        unsigned c = 0;
#pragma unroll
        for (int q = 0; q < 4; ++q)
        {
            if (i * 4 + q >= S) w[q] = 0;
            c += (w[q] > tau0);
        }
        unsigned incl = c;
#pragma unroll
        for (int off = 1; off < 32; off <<= 1)
        {
            unsigned t = __shfl_up_sync(FULL_MASK, incl, off);
            if (lane >= off) incl += t;
        }
        unsigned wbase = 0;
        if (lane == 31 && incl > 0) wbase = atomicAdd(&a.st->ccount, incl);
        wbase = __shfl_sync(FULL_MASK, wbase, 31);
        unsigned pos = wbase + incl - c;
#pragma unroll
        for (int q = 0; q < 4; ++q)
        {
            if (w[q] > tau0) cbuf[pos++] = w[q];
        }
    }

    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) is_last = atomicAdd(&a.st->done, 1u) == gridDim.x - 1;
    __syncthreads();
    if (!is_last) return;
    __threadfence();

    // Last block: a final refresh can only raise tau, so cbuf still holds every key > tau
    if (threadIdx.x < 32)
    {
        refresh_tau(a, k, false);
        __threadfence();
    }
    if (threadIdx.x == 0) count = 0;
    __syncthreads();

    unsigned tau = __ldcg(&a.st->tau);
    unsigned C0 = __ldcg(&a.st->ccount);
    for (unsigned i = threadIdx.x; i < C0; i += blockDim.x)
    {
        unsigned u = __ldcg(cbuf + i);
        if (u > tau)
        {
            unsigned p = atomicAdd(&count, 1u);
            if (p < FIN_CAP) cand[p] = u;
        }
    }
    __syncthreads();

    unsigned C = count;
    unsigned pad = tau;
    unsigned G = C;
    unsigned* keys = cand;

    // Many candidates: find the exact k-th largest first (in shared memory if they fit,
    // otherwise over cbuf), then sort only the G < k keys above it
    if (C > FIN_CAP || C > max(2 * k, 512u))
    {
        bool in_smem = C <= FIN_CAP;
        unsigned greater;
        if (in_smem)
            pad = block_radix_select(C, k, [&](unsigned i, unsigned& u) { u = cand[i]; return true; },
                                     hist, sel, greater);
        else
            pad = block_radix_select(C0, k, [&](unsigned i, unsigned& u) { u = __ldcg(cbuf + i); return u > tau; },
                                     hist, sel, greater);
        G = greater;
        if (threadIdx.x == 0) count = 0;
        __syncthreads();
        unsigned n_src = in_smem ? C : C0;
        for (unsigned i = threadIdx.x; i < n_src; i += blockDim.x)
        {
            unsigned u = in_smem ? cand[i] : __ldcg(cbuf + i);
            if (u > pad) top[atomicAdd(&count, 1u)] = u;
        }
        __syncthreads();
        keys = top;
    }

    unsigned n = 1;
    while (n < G) n <<= 1;
    for (unsigned i = G + threadIdx.x; i < n; i += blockDim.x) keys[i] = 0;
    __syncthreads();
    block_bitonic_desc(keys, n);

    for (unsigned i = threadIdx.x; i < k; i += blockDim.x)
    {
        output[i] = uint_to_float(i < G ? keys[i] : pad);
    }
}

// One refresh of tau (after the warm-up, and before the host sort for k > FAST_K)
__global__ void refresh_kernel(Acc a, unsigned k, bool allow_zoom)
{
    refresh_tau(a, k, allow_zoom);
}

// Emit sorted survivors > tau, padded with tau
__global__ void emit_kernel(const unsigned* __restrict__ sorted, unsigned S,
                            const State* __restrict__ st, float* __restrict__ output, unsigned k)
{
    unsigned tau = st->tau;
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < k; i += gridDim.x * blockDim.x)
    {
        unsigned u = (i < S && sorted[i] > tau) ? sorted[i] : tau;
        output[i] = uint_to_float(u);
    }
}

// Workspace cached across calls (CUB's temp storage is also preallocated in the benchmark)
struct Workspace
{
    unsigned* region = nullptr;   // State + c8 + c16 + z8 + z16
    unsigned* log = nullptr;
    unsigned* cbuf = nullptr;     // compacted candidates (finalize)
    size_t log_cap = 0;
    unsigned* sorted = nullptr;
    size_t sorted_cap = 0;
    void* sort_temp = nullptr;
    size_t sort_temp_bytes = 0;
    int main_blocks = 0;
    int sms = 0;
    Acc acc{};
};

static Workspace g_ws;

static const size_t STATE_WORDS = 64; // padded
static const size_t C_WORDS = 256 + 65536;
static const size_t Z_WORDS = ZOOM_SLOTS * (256 + 65536);

static void ensure_workspace(int N)
{
    if (!g_ws.region)
    {
        cudaMalloc(&g_ws.region, (STATE_WORDS + C_WORDS + Z_WORDS) * sizeof(unsigned));
        g_ws.acc.st = reinterpret_cast<State*>(g_ws.region);
        g_ws.acc.c8 = g_ws.region + STATE_WORDS;
        g_ws.acc.c16 = g_ws.acc.c8 + 256;
        g_ws.acc.z8 = g_ws.acc.c16 + 65536;
        g_ws.acc.z16 = g_ws.acc.z8 + ZOOM_SLOTS * 256;

        int dev = 0, sms = 0, per_sm = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, accumulate_kernel<true>, BLOCK_DIM, 0);
        g_ws.main_blocks = sms * std::max(per_sm, 1);
        g_ws.sms = sms;
    }
    // Survivors <= N, plus at most one partly used chunk per warp, so it never overflows
    size_t need = (size_t)N + (size_t)g_ws.main_blocks * (BLOCK_DIM / 32) * LOG_CHUNK + 4;
    if (g_ws.log_cap < need)
    {
        cudaFree(g_ws.log);
        cudaFree(g_ws.cbuf);
        cudaMalloc(&g_ws.log, need * sizeof(unsigned));
        cudaMalloc(&g_ws.cbuf, need * sizeof(unsigned));
        g_ws.log_cap = need;
        g_ws.acc.log = g_ws.log;
    }
}

static unsigned choose_perm_mul(unsigned T)
{
    if (T <= 1) return 1;
    unsigned A = 2654435761u % T;
    if (A == 0) A = 1;
    while (std::gcd(A, T) != 1) A = (A + 1) % T == 0 ? 1 : A + 1;
    return A;
}

// Entrypoint: input and output are device pointers
extern "C" void solve(const float* input, float* output, int N, int k)
{
    if (N <= 0 || k <= 0) return;
    if (k > N) k = N;

    ensure_workspace(N);
    Acc a = g_ws.acc;

    unsigned num_tiles = (unsigned)(((size_t)N + TILE - 1) / TILE);
    unsigned perm_mul = choose_perm_mul(num_tiles);
    unsigned delta = std::max(32u, (unsigned)k / 4);
    int blocks = (int)std::min<unsigned>(num_tiles, (unsigned)g_ws.main_blocks);
    // Chunk padding costs up to warps * chunk log slots; keep that below ~N/1000
    unsigned warps = (unsigned)blocks * (BLOCK_DIM / 32);
    unsigned log_chunk = (unsigned)std::clamp<size_t>((size_t)N / (1000 * (size_t)warps), 1, LOG_CHUNK);

    init_kernel<<<64, 1024>>>(input, N, k, a, (unsigned)(C_WORDS + Z_WORDS));

    // Without a seed (large k), a warm-up runs the first few tiles with tau = 0 on a few
    // blocks, then one refresh, so the main pass never starts with a whole wave of
    // unfiltered keys
    bool seeded = (unsigned)k <= std::min<unsigned>((unsigned)N, SEED_MAX_K);
    size_t warm_keys = std::max<size_t>((size_t)WARMUP_KEYS_PER_K * k, 8192);
    unsigned warm_tiles = seeded ? 0 : (unsigned)std::min<size_t>(num_tiles, (warm_keys + TILE - 1) / TILE);

    bool aligned = (reinterpret_cast<uintptr_t>(input) % 16 == 0);
    auto accumulate = [&](unsigned begin, unsigned end, int grid) {
        if (aligned)
            accumulate_kernel<true><<<grid, BLOCK_DIM>>>(input, N, k, delta, num_tiles, begin, end, perm_mul, log_chunk, a);
        else
            accumulate_kernel<false><<<grid, BLOCK_DIM>>>(input, N, k, delta, num_tiles, begin, end, perm_mul, log_chunk, a);
    };
    if (warm_tiles > 0)
    {
        accumulate(0, warm_tiles, (int)std::min<unsigned>(warm_tiles, (unsigned)blocks));
        refresh_kernel<<<1, 32>>>(a, k, true);
    }
    if (warm_tiles < num_tiles) accumulate(warm_tiles, num_tiles, blocks);

    if (k <= FAST_K)
    {
        finalize_kernel<<<g_ws.sms, FIN_THREADS>>>(a, g_ws.cbuf, output, k);
        return;
    }

    // Large k: sort the (small) survivor log with CUB and emit with padding
    refresh_kernel<<<1, 32>>>(a, k, false);
    unsigned S = 0;
    cudaMemcpy(&S, &a.st->cursor, sizeof(unsigned), cudaMemcpyDeviceToHost);
    if (S > 0)
    {
        if (g_ws.sorted_cap < S)
        {
            cudaFree(g_ws.sorted);
            cudaMalloc(&g_ws.sorted, (size_t)S * sizeof(unsigned));
            g_ws.sorted_cap = S;
        }
        size_t bytes = 0;
        cub::DeviceRadixSort::SortKeysDescending(nullptr, bytes, g_ws.log, g_ws.sorted, (int)S);
        if (g_ws.sort_temp_bytes < bytes)
        {
            cudaFree(g_ws.sort_temp);
            cudaMalloc(&g_ws.sort_temp, bytes);
            g_ws.sort_temp_bytes = bytes;
        }
        cub::DeviceRadixSort::SortKeysDescending(g_ws.sort_temp, bytes, g_ws.log, g_ws.sorted, (int)S);
    }
    emit_kernel<<<(k + 255) / 256, 256>>>(g_ws.sorted, S, a.st, output, k);
}

// Survivor count and final threshold of the last solve() call (for analysis)
static void last_stats(unsigned& survivors, float& tau)
{
    State st;
    cudaMemcpy(&st, g_ws.acc.st, sizeof(State), cudaMemcpyDeviceToHost);
    std::vector<unsigned> c8(256);
    cudaMemcpy(c8.data(), g_ws.acc.c8, 256 * sizeof(unsigned), cudaMemcpyDeviceToHost);
    survivors = std::accumulate(c8.begin(), c8.end(), 0u); // each survivor is counted once in c8
    unsigned u = st.tau;
    u = (u & 0x80000000u) ? (u ^ 0x80000000u) : ~u;
    std::memcpy(&tau, &u, sizeof(float));
}

// ============================================================================
// Comprehensive Validation & Performance Benchmark Suite
// (same tests and benchmark as top_k_selection.cu, plus extra cases below)
// ============================================================================

bool run_test(const std::string& test_name, const std::vector<float>& h_input, int k, int offset = 0)
{
    int N = static_cast<int>(h_input.size());

    // 1. CPU Reference (Sort in descending order)
    std::vector<float> cpu_sorted = h_input;
    std::sort(cpu_sorted.begin(), cpu_sorted.end(), std::greater<float>());
    std::vector<float> expected(cpu_sorted.begin(), cpu_sorted.begin() + k);

    // 2. GPU Run: Custom Top-K (offset > 0 tests a misaligned input pointer)
    float* d_buf = nullptr;
    float* d_output = nullptr;
    cudaMalloc(&d_buf, (N + offset) * sizeof(float));
    cudaMalloc(&d_output, k * sizeof(float));
    float* d_input = d_buf + offset;

    cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    solve(d_input, d_output, N, k);
    cudaDeviceSynchronize();

    std::vector<float> actual(k);
    cudaMemcpy(actual.data(), d_output, k * sizeof(float), cudaMemcpyDeviceToHost);

    // 3. Verification
    bool passed = true;
    for (int i = 0; i < k; ++i)
    {
        if (expected[i] != actual[i] && std::fabs(expected[i] - actual[i]) > 1e-5f)
        {
            passed = false;
            std::cerr << "FAIL [" << test_name << "] at index " << i
                      << ": Expected " << expected[i] << ", Got " << actual[i] << std::endl;
            break;
        }
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        passed = false;
        std::cerr << "FAIL [" << test_name << "] CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    if (passed)
    {
        std::cout << "  [PASS] " << test_name << " (N=" << N << ", k=" << k << ")" << std::endl;
    }

    cudaFree(d_buf);
    cudaFree(d_output);
    return passed;
}

enum class Dist { Uniform, Gaussian, SortedAscending };

void benchmark_performance(int N, int k, int num_iters = 10, Dist dist = Dist::Uniform)
{
    const char* dist_name = dist == Dist::Uniform ? "uniform [-1000, 1000]"
                          : dist == Dist::Gaussian ? "gaussian N(0, 1)"
                          : "sorted ascending";
    std::cout << "\n------------------------------------------------------------" << std::endl;
    std::cout << " Benchmark: N = " << N << ", K = " << k
              << " (" << (N * sizeof(float)) / (1024.0 * 1024.0) << " MB input, " << dist_name << ")" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    // Generate input data
    std::vector<float> h_input(N);
    std::mt19937 gen(42);
    if (dist == Dist::Gaussian)
    {
        std::normal_distribution<float> dis(0.0f, 1.0f);
        for (int i = 0; i < N; ++i) h_input[i] = dis(gen);
    }
    else
    {
        std::uniform_real_distribution<float> dis(-1000.0f, 1000.0f);
        for (int i = 0; i < N; ++i) h_input[i] = dis(gen);
        if (dist == Dist::SortedAscending) std::sort(h_input.begin(), h_input.end());
    }

    // Allocate GPU buffers
    float *d_input, *d_out_custom, *d_out_cub_topk, *d_out_cub_sort;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_out_custom, k * sizeof(float));
    cudaMalloc(&d_out_cub_topk, k * sizeof(float));
    cudaMalloc(&d_out_cub_sort, N * sizeof(float));

    cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // Setup cub::DeviceTopK
    auto requirements = cuda::execution::require(
        cuda::execution::determinism::not_guaranteed,
        cuda::execution::output_ordering::unsorted
    );
    auto env = cuda::std::execution::env{requirements};
    void* d_topk_temp = nullptr;
    size_t topk_temp_bytes = 0;
    cub::DeviceTopK::MaxKeys(d_topk_temp, topk_temp_bytes, d_input, d_out_cub_topk, N, k, env);
    cudaMalloc(&d_topk_temp, topk_temp_bytes);

    // Setup cub::DeviceRadixSort
    void* d_sort_temp = nullptr;
    size_t sort_temp_bytes = 0;
    cub::DeviceRadixSort::SortKeysDescending(d_sort_temp, sort_temp_bytes, d_input, d_out_cub_sort, N);
    cudaMalloc(&d_sort_temp, sort_temp_bytes);

    // CUDA timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up runs
    for (int i = 0; i < 3; ++i)
    {
        solve(d_input, d_out_custom, N, k);
        cub::DeviceTopK::MaxKeys(d_topk_temp, topk_temp_bytes, d_input, d_out_cub_topk, N, k, env);
        cub::DeviceRadixSort::SortKeysDescending(d_sort_temp, sort_temp_bytes, d_input, d_out_cub_sort, N);
    }
    cudaDeviceSynchronize();

    // 1. Benchmark Custom Top-K (single-pass monotone accumulator)
    float time_custom = 0.0f;
    {
        cudaEventRecord(start);
        for (int i = 0; i < num_iters; ++i)
        {
            solve(d_input, d_out_custom, N, k);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float total_ms = 0;
        cudaEventElapsedTime(&total_ms, start, stop);
        time_custom = total_ms / num_iters;
    }
    unsigned survivors = 0;
    float final_tau = 0.0f;
    last_stats(survivors, final_tau);

    // 2. Benchmark CCCL cub::DeviceTopK::MaxKeys
    float time_cub_topk = 0.0f;
    {
        cudaEventRecord(start);
        for (int i = 0; i < num_iters; ++i)
        {
            cub::DeviceTopK::MaxKeys(d_topk_temp, topk_temp_bytes, d_input, d_out_cub_topk, N, k, env);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float total_ms = 0;
        cudaEventElapsedTime(&total_ms, start, stop);
        time_cub_topk = total_ms / num_iters;
    }

    // 3. Benchmark CUB Full Radix Sort (cub::DeviceRadixSort::SortKeysDescending)
    float time_cub_sort = 0.0f;
    {
        cudaEventRecord(start);
        for (int i = 0; i < num_iters; ++i)
        {
            cub::DeviceRadixSort::SortKeysDescending(d_sort_temp, sort_temp_bytes, d_input, d_out_cub_sort, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float total_ms = 0;
        cudaEventElapsedTime(&total_ms, start, stop);
        time_cub_sort = total_ms / num_iters;
    }

    // Verify CCCL cub::DeviceTopK correctness
    std::vector<float> h_cub_topk(k);
    cudaMemcpy(h_cub_topk.data(), d_out_cub_topk, k * sizeof(float), cudaMemcpyDeviceToHost);
    std::sort(h_cub_topk.begin(), h_cub_topk.end(), std::greater<float>());

    std::vector<float> h_custom(k);
    cudaMemcpy(h_custom.data(), d_out_custom, k * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<float> cpu_ref = h_input;
    std::sort(cpu_ref.begin(), cpu_ref.end(), std::greater<float>());

    bool custom_correct = true;
    bool cub_correct = true;
    for (int i = 0; i < k; ++i)
    {
        if (std::fabs(h_custom[i] - cpu_ref[i]) > 1e-5f) custom_correct = false;
        if (std::fabs(h_cub_topk[i] - cpu_ref[i]) > 1e-5f) cub_correct = false;
    }

    // Print Performance Table
    std::cout << std::fixed << std::setprecision(4);
    std::cout << " Method                        | Time (ms) | Speedup vs Full Sort | Correct?" << std::endl;
    std::cout << "-------------------------------|-----------|----------------------|---------" << std::endl;
    std::cout << " Monotone Top-K (1-pass)       | " << std::setw(8) << time_custom
              << "  | " << std::setw(18) << (time_cub_sort / time_custom) << "x | "
              << (custom_correct ? "PASS" : "FAIL") << std::endl;
    std::cout << " CCCL cub::DeviceTopK::MaxKeys | " << std::setw(8) << time_cub_topk
              << "  | " << std::setw(18) << (time_cub_sort / time_cub_topk) << "x | "
              << (cub_correct ? "PASS" : "FAIL") << std::endl;
    std::cout << " CUB Full Radix Sort (Baseline)| " << std::setw(8) << time_cub_sort
              << "  | " << std::setw(18) << 1.0 << "x | PASS" << std::endl;
    std::cout << " Survivors written to log: " << survivors
              << " (" << std::setprecision(3) << 100.0 * survivors / N << "% of N), final tau = "
              << std::setprecision(6) << final_tau << std::endl;

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_out_custom);
    cudaFree(d_out_cub_topk);
    cudaFree(d_out_cub_sort);
    cudaFree(d_topk_temp);
    cudaFree(d_sort_temp);
}

int main()
{
    std::cout << "============================================================" << std::endl;
    std::cout << "   CUDA Top-K Correctness Validation Suite                  " << std::endl;
    std::cout << "============================================================" << std::endl;

    bool all_passed = true;

    // Test 1: Small Array
    {
        std::vector<float> data = {3.5f, 1.2f, 9.8f, -2.1f, 4.4f, 7.0f, -0.5f, 10.1f, 0.0f, 6.3f};
        all_passed &= run_test("Small Array", data, 5);
    }

    // Test 2: Standard Uniform Floats
    {
        int N = 100000;
        int k = 50;
        std::vector<float> data(N);
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        for (int i = 0; i < N; ++i) data[i] = dis(gen);
        all_passed &= run_test("Uniform [0, 1]", data, k);
    }

    // Test 3: Mixed Positive, Negative, Zero, and Large Range
    {
        int N = 200000;
        int k = 100;
        std::vector<float> data(N);
        std::mt19937 gen(123);
        std::uniform_real_distribution<float> dis(-100000.0f, 100000.0f);
        for (int i = 0; i < N; ++i) data[i] = dis(gen);
        all_passed &= run_test("Mixed Signs [-100k, 100k]", data, k);
    }

    // Test 4: Heavy Duplicates and Ties (small discrete alphabet)
    {
        int N = 50000;
        int k = 40;
        std::vector<float> choices = {-10.0f, -2.5f, 0.0f, 3.14f, 5.0f, 7.7f, 9.9f, 42.0f};
        std::vector<float> data(N);
        std::mt19937 gen(777);
        std::uniform_int_distribution<int> dis(0, choices.size() - 1);
        for (int i = 0; i < N; ++i) data[i] = choices[dis(gen)];
        all_passed &= run_test("Heavy Duplicates / Ties", data, k);
    }

    // Test 5: All Elements Identical
    {
        int N = 10000;
        int k = 20;
        std::vector<float> data(N, 7.777f);
        all_passed &= run_test("All Elements Identical", data, k);
    }

    // Test 6: Single Top Element (k = 1)
    {
        int N = 250000;
        int k = 1;
        std::vector<float> data(N);
        std::mt19937 gen(999);
        std::uniform_real_distribution<float> dis(-500.0f, 500.0f);
        for (int i = 0; i < N; ++i) data[i] = dis(gen);
        all_passed &= run_test("k = 1 (Maximum)", data, k);
    }

    // Test 7: Full Sort (k = N)
    {
        int N = 256;
        int k = 256;
        std::vector<float> data(N);
        std::mt19937 gen(2024);
        std::uniform_real_distribution<float> dis(-10.0f, 10.0f);
        for (int i = 0; i < N; ++i) data[i] = dis(gen);
        all_passed &= run_test("Full Sort (k = N = 256)", data, k);
    }

    // ---- Extra cases for the paths this design adds ----
    std::cout << "  -- extra cases --" << std::endl;
    {
        int N = 1000000;
        std::vector<float> data(N);
        std::mt19937 gen(5);
        std::uniform_real_distribution<float> dis(-1000.0f, 1000.0f);
        for (int i = 0; i < N; ++i) data[i] = dis(gen);
        all_passed &= run_test("Uniform, k = 2000 (largest fast-path k)", data, 2000);
        all_passed &= run_test("Uniform, k = 20000 (host sort path)", data, 20000);
        all_passed &= run_test("Misaligned input pointer", data, 100, 1);

        std::vector<float> asc = data;
        std::sort(asc.begin(), asc.end());
        all_passed &= run_test("Sorted ascending (adversarial order)", asc, 100);
        all_passed &= run_test("Sorted ascending, k = 1500 (warm-up path)", asc, 1500);
        std::vector<float> desc(asc.rbegin(), asc.rend());
        all_passed &= run_test("Sorted descending", desc, 100);
    }
    {
        int N = 1000003; // not a multiple of 4 or of the tile size
        std::vector<float> data(N);
        std::mt19937 gen(6);
        std::normal_distribution<float> dis(0.0f, 1.0f);
        for (int i = 0; i < N; ++i) data[i] = dis(gen);
        all_passed &= run_test("Gaussian, odd N", data, 100);
    }
    {
        int N = 1000000;
        std::vector<float> data(N, 7.777f);
        all_passed &= run_test("All identical, large N", data, 100);
    }
    {
        // Many copies of the k-th value: 300 values > 5, then 100k copies of 5
        int N = 1000000;
        std::vector<float> data(N);
        std::mt19937 gen(8);
        std::uniform_real_distribution<float> dis(-1000.0f, 4.0f);
        for (int i = 0; i < N; ++i) data[i] = dis(gen);
        for (int i = 0; i < 100000; ++i) data[i] = 5.0f;
        for (int i = 100000; i < 100300; ++i) data[i] = 6.0f + (i % 97);
        std::shuffle(data.begin(), data.end(), gen);
        all_passed &= run_test("Many copies of the k-th value", data, 1000);
    }
    {
        int N = 5000;
        std::vector<float> data(N);
        std::mt19937 gen(9);
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        for (int i = 0; i < N; ++i) data[i] = dis(gen);
        all_passed &= run_test("k = N = 5000", data, N);
    }

    std::cout << "\nValidation Status: " << (all_passed ? "ALL TESTS PASSED!" : "SOME TESTS FAILED!") << std::endl;

    // ============================================================
    // Performance Comparisons
    // ============================================================
    std::cout << "\n============================================================" << std::endl;
    std::cout << "   Performance Comparison: Custom vs CCCL TopK vs Full Sort " << std::endl;
    std::cout << "============================================================" << std::endl;

    // Same three benchmarks as top_k_selection.cu
    benchmark_performance(1000000, 32, 10);
    benchmark_performance(10000000, 100, 10);
    benchmark_performance(50000000, 100, 5);

    // Extra: other distributions / large k
    benchmark_performance(50000000, 100, 5, Dist::Gaussian);
    benchmark_performance(50000000, 100, 5, Dist::SortedAscending);
    benchmark_performance(50000000, 10000, 5);

    return all_passed ? 0 : 1;
}
