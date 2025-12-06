/**
 * @file distances.cpp
 * @brief Distance (L2 squared) kernel implementations
 * @date 11-18-2025
 */

#include "distances.h"
#include <immintrin.h>
#include <cstddef>
#include <vector>
#include <iostream>

float _simd_horizontal_sum(__m256 dist, float rem_dist){
    auto tmp = _mm256_hadd_ps(dist, dist);
    __m128 tmp2 = _mm_add_ps(_mm256_extractf128_ps(tmp, 0), _mm256_extractf128_ps(tmp, 1));
    __m128 tmp3 = _mm_hadd_ps(tmp2, tmp2);
    float ret_dist = 0.0f;
    ret_dist += _mm_cvtss_f32(tmp3);
    ret_dist += rem_dist;
    return ret_dist;
}


// Serial distance kernel (L2 squared)
float distance_scalar(const float *a, const float *b, size_t d) {
    float distance = 0.0f;
    for (size_t i = 0; i < d; i++) {
        float diff = a[i] - b[i];
        distance += diff * diff;
    }
    return distance;
}

// SIMD-ized distance kernel (L2 squared)
float distance_simd(const float *a, const float *b, size_t d) {
    __m256 dist = _mm256_set1_ps(0.0f);
    size_t SIMD_LANE_LENGTH = 8;
    auto N = d / SIMD_LANE_LENGTH;
    auto rem = d % SIMD_LANE_LENGTH;

    for(size_t i =0; i < N*SIMD_LANE_LENGTH; i += SIMD_LANE_LENGTH){
        __m256 a8 = _mm256_loadu_ps(&a[i]);
        __m256 b8 = _mm256_loadu_ps(&b[i]);
        __m256 diff = _mm256_sub_ps(a8, b8);
        auto mult = _mm256_mul_ps(diff, diff);
        dist = _mm256_add_ps(dist, mult);
    }

    float rem_dist = 0.0f;
    for (size_t i = N*SIMD_LANE_LENGTH; i < d; i++) {
        float diff = a[i] - b[i];
        rem_dist += diff * diff;
    }

    auto tmp = _mm256_hadd_ps(dist, dist);
    __m128 tmp2 = _mm_add_ps(_mm256_extractf128_ps(tmp, 0), _mm256_extractf128_ps(tmp, 1));
    __m128 tmp3 = _mm_hadd_ps(tmp2, tmp2);
    float ret_dist = 0.0f;
    ret_dist += _mm_cvtss_f32(tmp3);
    ret_dist += rem_dist;
    
    return ret_dist;
}

float* distance_omp_simd(const float *a, const float *b, size_t d, size_t n){
    float* distances = new float[n]();
    #pragma omp parallel for
    for(size_t i = 0; i < n; i++){
        distances[i] = distance_simd(a, b+d*i, d);
    }

    return distances;
}



float* distance_cache(const float *a, const float *b, size_t d, size_t n){
    size_t CACHE_LINE_SIZE = 16;
    float* distances = new float[n]();
    if (d <= CACHE_LINE_SIZE){
        for (size_t i = 0; i < n; i++){
            distances[i] = distance_scalar(a, b+d*i,d);
        }
        return distances;
    } else {
        
        auto N = d / CACHE_LINE_SIZE;
        auto rem = d % CACHE_LINE_SIZE;

        for (size_t i =0; i < N*CACHE_LINE_SIZE; i += CACHE_LINE_SIZE){ //do sum for each 16 byte block of floats
            for (size_t j = 0; j < n; j++){ //do sum for all floats in b
                for (size_t k = i; k < i+CACHE_LINE_SIZE; k++){ //distance calc the current 16 floats and update distances
                    float diff = a[k] - b[d*j+k]; //get the kth float of a, and the kth float of the jth float vec of b
                    distances[j] += diff * diff;
                }
            }
        }
        
        for (size_t j = 0; j < n; j++){ //do sum for all floats in b
            for (size_t k = N*CACHE_LINE_SIZE; k < d; k++){ //distance calc the current 16 floats and update distances
                float diff = a[k] - b[d*j+k]; //get the kth float of a, and the kth float of the jth float vec of b
                distances[j] += diff * diff;
            }
        }

        return distances;
        

    }
   
}

float* distance_cache_simd(const float *a, const float *b, size_t d, size_t n){
    __m256 dist = _mm256_set1_ps(0.0f);
    size_t SIMD_LANE_LENGTH = 8;
    auto N = d / SIMD_LANE_LENGTH;
    auto rem = d % SIMD_LANE_LENGTH;
    size_t TILE = 16;
    size_t VEC_BLOCK = 4;
    float* distances = new float[n]();
    for (size_t i = 0; i < d; i += TILE) {
        size_t tile_end = std::min(i + TILE, d);
        size_t fullBlocks = (n / VEC_BLOCK) * VEC_BLOCK;

        for (size_t jBlock = 0; jBlock < fullBlocks; jBlock += VEC_BLOCK) {
            const float* b0 = b + (jBlock + 0) * d;
            const float* b1 = b + (jBlock + 1) * d;
            const float* b2 = b + (jBlock + 2) * d;
            const float* b3 = b + (jBlock + 3) * d;

            __m256 acc0 = _mm256_set1_ps(0.0f);
            __m256 acc1 = _mm256_set1_ps(0.0f);
            __m256 acc2 = _mm256_set1_ps(0.0f);
            __m256 acc3 = _mm256_set1_ps(0.0f);

            size_t k = i;
            for (; k + 7 < tile_end; k += 8) {
                __m256 a8 = _mm256_loadu_ps(&a[k]);

                __m256 x0 = _mm256_loadu_ps(&b0[k]);
                __m256 x1 = _mm256_loadu_ps(&b1[k]);
                __m256 x2 = _mm256_loadu_ps(&b2[k]);
                __m256 x3 = _mm256_loadu_ps(&b3[k]);

                __m256 d0 = _mm256_sub_ps(a8, x0);
                __m256 d1 = _mm256_sub_ps(a8, x1);
                __m256 d2 = _mm256_sub_ps(a8, x2);
                __m256 d3 = _mm256_sub_ps(a8, x3);

                __m256 s0 = _mm256_mul_ps(d0, d0);
                __m256 s1 = _mm256_mul_ps(d1, d1);
                __m256 s2 = _mm256_mul_ps(d2, d2);
                __m256 s3 = _mm256_mul_ps(d3, d3);

                acc0 = _mm256_add_ps(acc0,s0);
                acc1 = _mm256_add_ps(acc1,s1);
                acc2 = _mm256_add_ps(acc2,s2);
                acc3 = _mm256_add_ps(acc3,s3);
            }

            float rem0, rem1, rem2, rem3;
            rem0 = rem1 = rem2 = rem3 = 0.0f;
            
            for (; k < tile_end; k++) {
                float ak = a[k];
                float d0 = ak - b0[k];
                float d1 = ak - b1[k];
                float d2 = ak - b2[k];
                float d3 = ak - b3[k];

                rem0 += d0 * d0;
                rem1 += d1 * d1;
                rem2 += d2 * d2;
                rem3 += d3 * d3;
            }

            distances[jBlock + 0] += _simd_horizontal_sum(acc0, rem0);
            distances[jBlock + 1] += _simd_horizontal_sum(acc1, rem1);
            distances[jBlock + 2] += _simd_horizontal_sum(acc2, rem2);
            distances[jBlock + 3] += _simd_horizontal_sum(acc3, rem3);
        }
        for (size_t j = fullBlocks; j < n; j++) {
            const float* bj = b + j * d;

            float acc = 0.0f;
            size_t k = i;
            for (; k < tile_end; k++) {
                float diff = a[k] - bj[k];
                acc += diff * diff;
            }
            distances[j] += acc;
        }
    }

    return distances;

    
   
}



template <DistanceKernel kernel>
float distance(const float *a, const float *b, size_t d) {
    if constexpr (kernel == DistanceKernel::SCALAR) {
        return distance_scalar(a, b, d);
    } else if constexpr (kernel == DistanceKernel::SIMD) {
        return distance_simd(a, b, d);
    } 
    else if constexpr (kernel == DistanceKernel::CACHE){
        return distance_scalar(a, b, d);
    }
    else if constexpr (kernel == DistanceKernel::CACHESIMD){
        return distance_scalar(a, b, d);
    } else if constexpr (kernel == DistanceKernel::OMPSIMD){
        return distance_scalar(a, b, d);
    }
    else {
        static_assert(false, "Invalid distance kernel");
    }
}

template <DistanceKernel kernel>
float* distance(const float *a, const float *b, size_t d, size_t n) {
    if constexpr (kernel==DistanceKernel::CACHE){
        return distance_cache(a,b,d,n);
    } else if constexpr (kernel==DistanceKernel::CACHESIMD){
        return distance_cache_simd(a,b,d,n);
    } else if constexpr (kernel==DistanceKernel::OMPSIMD){
        return distance_omp_simd(a,b,d,n);
    }
    else{
        static_assert(false, "Invalid distance kernel");
    }
}

// Explicit template instantiations
template float distance<DistanceKernel::SCALAR>(const float *, const float *, size_t);
template float distance<DistanceKernel::SIMD>(const float *, const float *, size_t);
template float distance<DistanceKernel::CACHE>(const float *, const float *, size_t);
template float distance<DistanceKernel::CACHESIMD>(const float *, const float *, size_t);
template float distance<DistanceKernel::OMPSIMD>(const float *, const float *, size_t);
template float* distance<DistanceKernel::CACHE>(const float *, const float *, size_t, size_t);
template float* distance<DistanceKernel::CACHESIMD>(const float *, const float *, size_t, size_t);
template float* distance<DistanceKernel::OMPSIMD>(const float *, const float *, size_t, size_t);
