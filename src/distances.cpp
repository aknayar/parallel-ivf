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
    size_t CACHE_LINE_SIZE = 16;
    float* distances = new float[n]();
    if (d <= CACHE_LINE_SIZE){
        for (size_t i = 0; i < n; i++){
            distances[i] = distance_simd(a, b+d*i,d);
        }
        return distances;
    } else {
        
        auto N = d / CACHE_LINE_SIZE;
        auto rem = d % CACHE_LINE_SIZE;

        for (size_t i =0; i < N*CACHE_LINE_SIZE; i += CACHE_LINE_SIZE){ //do sum for each 16 byte block of floats
            for (size_t j = 0; j < n; j++){ //do sum for all floats in b
                auto dist = distance_simd(a+i, b + d*j +i, CACHE_LINE_SIZE); //do the SIMD-ized distance comp for 16 bytes at a time
                distances[j]+=dist;
            }
        }
        
        for (size_t j = 0; j < n; j++){ //do sum for all floats in b
            auto simd_d = d - N * CACHE_LINE_SIZE;
            auto k = N*CACHE_LINE_SIZE;
            auto dist = distance_simd(a+k, b + d*j +k, simd_d); //do the SIMD-ized distance comp for 16 bytes at a time
            distances[j]+=dist;
        }

        return distances;
        

    }
   
}

template <typename T>
void debug_type() {
    std::cout << "Dumping type: " << typeid(T).name() << std::endl;
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
    }
    else {
        debug_type<kernel>();
        static_assert(false, "Invalid distance kernel");
    }
}

template <DistanceKernel kernel>
float* distance(const float *a, const float *b, size_t d, size_t n) {
    if constexpr (kernel==DistanceKernel::CACHE){
        return distance_cache(a,b,d,n);
    } else if constexpr (kernel==DistanceKernel::CACHESIMD){
        return distance_cache_simd(a,b,d,n);
    } 
    else{
        debug_type<kernel>();
        static_assert(false, "Invalid distance kernel");
    }
}

// Explicit template instantiations
template float distance<DistanceKernel::SCALAR>(const float *, const float *, size_t);
template float distance<DistanceKernel::SIMD>(const float *, const float *, size_t);
template float distance<DistanceKernel::CACHE>(const float *, const float *, size_t);
template float distance<DistanceKernel::CACHESIMD>(const float *, const float *, size_t);
template float* distance<DistanceKernel::CACHE>(const float *, const float *, size_t, size_t);
template float* distance<DistanceKernel::CACHESIMD>(const float *, const float *, size_t, size_t);
