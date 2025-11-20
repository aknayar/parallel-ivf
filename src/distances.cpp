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
    //std::cout<<"Starting SIMD distance" << std::endl;
    __m256 dist = _mm256_set1_ps(0.0f);
    size_t SIMD_LANE_LENGTH = 8;
    auto N = d / SIMD_LANE_LENGTH;
    auto rem = d % SIMD_LANE_LENGTH;
    // std::cout<<"Initialized stuff correctly " << "will run for "<< N << " loops \n"<< std::endl;
    for(size_t i =0; i < N*SIMD_LANE_LENGTH; i += SIMD_LANE_LENGTH){

       // std::cout<<"Loop "<<i<<" \n"<<std::endl;
        // for(size_t j = i; j < i+8; j++){
        //     std::cout<<a[i]<<" ";
        //     std::cout<<b[i]<<" ";
        // }
        // std::cout<<"\n";
        __m256 a8 = _mm256_loadu_ps(&a[i]);
        __m256 b8 = _mm256_loadu_ps(&b[i]);
        __m256 diff = _mm256_sub_ps(a8, b8);
        auto mult = _mm256_mul_ps(diff, diff);
      //  std::cout<<"4\n"<<std::endl;
        dist = _mm256_add_ps(dist, mult);
      //  std::cout<<"5\n"<<std::endl;

    }

    // std::cout<<"Added distances \n";

    float rem_dist = 0.0f;

    for (size_t i = N*SIMD_LANE_LENGTH; i < d; i++) {
        float diff = a[i] - b[i];
        rem_dist += diff * diff;
    }

    // std::cout<<"Distance rems \n" << std::endl;

    auto tmp = _mm256_hadd_ps(dist, dist);
    // std::cout<<"1 \n" << std::endl;
    __m128 tmp2 = _mm_add_ps(_mm256_extractf128_ps(tmp, 0), _mm256_extractf128_ps(tmp, 1));
    // std::cout<<"2 \n" << std::endl;
    __m128 tmp3 = _mm_hadd_ps(tmp2, tmp2);
    // std::cout<<"3 \n" << std::endl;
    float ret_dist = 0.0f;
    // std::cout<<"4 \n" << std::endl;
    ret_dist += _mm_cvtss_f32(tmp3);
    // std::cout<<"5 \n" << std::endl;
   
    ret_dist += rem_dist;
    //std::cout<<"6 \n" << std::endl;
    
    return ret_dist;

    



    // TODO: Implement SIMD version
    return distance_scalar(a, b, d); // Fallback to scalar for now
}

template <DistanceKernel kernel>
float distance(const float *a, const float *b, size_t d) {
    if constexpr (kernel == DistanceKernel::SCALAR) {
        return distance_scalar(a, b, d);
    } else if constexpr (kernel == DistanceKernel::SIMD) {
        return distance_simd(a, b, d);
    } else {
        static_assert(false, "Invalid distance kernel");
    }
}

// Explicit template instantiations
template float distance<DistanceKernel::SCALAR>(const float *, const float *, size_t);
template float distance<DistanceKernel::SIMD>(const float *, const float *, size_t);
