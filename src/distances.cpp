/**
 * @file distances.cpp
 * @brief Distance (L2 squared) kernel implementations
 * @date 11-18-2025
 */

#include "distances.h"

#include <cstddef>
#include <vector>

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
