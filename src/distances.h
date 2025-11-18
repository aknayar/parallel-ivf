/**
 * @file distances.h
 * @brief Distance (L2 squared) kernel declarations
 * @date 11-18-2025
 */

#ifndef DISTANCES_H
#define DISTANCES_H

#include <cstddef>
#include <vector>

// Enum for distance type
enum DistanceKernel {
    NAIVE,
    SIMD,
};

// Serial distance kernel
float distance_naive(const float *a, const float *b, size_t d);

// SIMD-ized distance kernel
float distance_simd(const float *a, const float *b, size_t d);

#endif // DISTANCES_H
