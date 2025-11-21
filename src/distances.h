/**
 * @file distances.h
 * @brief Distance (L2 squared) kernel declarations
 * @date 11-18-2025
 */

#ifndef DISTANCES_H
#define DISTANCES_H

#include <cstddef>
#include <vector>
#include <omp.h>

// Enum for distance type
enum DistanceKernel {
    SCALAR,
    SIMD,
};

// Enum for parallel type
enum ParallelType {
    SERIAL,
    QUERY_PARALLEL,
    CANDIDATE_PARALLEL,
};

// Serial distance kernel (L2 squared)
float distance_scalar(const float *a, const float *b, size_t d);

// SIMD-ized distance kernel (L2 squared)
float distance_simd(const float *a, const float *b, size_t d);

template <DistanceKernel kernel>
float distance(const float *a, const float *b, size_t d);

#endif // DISTANCES_H
