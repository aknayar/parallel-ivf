/**
 * @file kmeans.h
 * @brief K-means declarations
 * @date 11-18-2025
 */

#ifndef KMEANS_H
#define KMEANS_H

#include <cstddef>
#include <vector>

#include "distances.h"

template <DistanceKernel DistanceKernel> struct KMeans {
  public:
    KMeans(size_t d, size_t k) : d(d), k(k) {}

    void train(size_t n, const float *data, float *centroids, size_t nlist);

  private:
    size_t d, k;

    void init_centroids(size_t n, const float *data, float *centroids,
                        size_t nlist);

    void learn_centroids(size_t n, const float *data, float *centroids,
                         size_t nlist);
};

#endif // KMEANS_H
