/**
 * @file IVF.h
 * @brief Inverted vector file base class
 * @date 11-18-2025
 */

#ifndef IVF_H
#define IVF_H

#include <cstddef>
#include <iostream>
#include <vector>
#include "distances.h"
#include "kmeans.h"

template <DistanceKernel DistanceKernel, ParallelType ParallelType> struct IVF {
    KMeans<DistanceKernel, ParallelType> kmeans;
    size_t d, nlist, nprobe, maxlabel = 0;
    std::vector<std::vector<float>> inv_lists; // Inverted lists
    std::vector<std::vector<size_t>> labels;   // Associated labels
    std::vector<float> centroids;              // Centroids

    IVF(size_t d, size_t nlist) : kmeans(d, nlist), d(d), nlist(nlist) {}
    ~IVF() = default;

    void train(const size_t n_train, const float *train_data);

    void build(const size_t n_train, const float *train_data);

    void add(const size_t n_add, const float *add_data);

    std::vector<std::vector<size_t>> search(const size_t n_queries,
                                            const float *queries,
                                            const size_t k,
                                            const size_t nprobe) const;

  private:
    std::vector<size_t> _top_n_centroids(const float *vector, size_t n) const;
};

#endif // IVF_H
