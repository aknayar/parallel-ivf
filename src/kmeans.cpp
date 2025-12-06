/**
 * @file kmeans.cpp
 * @brief K-means implementation
 * @date 11-18-2025
 */

#include "kmeans.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>

#define RANDOM_SEED 5

template <DistanceKernel DistanceKernel, ParallelType ParallelType>
void KMeans<DistanceKernel, ParallelType>::train(size_t n, const float *data,
                                                 float *centroids,
                                                 size_t nlist) {
    init_centroids(n, data, centroids, nlist);
    learn_centroids(n, data, centroids, nlist);
}

template <DistanceKernel DistanceKernel, ParallelType ParallelType>
void KMeans<DistanceKernel, ParallelType>::init_centroids(size_t n,
                                                          const float *data,
                                                          float *centroids,
                                                          size_t nlist) {
    // Following https://en.wikipedia.org/wiki/K-means%2B%2B
    std::mt19937 gen(RANDOM_SEED);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    size_t point_bytes = d * sizeof(float);

    // 1: Randomly choose the first cluster
    std::uniform_int_distribution<> dis(0, n - 1);
    size_t c_idx = dis(gen);
    memcpy(centroids, data + c_idx * d, point_bytes);
    size_t num_c = 1;

    // 2: Choose the remaining centroids
    while (num_c < k) {
        std::vector<float> dists(n);

#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            const float *pt = data + i * d;

            if constexpr (DistanceKernel == DistanceKernel::CACHE ||
                          DistanceKernel == DistanceKernel::CACHESIMD) {
                float *distances =
                    distance<DistanceKernel>(pt, centroids, d, num_c);
                float min_dist = distances[0];
                for (size_t j = 0; j < num_c; j++) {
                    min_dist = std::min(min_dist, distances[j]);
                }
                dists[i] = min_dist;

            } else {
                float min_dist = distance<DistanceKernel>(pt, centroids, d);

                for (size_t j = 0; j < num_c; j++) {
                    float *cent = centroids + j * d;
                    min_dist = std::min(min_dist,
                                        distance<DistanceKernel>(pt, cent, d));
                }
                dists[i] = min_dist;
            }
        }

        float total = std::accumulate(dists.begin(), dists.end(), 0.0f);
        float thresh = total * uniform(gen);
        float cumul = 0.0f;

        for (size_t i = 0; i < n; i++) {
            const float *pt = data + i * d;
            cumul += dists[i];
            if (cumul > thresh) {
                memcpy(centroids + num_c * d, pt, point_bytes);
                num_c++;
                break;
            }
        }
    }
}

template <DistanceKernel DistanceKernel, ParallelType ParallelType>
void KMeans<DistanceKernel, ParallelType>::learn_centroids(size_t n,
                                                           const float *data,
                                                           float *centroids,
                                                           size_t nlist) {
    std::vector<std::vector<size_t>> assign(k);

    bool converged = false;
    while (!converged) {
        // Clear assign
        for (size_t i = 0; i < k; i++) {
            assign[i].clear();
        }

        std::vector<size_t> assignments(n);

        // Assign points to closest centroids
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            if constexpr (DistanceKernel == DistanceKernel::CACHE ||
                          DistanceKernel == DistanceKernel::CACHESIMD) {
                const float *pt = data + i * d;
                size_t c_idx = 0;
                float *distances =
                    distance<DistanceKernel>(pt, centroids, d, k);
                float min_dist = distances[0];
                for (size_t j = 0; j < k; j++) {
                    auto curr_dist = distances[j];
                    if (curr_dist < min_dist) {
                        min_dist = curr_dist;
                        c_idx = j;
                    }
                }
                assignments[i] = c_idx;
            } else {
                const float *pt = data + i * d;
                size_t c_idx = 0;
                float min_dist = distance<DistanceKernel>(pt, centroids, d);
                for (size_t j = 0; j < k; j++) {
                    float curr_dist =
                        distance<DistanceKernel>(pt, centroids + j * d, d);
                    if (curr_dist < min_dist) {
                        min_dist = curr_dist;
                        c_idx = j;
                    }
                }
                assignments[i] = c_idx;
            }
        }

        for (size_t i = 0; i < n; i++) {
            assign[assignments[i]].push_back(i);
        }

        // Update centroids
        std::vector<float> new_centroids(k * d);
#pragma omp parallel for
        for (size_t i = 0; i < k; i++) {
            const std::vector<size_t> &assignment = assign[i];
            if (assignment.empty()) {
                continue;
            }
            float *cent = new_centroids.data() + i * d;
            for (size_t j = 0; j < d; j++) {
                float sum = 0.0f;
                for (size_t p : assignment) {
                    sum += data[p * d + j];
                }
                cent[j] = sum / assignment.size();
            }
        }

        // Check convergence
        converged = true;
        for (size_t i = 0; i < k; i++) {
            const float *cent = centroids + i * d;
            const float *new_cent = new_centroids.data() + i * d;
            for (size_t j = 0; j < d; j++) {
                if (cent[j] != new_cent[j]) {
                    converged = false;
                    break;
                }
            }
            if (!converged)
                break;
        }

        // Copy new centroids if not converged
        if (!converged) {
            memcpy(centroids, new_centroids.data(), k * d * sizeof(float));
        }
    }
}

// Explicit template instantiations
template class KMeans<DistanceKernel::SCALAR, ParallelType::SERIAL>;
template class KMeans<DistanceKernel::SIMD, ParallelType::SERIAL>;
template class KMeans<DistanceKernel::CACHE, ParallelType::SERIAL>;
template class KMeans<DistanceKernel::CACHESIMD, ParallelType::SERIAL>;
template class KMeans<DistanceKernel::SCALAR, ParallelType::QUERY_PARALLEL>;
template class KMeans<DistanceKernel::SIMD, ParallelType::QUERY_PARALLEL>;
template class KMeans<DistanceKernel::CACHE, ParallelType::QUERY_PARALLEL>;
template class KMeans<DistanceKernel::CACHESIMD, ParallelType::QUERY_PARALLEL>;
template class KMeans<DistanceKernel::SCALAR, ParallelType::CANDIDATE_PARALLEL>;
template class KMeans<DistanceKernel::SIMD, ParallelType::CANDIDATE_PARALLEL>;
template class KMeans<DistanceKernel::CACHE, ParallelType::CANDIDATE_PARALLEL>;
template class KMeans<DistanceKernel::CACHESIMD,
                      ParallelType::CANDIDATE_PARALLEL>;
