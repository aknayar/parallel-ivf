/**
 * @file kmeans.cpp
 * @brief K-means implementation
 * @date 11-18-2025
 */

#include "kmeans.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <random>

#define RANDOM_SEED 5

template <DistanceKernel DistanceKernel>
void KMeans<DistanceKernel>::train(size_t n, const float *data,
                                   float *centroids, size_t nlist) {
    init_centroids(n, data, centroids, nlist);
    learn_centroids(n, data, centroids, nlist);
}

template <DistanceKernel DistanceKernel>
void KMeans<DistanceKernel>::init_centroids(size_t n, const float *data,
                                            float *centroids, size_t nlist) {
    // Following https://en.wikipedia.org/wiki/K-means%2B%2B
    std::mt19937 gen(RANDOM_SEED);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    size_t point_bytes = d * sizeof(float);

    // 1: Randomly choose the first cluster
    std::uniform_int_distribution<> dis(0, n - 1);
    size_t c_idx = dis(gen);
    printf("Chose first centroid! %zu / %zu\n", c_idx, n);
    fflush(stdout);
    printf("%p %p\n", centroids, data + c_idx * d);
    fflush(stdout);
    memcpy(centroids, data + c_idx * d, point_bytes);
    size_t num_c = 1;

    printf("Tryna init centroids!\n");
    fflush(stdout);

    // 2: Choose the remaining centroids
    while (num_c < k) {
        std::vector<float> dists(n);

        for (size_t i = 0; i < n; i++) {
            const float *pt = data + i * d;
            float min_dist = distance<DistanceKernel>(pt, centroids, d);

            for (size_t j = 0; j < num_c; j++) {
                float *cent = centroids + j * d;
                min_dist =
                    std::min(min_dist, distance<DistanceKernel>(pt, cent, d));
            }
            dists[i] = min_dist;
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

template <DistanceKernel DistanceKernel>
void KMeans<DistanceKernel>::learn_centroids(size_t n, const float *data,
                                             float *centroids, size_t nlist) {
    printf("Tryna learn centroids!\n");
    fflush(stdout);
    std::vector<std::vector<size_t>> assign(k);

    bool converged = false;
    while (!converged) {
        // Clear assign
        for (size_t i = 0; i < k; i++) {
            assign[i].clear();
        }

        // Assign points to closest centroids
        for (size_t i = 0; i < n; i++) {
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
            assign[c_idx].push_back(i);
        }

        // Update centroids
        std::vector<float> new_centroids(k * d);
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
template class KMeans<DistanceKernel::SCALAR>;
template class KMeans<DistanceKernel::SIMD>;