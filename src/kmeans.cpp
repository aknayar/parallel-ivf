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
void KMeans<DistanceKernel>::init_centroids(size_t n, const float *data,
                                    float *centroids) {
    // Following https://en.wikipedia.org/wiki/K-means%2B%2B
    std::mt19937 gen(RANDOM_SEED);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    size_t point_bytes = d * sizeof(float);

    // 1: Randomly choose the first cluster
    std::uniform_int_distribution<> dis(0, n - 1);
    size_t first_centroid_idx = dis(gen);
    memcpy(centroids, data + first_centroid_idx * d, point_bytes);
    size_t num_clusters = 1;

    // 2: Choose the remaining centroids
    while (num_clusters < k) {
        std::vector<float> distances(n);

        for (size_t i = 0; i < n; i++) {
            const float *point = data + i * d;
            float min_distance = distance<DistanceKernel>(point, centroids, d);

            for (size_t j = 0; j < num_clusters; j++) {
                min_distance =
                    std::min(min_distance,
                             distance<DistanceKernel>(point, centroids + j * d, d));
            }
            distances[i] = min_distance;
        }

        float total = std::accumulate(distances.begin(), distances.end(), 0.0f);
        float threshold = total * uniform(gen);
        float cumulative = 0.0f;

        for (size_t i = 0; i < n; i++) {
            const float *point = data + i * d;
            cumulative += distances[i];
            if (cumulative > threshold) {
                memcpy(centroids + num_clusters * d, point, point_bytes);
                num_clusters++;
                break;
            }
        }
    }
}
