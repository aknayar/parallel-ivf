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
void KMeans<DistanceKernel>::train(size_t n, const float *data, float *centroids) {
    init_centroids(n, data, centroids);
    learn_centroids(n, data, centroids);
}

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

template <DistanceKernel DistanceKernel>
void KMeans<DistanceKernel>::learn_centroids(size_t n, const float *data,
                                             float *centroids) {
    std::vector<std::vector<size_t>> assignments(k);

    bool converged = false;
    while (!converged) {
        // Clear assignments
        for (size_t i = 0; i < k; i++) {
            assignments[i].clear();
        }

        // Assign points to closest centroids
        for (size_t i = 0; i < n; i++) {
            const float *point = data + i * d;
            size_t closest_centroid_idx = 0;
            float min_distance = distance<DistanceKernel>(point, centroids, d);
            for (size_t j = 0; j < k; j++) {
                float curr_distance = distance<DistanceKernel>(point, centroids + j * d, d);
                if (curr_distance < min_distance) {
                    min_distance = curr_distance;
                    closest_centroid_idx = j;
                }
            }
            assignments[closest_centroid_idx].push_back(i);
        }

        // Update centroids
        std::vector<float> new_centroids(k * d);
        for (size_t i = 0; i < k; i++) {
            const std::vector<size_t> &assignment = assignments[i];
            if (assignment.empty()) {
                continue;
            }
            float *centroid = new_centroids.data() + i * d;
            for (size_t j = 0; j < d; j++) {
                float sum = 0.0f;
                for (size_t p : assignment) {
                    sum += data[p * d + j];
                }
                centroid[j] = sum / assignment.size();
            }
        }

        // Check convergence
        bool converged = true;
        for (size_t i = 0; i < k; i++) {
            const float *centroid = centroids + i * d;
            const float *new_centroid = new_centroids.data() + i * d;
            for (size_t j = 0; j < d; j++) {
                if (centroid[j] != new_centroid[j]) {
                    converged = false;
                    break;
                }
            }
        }
    }
}