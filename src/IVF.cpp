/**
 * @file IVF.cpp
 * @brief Serial implementation of IVF
 * @date 11-18-2025
 */

#include "IVF.h"

#include "distances.h"
#include <limits>
#include <chrono>
#include <queue>
#include <utility>

template <DistanceKernel DistanceKernel, ParallelType ParallelType>
void IVF<DistanceKernel, ParallelType>::train(const size_t n_train,
                                              const float *train_data) {
    this->centroids.resize(this->nlist * this->d);
    this->kmeans.train(n_train, train_data, this->centroids.data(),
                       this->nlist);
}

template <DistanceKernel DistanceKernel, ParallelType ParallelType>
void IVF<DistanceKernel, ParallelType>::build(const size_t n_train,
                                              const float *train_data) {
    // should include some sort of error handling if centroids not builts

    size_t num_threads = omp_get_max_threads();

    if (this->centroids.empty()) {
        return;
    }

    // now build empty vector list
    // for each point in data find closest centroid
    // add it to list
    // add label to lable list
    this->inv_lists.clear();
    this->labels.clear();
    this->inv_lists.resize(this->nlist);
    this->labels.resize(this->nlist);
    const float *cent_data = this->centroids.data();

    std::vector<size_t> labels(n_train);

#pragma omp parallel for if (num_threads > 1)
    for (size_t i = 0; i < n_train; i++) {
        const float *x = train_data + i * this->d;

        auto bciVec = this->_top_n_centroids(x, 1);
        labels[i] = bciVec[0];
    }

    // reserve space
    std::vector<size_t> counts(this->nlist);
    for (size_t i = 0; i < n_train; i++) {
        counts[labels[i]]++;
    }

    for (size_t i = 0; i < this->nlist; i++) {
        this->inv_lists[i].reserve(counts[i] * this->d);
        this->labels[i].reserve(counts[i]);
    }

    for (size_t i = 0; i < n_train; i++) {
        auto &list = this->inv_lists[labels[i]];
        list.insert(list.end(), train_data + i * this->d,
                    train_data + (i + 1) * this->d);
        this->labels[labels[i]].emplace_back(i);
    }

    this->maxlabel = n_train - 1;

    if constexpr (DistanceKernel == DistanceKernel::CACHEV2 ||
                  DistanceKernel == DistanceKernel::CACHESIMDV2) {
        // interleave dimensions
        size_t num_chunks = (this->d + CHUNK_SIZE - 1) / CHUNK_SIZE;

#pragma omp parallel for if (num_threads > 1)
        for (size_t k = 0; k < this->nlist; k++) {
            size_t CHUNK_WIDTH = CHUNK_SIZE * this->labels[k].size();
            std::vector<float> interleaved_data(this->inv_lists[k].size());
            for (size_t i = 0; i < this->labels[k].size(); i++) {
                for (size_t j = 0; j < num_chunks; j++) {
                    size_t real_chunk_size =
                        std::min<size_t>(CHUNK_SIZE, this->d - j * CHUNK_SIZE);
                    for (size_t l = 0; l < real_chunk_size; l++) {
                        interleaved_data[j * CHUNK_WIDTH + i * real_chunk_size +
                                         l] =
                            this->inv_lists[k]
                                           [i * this->d + j * CHUNK_SIZE + l];
                    }
                }
            }
            this->inv_lists[k] = interleaved_data;
        }
    }
}

template <DistanceKernel DistanceKernel, ParallelType ParallelType>
std::vector<std::vector<size_t>>
IVF<DistanceKernel, ParallelType>::search(const size_t n_queries,
                                          const float *queries, const size_t k,
                                          const size_t n_probe) const {
    std::vector<std::vector<size_t>> ret_labels;
    ret_labels.resize(n_queries);

    size_t num_threads = omp_get_max_threads();

    double total_time = 0.0;

#pragma omp                                                                    \
    parallel for if (num_threads > 1 &&                                        \
                         (ParallelType == ParallelType::QUERY_PARALLEL ||      \
                              ParallelType ==                                  \
                                      ParallelType::QUERYCANDIDATE_PARALLEL))
    for (size_t i = 0; i < n_queries; i++) {
        auto start_time = std::chrono::high_resolution_clock::now();
        const float *q = queries + i * this->d;
        auto bciVec = this->_top_n_centroids(
            q, n_probe); // get indices of nprobe closest centroids
        size_t n_probe_clamped = bciVec.size();
        std::priority_queue<std::pair<float, size_t>> pq;

        for (size_t j = 0; j < n_probe_clamped;
             j++) { // do the below for all centroids indices in bciVec (equal
                    // to nprobe)

            auto ii = bciVec[j]; // our current centroid index (used to index
                                 // into ivf)
            auto &curr_list = this->inv_lists[ii];
            auto num_vectors_in_list =
                curr_list.size() / this->d; // find number of vectors in list
            auto curr_list_data = curr_list.data();

            if constexpr (ParallelType == ParallelType::CANDIDATE_PARALLEL ||
                          ParallelType ==
                              ParallelType::QUERYCANDIDATE_PARALLEL) {
                static_assert(DistanceKernel != DistanceKernel::CACHEV2 &&
                                  DistanceKernel != DistanceKernel::CACHESIMDV2,
                              "CacheV2 can't be used in candidate-parallel or "
                              "query-candidate-parallel");
                std::vector<float> distances(num_vectors_in_list);
#pragma omp parallel for if (num_threads > 1)
                for (size_t vi = 0; vi < num_vectors_in_list; vi++) {
                    const float *vec =
                        curr_list_data +
                        vi * this->d; // our current vector within curr_list
                    distances[vi] = distance<DistanceKernel>(q, vec, this->d) *
                                    -1.0; // get distance
                }

                for (size_t vi = 0; vi < num_vectors_in_list; vi++) {
                    auto pq_distance = distances[vi];
                    auto label = this->labels[ii][vi];
                    auto pair = std::make_pair(pq_distance, label);
                    pq.push(pair);
                }
            } else {
                if constexpr (DistanceKernel == DistanceKernel::CACHE ||
                              DistanceKernel == DistanceKernel::CACHESIMD ||
                              DistanceKernel == DistanceKernel::OMPSIMD) {
                    float *distances = distance<DistanceKernel>(
                        q, curr_list_data, this->d, num_vectors_in_list);
                    for (size_t vi = 0; vi < num_vectors_in_list; vi++) {
                        auto pq_distance = -distances[vi];
                        auto label = this->labels[ii][vi];
                        auto pair = std::make_pair(pq_distance, label);
                        pq.push(pair);
                    }

                    delete[] distances;

                }

                else {
                    if constexpr (DistanceKernel == DistanceKernel::CACHEV2 ||
                                  DistanceKernel ==
                                      DistanceKernel::CACHESIMDV2) {
                        std::vector<float> distances(num_vectors_in_list, 0.0f);
                        size_t num_chunks =
                            (this->d + CHUNK_SIZE - 1) / CHUNK_SIZE;
                        const float *q_chunk = q;
                        const float *curr_list_data_chunk = curr_list_data;

                        // calculate distances
                        for (size_t c = 0; c < num_chunks; c++) {
                            size_t real_chunk_size = std::min<size_t>(
                                CHUNK_SIZE, this->d - c * CHUNK_SIZE);
                            for (size_t vi = 0; vi < num_vectors_in_list;
                                 vi++) {
                                distances[vi] += distance<DistanceKernel>(
                                    q_chunk, curr_list_data_chunk,
                                    real_chunk_size);
                                curr_list_data_chunk += real_chunk_size;
                            }
                            q_chunk += real_chunk_size;
                        }

                        // add to result
                        for (size_t vi = 0; vi < num_vectors_in_list; vi++) {
                            auto pq_distance = -distances[vi];
                            auto label = this->labels[ii][vi];
                            auto pair = std::make_pair(pq_distance, label);
                            pq.push(pair);
                        }
                    } else {
                        for (size_t vi = 0; vi < num_vectors_in_list; vi++) {
                            const float *vec =
                                curr_list_data +
                                vi * this->d; // our current vector within
                                              // curr_list
                            auto pq_distance =
                                distance<DistanceKernel>(q, vec, this->d) *
                                -1.0; // get distance
                            auto label =
                                this->labels[ii][vi]; // get label - find list
                                                      // with ii, find label
                                                      // w/in list with k
                            auto pair = std::make_pair(pq_distance, label);
                            pq.push(pair);
                        }
                    }
                }
            }
        }
        size_t num_to_add = std::min(k, (size_t)pq.size());
        for (size_t j = 0; j < num_to_add;
             j++) { // take the k closest vectors and put them on the right
                    // index in ret_vector
            auto [_, index] = pq.top();
            ret_labels[i].push_back(index);
            pq.pop();
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        total_time += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    }

    std::cout << "Average time per query: " << total_time / n_queries << "us" << std::endl;

    return ret_labels;
}

template <DistanceKernel DistanceKernel, ParallelType ParallelType>
std::vector<size_t>
IVF<DistanceKernel, ParallelType>::_top_n_centroids(const float *vector,
                                                    size_t n) const {

    if (n > this->nlist) {
        n = this->nlist;
    }

    std::vector<size_t> ret_vector;
    std::priority_queue<std::pair<float, size_t>> pq;
    const float *cent_data = this->centroids.data();

    if constexpr (DistanceKernel == DistanceKernel::CACHE ||
                  DistanceKernel == DistanceKernel::CACHESIMD ||
                  DistanceKernel == DistanceKernel::OMPSIMD) {
        float *distances =
            distance<DistanceKernel>(vector, cent_data, this->d, this->nlist);
        for (size_t c = 0; c < this->nlist; c++) {
            auto pq_distance = -distances[c];

            auto pair = std::make_pair(pq_distance, c);
            pq.push(pair);
        }

    } else {
        for (size_t c = 0; c < this->nlist; c++) {
            const float *cent = cent_data + c * this->d;

            auto pq_distance =
                distance<DistanceKernel>(vector, cent, this->d) * -1.0;
            auto pair = std::make_pair(pq_distance, c);
            pq.push(pair);
        }
    }

    for (size_t i = 0; i < n; i++) {
        auto [_, index] = pq.top();
        ret_vector.push_back(index);
        pq.pop();
    }
    return ret_vector;
}

// Explicit template instantiations
template class IVF<DistanceKernel::SCALAR, ParallelType::SERIAL>;
template class IVF<DistanceKernel::SIMD, ParallelType::SERIAL>;
template class IVF<DistanceKernel::CACHE, ParallelType::SERIAL>;
template class IVF<DistanceKernel::CACHESIMD, ParallelType::SERIAL>;
template class IVF<DistanceKernel::OMPSIMD, ParallelType::SERIAL>;
template class IVF<DistanceKernel::CACHEV2, ParallelType::SERIAL>;
template class IVF<DistanceKernel::CACHESIMDV2, ParallelType::SERIAL>;

template class IVF<DistanceKernel::SCALAR, ParallelType::QUERY_PARALLEL>;
template class IVF<DistanceKernel::SIMD, ParallelType::QUERY_PARALLEL>;
template class IVF<DistanceKernel::CACHE, ParallelType::QUERY_PARALLEL>;
template class IVF<DistanceKernel::CACHESIMD, ParallelType::QUERY_PARALLEL>;
template class IVF<DistanceKernel::CACHEV2, ParallelType::QUERY_PARALLEL>;
template class IVF<DistanceKernel::CACHESIMDV2, ParallelType::QUERY_PARALLEL>;
template class IVF<DistanceKernel::OMPSIMD, ParallelType::QUERY_PARALLEL>;

template class IVF<DistanceKernel::SCALAR, ParallelType::CANDIDATE_PARALLEL>;
template class IVF<DistanceKernel::SIMD, ParallelType::CANDIDATE_PARALLEL>;
template class IVF<DistanceKernel::CACHE, ParallelType::CANDIDATE_PARALLEL>;
template class IVF<DistanceKernel::CACHESIMD, ParallelType::CANDIDATE_PARALLEL>;
template class IVF<DistanceKernel::OMPSIMD, ParallelType::CANDIDATE_PARALLEL>;

template class IVF<DistanceKernel::SCALAR,
                   ParallelType::QUERYCANDIDATE_PARALLEL>;
template class IVF<DistanceKernel::SIMD, ParallelType::QUERYCANDIDATE_PARALLEL>;
template class IVF<DistanceKernel::CACHE,
                   ParallelType::QUERYCANDIDATE_PARALLEL>;
template class IVF<DistanceKernel::CACHESIMD,
                   ParallelType::QUERYCANDIDATE_PARALLEL>;
template class IVF<DistanceKernel::OMPSIMD,
                   ParallelType::QUERYCANDIDATE_PARALLEL>;
