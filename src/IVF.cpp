/**
 * @file IVF.cpp
 * @brief Serial implementation of IVF
 * @date 11-18-2025
 */

#include "IVF.h"

#include "distances.h"
#include <limits>
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

    for (size_t i = 0; i < n_train; i++) {
        const float *x = train_data + i * this->d;

        auto bciVec = this->_top_n_centroids(x, 1);
        auto bci = bciVec[0];
        auto &list = this->inv_lists[bci];
        list.insert(list.end(), x, x + this->d);

        this->labels[bci].emplace_back(i);
    }
    this->maxlabel = n_train - 1;
}

template <DistanceKernel DistanceKernel, ParallelType ParallelType>
void IVF<DistanceKernel, ParallelType>::add(const size_t n_add,
                                            const float *add_data) {
    if (this->centroids.empty() ||
        this->inv_lists.empty()) { // if we have not trained or not built,
                                   // nothing should happen
        return;
    }

    for (size_t i = 0; i < n_add; i++) {
        const float *x = add_data + i * this->d;

        auto bciVec = this->_top_n_centroids(x, 1);
        auto bci = bciVec[0];
        auto &list = this->inv_lists[bci];
        list.insert(list.end(), x, x + this->d);

        this->labels[bci].emplace_back(this->maxlabel + 1);
        this->maxlabel++;
    }
}

template <DistanceKernel DistanceKernel, ParallelType ParallelType>
std::vector<std::vector<size_t>>
IVF<DistanceKernel, ParallelType>::search(const size_t n_queries,
                                          const float *queries, const size_t k,
                                          const size_t n_probe) const {
    std::vector<std::vector<size_t>> ret_labels;
    ret_labels.resize(n_queries);

#pragma omp parallel for if (ParallelType == ParallelType::QUERY_PARALLEL)
    for (size_t i = 0; i < n_queries; i++) {

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

            if constexpr (ParallelType == ParallelType::CANDIDATE_PARALLEL) {
                std::vector<float> distances(num_vectors_in_list);
#pragma omp parallel for
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
                              DistanceKernel == DistanceKernel::CACHESIMD) {
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
                    for (size_t vi = 0; vi < num_vectors_in_list; vi++) {
                        const float *vec =
                            curr_list_data +
                            vi * this->d; // our current vector within curr_list
                        auto pq_distance =
                            distance<DistanceKernel>(q, vec, this->d) *
                            -1.0; // get distance
                        auto label =
                            this->labels[ii]
                                        [vi]; // get label - find list with ii,
                                              // find label w/in list with k
                        auto pair = std::make_pair(pq_distance, label);
                        pq.push(pair);
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
    }

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
                  DistanceKernel == DistanceKernel::CACHESIMD) {
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
template class IVF<DistanceKernel::SCALAR, ParallelType::QUERY_PARALLEL>;
template class IVF<DistanceKernel::SIMD, ParallelType::QUERY_PARALLEL>;
template class IVF<DistanceKernel::CACHE, ParallelType::QUERY_PARALLEL>;
template class IVF<DistanceKernel::CACHESIMD, ParallelType::QUERY_PARALLEL>;
template class IVF<DistanceKernel::SCALAR, ParallelType::CANDIDATE_PARALLEL>;
template class IVF<DistanceKernel::SIMD, ParallelType::CANDIDATE_PARALLEL>;
template class IVF<DistanceKernel::CACHE, ParallelType::CANDIDATE_PARALLEL>;
template class IVF<DistanceKernel::CACHESIMD, ParallelType::CANDIDATE_PARALLEL>;
