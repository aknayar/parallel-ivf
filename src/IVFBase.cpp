/**
 * @file IVFBase.cpp
 * @brief Serial implementation of IVF
 * @date 11-18-2025
 */

#include "IVFBase.h"

void IVFBase::train(const size_t n_train, const float *train_data) {
    centroids.resize(nlist * d);
    kmeans.train(n_train, train_data, centroids.data(), nlist);
}

void IVFBase::build(const size_t n_train, const float *train_data){
    //should include some sort of error handling if centroids not buitl

}

void IVFBase::add(const size_t n_add, const float *add_data) {
    // TODO: Implement
}

void IVFBase::search(const size_t n_queries, const float *queries,
                     const size_t k, const size_t nprobe) const {
    // TODO: Implement
}
