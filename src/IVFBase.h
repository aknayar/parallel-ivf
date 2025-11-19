/**
 * @file IVFBase.h
 * @brief Base implementation of IVF
 * @date 11-18-2025
 */

#ifndef IVF_BASE_H
#define IVF_BASE_H

#include "IVF.h"
#include "distances.h"
#include "kmeans.h"

struct IVFBase : IVF {
    public:
        KMeans<DistanceKernel::SCALAR> kmeans;

        IVFBase(size_t d, size_t nlist) : IVF(d, nlist), kmeans(d, nlist) {}

        void train(const size_t n_train, const float *train_data) override;
        void build(const size_t n_train, const float *train_data) override;
        void add(const size_t n_add, const float *add_data) override;
        void search(const size_t n_queries, const float *queries, const size_t k,
                    const size_t nprobe) const override;
    private:
        std::vector<size_t> _top_n_centroids(const float *vector, size_t n);
};

#endif // IVF_BASE_H
