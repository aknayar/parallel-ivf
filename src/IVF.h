/**
 * @file IVF.h
 * @brief Inverted vector file base class
 * @author Akash Nayar <akashnay@andrew.cmu.edu>
 * @author Dhruva Byrapatna <dbyrapat@andrew.cmu.edu>
 * @date 11-18-2025
 */

#ifndef IVF_H
#define IVF_H

#include <cstddef>
#include <vector>

struct IVF {
    size_t d, nlist, nprobe = 0;
    std::vector<std::vector<float>> inv_lists; // Inverted lists
    std::vector<std::vector<size_t>> labels;   // Associated labels
    std::vector<float> centroids;              // Centroids

    IVF(size_t d, size_t nlist) : d(d), nlist(nlist) {}

    virtual void train(const size_t n_train, const float *train_data) = 0;

    virtual void add(const size_t n_add, const float *add_data) = 0;

    virtual void search(const size_t n_queries, const float *queries,
                        const size_t k, const size_t nprobe) const = 0;
};

#endif // IVF_H