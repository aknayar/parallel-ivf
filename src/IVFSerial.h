/**
 * @file IVFSerial.h
 * @brief Serial implementation of IVF
 * @date 11-18-2025
 */

#ifndef IVF_SERIAL_H
#define IVF_SERIAL_H

#include "IVF.h"

struct IVFSerial : public IVF {
    IVFSerial(size_t d, size_t nlist) : IVF(d, nlist) {}

    void train(const size_t n_train, const float *train_data) override;
    void add(const size_t n_add, const float *add_data) override;
    void search(const size_t n_queries, const float *queries, const size_t k,
                const size_t nprobe) const override;
};

#endif // IVF_SERIAL_H
