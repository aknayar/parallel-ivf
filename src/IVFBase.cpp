/**
 * @file IVFBase.cpp
 * @brief Serial implementation of IVF
 * @date 11-18-2025
 */

#include "IVFBase.h"
#include <limits>

void IVFBase::train(const size_t n_train, const float *train_data) {
    centroids.resize(nlist * d);
    kmeans.train(n_train, train_data, centroids.data(), nlist);
}

void IVFBase::build(const size_t n_train, const float *train_data){
    //should include some sort of error handling if centroids not builts
    if (centroids.empty()){
        return;
    }

    //now build empty vector list
    //for each point in data find closest centroid
    //add it to list
    //add label to lable list
    inv_lists.resize(nlist);
    labels.resize(nlist);
    const float* cent_data = centroids.data();

    for(size_t i = 0; i < n_train; i++){

        const float* x = train_data + i * d;
        // std::vector<float>curr_vector;
        // for (size_t j = 0; j < d; j++){
        //     curr_vector.emplace_back(train_data[i*d+j]);
        // }
        // auto curr_vector_data = curr_vector.data();
        float min_distance = std::numeric_limits<float>::max();
        size_t bci = 0;
        
        for(size_t c=0; c < nlist; c++){
            const float* cent = cent_data + c*d;

            auto curr_distance = distance_scalar(x, cent, d);
            if (curr_distance < min_distance){
                min_distance = curr_distance;
                bci = c;
            }
            
        }

        auto &list = inv_lists[bci];
        list.insert(list.end(), x, x + d);

        
        labels[bci].emplace_back(i);
         
    }
    maxlabel = n_train-1;


}



void IVFBase::add(const size_t n_add, const float *add_data) {

    // TODO: Implement
}

void IVFBase::search(const size_t n_queries, const float *queries,
                     const size_t k, const size_t nprobe) const {
    // TODO: Implement
}


void IVFBase::_top_n_centroids(const float *vector){
    
}
