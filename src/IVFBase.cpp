/**
 * @file IVFBase.cpp
 * @brief Serial implementation of IVF
 * @date 11-18-2025
 */

#include "IVFBase.h"
#include <limits>
#include <queue>
#include <utility>


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
    if (centroids.empty() || inv_lists.empty()){ //if we have not trained or not built, nothing should happen
        return;
    }


    for(size_t i = 0; i < n_add; i++){
        const float* x = add_data + i * d;

        auto bciVec = _top_n_centroids(x, 1);
        auto bci = bciVec[0];
        auto &list = inv_lists[bci];
        list.insert(list.end(), x, x + d);

        labels[bci].emplace_back(maxlabel+1);
        maxlabel++;
    }

}

std::vector<std::vector<size_t>> IVFBase::search(const size_t n_queries, const float *queries,
                     const size_t k, const size_t n_probe) const {
    // TODO: Implement
    

    // if (n_probe > nlist){

    //     n_probe = nlist;
    // }

    std::vector<std::vector<size_t>> ret_labels;
    ret_labels.resize(n_queries);

    //for each query we want to 
    // find the top nprobe centroid indices
    // scan every vector in the corresponding ivf for similarity and rank
    // reutnr the k closest indices
    for(size_t i = 0; i < n_queries; i++){
        const float* q = queries+ i * d;
        auto bciVec = _top_n_centroids(q, n_probe); //get indices of nprobe closest centroids
        size_t n_probe_clamped = bciVec.size();
        std::priority_queue<std::pair<float,size_t>> pq;

        for (size_t j =0; j < n_probe_clamped; j++){ //do the below for all centroids indices in bciVec (equal to nprobe)

            auto ii = bciVec[j]; //our current centroid index (used to index into ivf)
            auto &curr_list = inv_lists[ii]; 
            auto num_vectors_in_list = curr_list.size() / d; //find number of vectors in list
            auto curr_list_data = curr_list.data(); 
            for (size_t vi = 0; vi < num_vectors_in_list; vi++){ //
                const float* vec = curr_list_data+vi*d; //our current vector within curr_list
                auto pq_distance = distance_scalar(q, vec, d) * -1.0; //get distance
                auto label = labels[ii][vi]; //get label - find list with ii, find label w/in list with k
                auto pair = std::make_pair(pq_distance,label); 
                pq.push(pair);
            }
        }
        size_t num_to_add = std::min(k, (size_t)pq.size());
        for(size_t j = 0; j < num_to_add; j++){ //take the k closest vectors and put them on the right index in ret_vector
            auto [_, index] = pq.top(); 
            ret_labels[i].push_back(index);
            pq.pop();
        }

    }


    return ret_labels;


}


std::vector<size_t> IVFBase::_top_n_centroids(const float *vector, size_t n) const{

    if (n > nlist){
        n = nlist;
    }

    std::vector<size_t> ret_vector;
    

    std::priority_queue<std::pair<float,size_t>> pq;

    const float* cent_data = centroids.data();

    for(size_t c=0; c < nlist; c++){
        const float* cent = cent_data + c*d;

        auto pq_distance = distance_scalar(vector, cent, d) * -1.0;
        auto pair = std::make_pair(pq_distance,c);
        pq.push(pair);
        
    }


    for(size_t i = 0; i < n; i++){
        auto [_, index] = pq.top();
        ret_vector.push_back(index);
        pq.pop();
    }
    return ret_vector;


}
