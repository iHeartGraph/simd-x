#include "header.h"
#include <iostream>

template <typename index_t,
         typename vertex_t,
         typename feature_t>
void cpu_pagerank(
        feature_t* &dist,
        vertex_t vert_count,
        index_t edge_count,
        index_t *beg_pos,
        vertex_t *csr){
    
    feature_t *dist_prev = new feature_t[vert_count];
    dist = new feature_t[vert_count];
    feature_t *degree_reverse = new feature_t[vert_count];

    feature_t init_rank = 1.0/vert_count;

    std::cout<<init_rank<<"\n";
    for(vertex_t i = 0; i < vert_count; i ++)
    {
        dist_prev[i] = init_rank;
        
        if(beg_pos[i+1] - beg_pos[i] != 0)
            degree_reverse[i] = 1.0/(beg_pos[i+1] - beg_pos[i]);
        else degree_reverse[i] = 0;
    }


    feature_t level = 0;
    while(true)
    {
        for(vertex_t i = 0; i < vert_count; i ++)
        {
            vertex_t frontier = i;

            index_t my_beg = beg_pos[frontier];
            index_t my_end = beg_pos[frontier+1];
            
            feature_t new_rank = 0;
            for(;my_beg < my_end; my_beg ++)
            {
                vertex_t nebr = csr[my_beg];
                new_rank += dist_prev[nebr];
            }
            dist[frontier] = (0.15 + 0.85 * new_rank) * degree_reverse[frontier];
        }
        level ++;

        feature_t rank_sum = 0;
        for(vertex_t i = 0; i < vert_count; i++)
            rank_sum += dist[i] * (beg_pos[i+1] - beg_pos[i]);
            //std::cout<<dist[i] * (beg_pos[i+1] - beg_pos[i])<<"\n";
        
        std::cout<<"Iteration "<<level<<": "<<rank_sum<<"\n";
        
        feature_t *tmp = dist;
        dist = dist_prev;
        dist_prev = tmp;
    }
    
    delete[] dist_prev;
    delete[] degree_reverse;
    return;
}
