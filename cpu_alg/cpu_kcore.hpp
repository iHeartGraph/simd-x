#include "header.h"

template <typename index_t,
         typename vertex_t,
         typename weight_t,
         typename feature_t>
void cpu_kcore(
        feature_t* &dist,
        feature_t k,
        vertex_t vert_count,
        index_t edge_count,
        index_t *beg_pos,
        vertex_t *csr){
    
    dist = new feature_t[vert_count];
    for(vertex_t i = 0; i < vert_count; i ++)
        dist[i] = beg_pos[i+1] - beg_pos[i];

    index_t update_count;
    index_t optimized_count;

    feature_t level = 0;
    while(true)
    {
        update_count = 0;
        optimized_count = 0;
        
        for(vertex_t i = 0; i < vert_count; i ++)
        {
            if(dist[i] > -1 * vert_count && dist[i] <= k)
            {
                vertex_t frontier = i;
                dist[frontier] = -1 * vert_count; //deactivate myself

                index_t my_beg = beg_pos[frontier];
                index_t my_end = beg_pos[frontier+1];
                
                for(;my_beg < my_end; my_beg ++)
                {
                    vertex_t nebr = csr[my_beg];
                    if(dist[nebr] > k)
                    {
                        --dist[nebr];
                        ++update_count;
                    }
                    
                    if((dist[nebr] <= k) && (dist[nebr] > -1 * vert_count))
                        ++optimized_count;
                }
            }
        }
        
        level ++;
        std::cout<<"Iteration "<<level<<", update ratio, optimized ratio "
            <<update_count*1.0/edge_count<<" "<<optimized_count*1.0/edge_count<<"\n";
        
        if (update_count == 0) break;
    }

    index_t remain_vert_count = 0;
    for(vertex_t i = 0; i < vert_count; i ++)
        if(dist[i] > k ) remain_vert_count ++;

    std::cout<<k<<"-core vertex count: "<<remain_vert_count<<"\n";
    return;
}
