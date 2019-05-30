#include "header.h"
#include "util.h"
#include "mapper.cuh"
#include "reducer.cuh"
#include "wtime.h"
#include "barrier.cuh"
#include "gpu_graph.cuh"
#include "meta_data.cuh"
#include "mapper_enactor.cuh"
#include "cpu_bfs.hpp"

/*user defined vertex behavior function*/
__inline__ __host__ __device__ feature_t user_mapper_push
(	vertex_t 	src,
	vertex_t	dest,
	feature_t	level,
	index_t*	beg_pos,
	weight_t	edge_weight,
	feature_t* 	vert_status,
	feature_t* 	vert_status_prev)
{
	feature_t feature_end = vert_status[dest];
	return (feature_end == INFTY ? level+1 : feature_end);
}

/*user defined vertex behavior function*/
__inline__ __host__ __device__ bool vertex_selector_push
(
  vertex_t vert_id, 
  feature_t level,
  vertex_t *adj_list, 
  index_t *beg_pos, 
  feature_t *vert_status,
  feature_t *vert_status_prev)
{
  //if(vert_status[vert_id]==level)	return true;
	//else return false;
	return (vert_status[vert_id]==level);
}

/*user defined vertex behavior function*/
__inline__ __host__ __device__ feature_t user_mapper_pull
(	vertex_t 		src,
	vertex_t		dest,
	feature_t		level,
	index_t*		beg_pos,
	weight_t		edge_weight,
	feature_t* 		vert_status,
	feature_t* 		vert_status_prev)
{
	return vert_status[src];
}

/*user defined vertex behavior function*/
__inline__ __host__ __device__ bool vertex_selector_pull
(
  vertex_t vert_id, 
  feature_t level,
  vertex_t *adj_list, 
  index_t *beg_pos, 
  feature_t *vert_status,
  feature_t *vert_status_prev)
{
  //if(vert_status[vert_id]==INFTY)	return true;
	//else return false;
	return (vert_status[vert_id]==INFTY);
}

__device__ cb_reducer vert_selector_push_d = vertex_selector_push;
__device__ cb_reducer vert_selector_pull_d = vertex_selector_pull;
__device__ cb_mapper vert_behave_push_d = user_mapper_push;
__device__ cb_mapper vert_behave_pull_d = user_mapper_pull;


/*init traversal*/
/*init traversal*/
__global__ void
init(vertex_t src_v, vertex_t vert_count, meta_data mdata)
{
	//
	////status
	//mdata.vert_status[src_v] = 0;

	index_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	while(tid < vert_count)
	{
		if(tid != src_v)
			mdata.vert_status[tid] = INFTY;
		else
		{
			mdata.vert_status[tid] = 0;
			
			mdata.worklist_mid[0] = src_v;
			mdata.worklist_sz_sml[0] = 0;
			mdata.worklist_sz_mid[0] = 1;
			mdata.worklist_sz_lrg[0] = 0;
			mdata.bitmap[src_v>>3] |= (1<<(src_v & 7));
		}
		tid += blockDim.x * gridDim.x;
	}
}

	int 
main(int args, char **argv)
{
	std::cout<<"Input: /path/to/exe /path/to/beg_pos /path/to/adj_list /path/weight_list src\n";
	if(args<5){std::cout<<"Wrong input\n";exit(-1);}
		
	double tm_map,tm_red,tm_scan;
	char *file_beg_pos = argv[1];
	char *file_adj_list = argv[2];
	char *file_weight_list = argv[3];
	vertex_t src_v = (vertex_t)atol(argv[4]);
	H_ERR(cudaSetDevice(0));	
	
	//Read graph to CPU
	graph<long, long, long,vertex_t, index_t, weight_t>
	*ginst=new graph<long, long, long,vertex_t, index_t, weight_t>
	(file_beg_pos, file_adj_list, file_weight_list);
	
	feature_t *level, *level_h;
	cudaMalloc((void **)&level, sizeof(feature_t));
	cudaMallocHost((void **)&level_h, sizeof(feature_t));
	
	cb_reducer vert_selector_push_h;
	cb_reducer vert_selector_pull_h;
	cudaMemcpyFromSymbol(&vert_selector_push_h,vert_selector_push_d,sizeof(cb_reducer));
	cudaMemcpyFromSymbol(&vert_selector_pull_h,vert_selector_pull_d,sizeof(cb_reducer));
	
	cb_mapper vert_behave_push_h;
	cb_mapper vert_behave_pull_h;
	cudaMemcpyFromSymbol(&vert_behave_push_h,vert_behave_push_d,sizeof(cb_reducer));
	cudaMemcpyFromSymbol(&vert_behave_pull_h,vert_behave_pull_d,sizeof(cb_reducer));
	
	//Init three data structures
	gpu_graph ggraph(ginst);
	meta_data mdata(ginst->vert_count, ginst->edge_count);
	Barrier global_barrier(BLKS_NUM);
	init<<<256,256>>>(src_v, ginst->vert_count, mdata);
	mapper compute_mapper(ggraph, mdata, vert_behave_push_h, vert_behave_pull_h);
	reducer worklist_gather(ggraph, mdata, vert_selector_push_h, vert_selector_pull_h);
	H_ERR(cudaThreadSynchronize());
	
	double time = wtime();
	mapper_merge_push
		(level, ggraph, mdata, compute_mapper, worklist_gather, global_barrier);
	//push_pull_opt(level, ggraph, mdata, compute_mapper, worklist_gather, global_barrier);
	time = wtime() - time;
	
	cudaMemcpy(level_h, level, sizeof(feature_t), cudaMemcpyDeviceToHost);	
	std::cout<<(int)level_h[0]<<" levels "<<time<<" second(s).\n";
    
    feature_t *gpu_dist = new feature_t[ginst->vert_count];
    cudaMemcpy(gpu_dist, mdata.vert_status, 
            sizeof(feature_t) * ginst->vert_count, cudaMemcpyDeviceToHost);

    feature_t *cpu_dist;
    cpu_bfs<index_t, vertex_t, feature_t>
        (cpu_dist, src_v, ginst->vert_count, ginst->edge_count, ginst->beg_pos,
         ginst->adj_list);
    if (memcmp(cpu_dist, gpu_dist, sizeof(feature_t) * ginst->vert_count) == 0)
        printf("Result correct\n");
    else printf("Result wrong!\n");
}
