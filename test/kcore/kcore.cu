#include "header.h"
#include "util.h"
#include "mapper.cuh"
#include "reducer.cuh"
#include "wtime.h"
#include "barrier.cuh"
#include "gpu_graph.cuh"
#include "meta_data.cuh"
#include "mapper_enactor.cuh"
#include "reducer_enactor.cuh"

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
	return 1;
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
	if(vert_status_prev[vert_id] == vert_status[vert_id])
	{
		if(vert_status[vert_id] <= K)
		{
			vert_status_prev[vert_id] = 0;
			vert_status[vert_id] = 0;
		}
		return false;
	}
	else
	{
		if(vert_status_prev[vert_id]>K &&vert_status[vert_id]<=K)
			return true;
		else
			return false;
	}

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
	return ((vert_status_prev[src] <= K)
			&& (vert_status_prev[src]>0)); 
}

/*user defined vertex behavior function*/
//Attention, we only use pull once
// - at the beginning of kcore
__inline__ __host__ __device__ bool vertex_selector_pull
(
  vertex_t vert_id, 
  feature_t level,
  vertex_t *adj_list, 
  index_t *beg_pos, 
  feature_t *vert_status,
  feature_t *vert_status_prev)
{
	return (vert_status[vert_id] > K);
}



__device__ cb_reducer vert_selector_push_d = vertex_selector_push;
__device__ cb_reducer vert_selector_pull_d = vertex_selector_pull;
__device__ cb_mapper vert_behave_push_d = user_mapper_push;
__device__ cb_mapper vert_behave_pull_d = user_mapper_pull;

__global__ void 
init(meta_data mdata, gpu_graph ggraph)
{
	index_t tid = threadIdx.x+blockIdx.x*blockDim.x;
	
	while(tid < ggraph.vert_count)
	{
		feature_t degree = ggraph.beg_pos[tid + 1] 
			- ggraph.beg_pos[tid];
		mdata.vert_status[tid] = degree;
		mdata.vert_status_prev[tid] = degree;
	
		tid += blockDim.x*gridDim.x;
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
	
	cb_reducer vert_selector_push_h;
	cb_reducer vert_selector_pull_h;
	H_ERR(cudaMemcpyFromSymbol(&vert_selector_push_h,vert_selector_push_d,sizeof(cb_reducer)));
	H_ERR(cudaMemcpyFromSymbol(&vert_selector_pull_h,vert_selector_pull_d,sizeof(cb_reducer)));
	
	cb_mapper vert_behave_push_h;
	cb_mapper vert_behave_pull_h;
	H_ERR(cudaMemcpyFromSymbol(&vert_behave_push_h,vert_behave_push_d,sizeof(cb_reducer)));
	H_ERR(cudaMemcpyFromSymbol(&vert_behave_pull_h,vert_behave_pull_d,sizeof(cb_reducer)));
	
	gpu_graph ggraph(ginst);
	meta_data mdata(ginst->vert_count, ginst->edge_count);
	Barrier global_barrier(BLKS_NUM);
	mapper compute_mapper(ggraph, mdata, vert_behave_push_h, vert_behave_pull_h);
	reducer worklist_gather(ggraph, mdata, vert_selector_push_h, vert_selector_pull_h);
	H_ERR(cudaThreadSynchronize());

	init<<<256, 256>>>(mdata, ggraph);
		
	H_ERR(cudaMemset(mdata.worklist_sz_sml, 0, sizeof(vertex_t)));
	H_ERR(cudaMemset(mdata.worklist_sz_mid, 0, sizeof(vertex_t)));
	H_ERR(cudaMemset(mdata.worklist_sz_lrg, 0, sizeof(vertex_t)));
	H_ERR(cudaThreadSynchronize());
	
	vertex_t *sml, *mid, *lrg;
	cudaMallocHost((void **)&sml, sizeof(vertex_t));
	cudaMallocHost((void **)&mid, sizeof(vertex_t));
	cudaMallocHost((void **)&lrg, sizeof(vertex_t));
	
	/*reducer*/
	double ltime = wtime();
	tm_red=wtime();
	//reducer_pull(0, ggraph, mdata, worklist_gather);
	reducer_strided_pull(0, ggraph, mdata, worklist_gather);
	tm_red=wtime()-tm_red;
	
	tm_map=wtime();
	mapper_pull(0, ggraph, mdata, compute_mapper);
	tm_map=wtime()-tm_map;
		
	int levels=0;
	
	H_ERR(cudaMemcpy(sml, mdata.worklist_sz_sml, 
				sizeof(vertex_t), cudaMemcpyDeviceToHost));
	H_ERR(cudaMemcpy(mid, mdata.worklist_sz_mid, 
				sizeof(vertex_t), cudaMemcpyDeviceToHost));
	H_ERR(cudaMemcpy(lrg, mdata.worklist_sz_lrg, 
				sizeof(vertex_t), cudaMemcpyDeviceToHost));
	printf("level-%d: %d (%lf, %lf)(map, reduce) seconds\n", levels, 
				sml[0]+mid[0]+lrg[0], tm_map, tm_red);
			
	
	H_ERR(cudaThreadSynchronize());
	H_ERR(cudaMemset(mdata.worklist_sz_sml, 0, sizeof(vertex_t)));
	H_ERR(cudaMemset(mdata.worklist_sz_mid, 0, sizeof(vertex_t)));
	H_ERR(cudaMemset(mdata.worklist_sz_lrg, 0, sizeof(vertex_t)));
	H_ERR(cudaThreadSynchronize());

	levels++;
	
	for(;;levels++)
	{
		tm_red=wtime();
		reducer_push(levels, ggraph, mdata, worklist_gather);
		tm_red=wtime()-tm_red;
		
		H_ERR(cudaMemcpy(sml, mdata.worklist_sz_sml, 
					sizeof(vertex_t), cudaMemcpyDeviceToHost));
		H_ERR(cudaMemcpy(mid, mdata.worklist_sz_mid, 
					sizeof(vertex_t), cudaMemcpyDeviceToHost));
		H_ERR(cudaMemcpy(lrg, mdata.worklist_sz_lrg, 
					sizeof(vertex_t), cudaMemcpyDeviceToHost));

		if(sml[0] + mid[0] + lrg[0] == 0) break;

		tm_map = wtime();
		mapper_push(levels, ggraph, mdata, compute_mapper);
		tm_map = wtime() - tm_map;

		printf("level-%d: %d (%lf, %lf)(map, reduce) seconds\n", levels, 
				sml[0]+mid[0]+lrg[0], tm_map, tm_red);
			
		H_ERR(cudaThreadSynchronize());
		H_ERR(cudaMemset(mdata.worklist_sz_sml, 0, sizeof(vertex_t)));
		H_ERR(cudaMemset(mdata.worklist_sz_mid, 0, sizeof(vertex_t)));
		H_ERR(cudaMemset(mdata.worklist_sz_lrg, 0, sizeof(vertex_t)));
		H_ERR(cudaThreadSynchronize());
	}

	ltime = wtime() - ltime;

	std::cout<<"Total time: "<<ltime<<" second(s).\n";
    
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
