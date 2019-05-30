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
	// 	if(feature_end==INFTY)
	//		return feature_src+1;
	//	else return feature_end;
	return vert_status_prev[src];
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
	return true;
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
	//index_t degree=beg_pos[active_edge_src+1]-beg_pos[active_edge_src];
	return vert_status_prev[src];
		//return (feature_src==level ? feature_src+1:feature_end);
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
	return true;
}



__device__ cb_reducer vert_selector_push_d = vertex_selector_push;
__device__ cb_reducer vert_selector_pull_d = vertex_selector_pull;
__device__ cb_mapper vert_behave_push_d = user_mapper_push;
__device__ cb_mapper vert_behave_pull_d = user_mapper_pull;

__global__ void 
init(meta_data mdata, gpu_graph ggraph)
{
	index_t tid = threadIdx.x+blockIdx.x*blockDim.x;
	float init_val = 1.0/ggraph.vert_count;

	while(tid < ggraph.vert_count)
	{
		mdata.vert_status[tid] = 0;
		index_t degree = ggraph.beg_pos[tid + 1] - ggraph.beg_pos[tid];
		if(degree != 0)
			mdata.vert_status_prev[tid] = init_val/degree;
		else 
			mdata.vert_status_prev[tid] = 0;
		
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
	tm_red=wtime();
	reducer_pull(0, ggraph, mdata, worklist_gather);
	tm_red=wtime()-tm_red;
	
	feature_t *level, *level_h;
	cudaMalloc((void **)&level, sizeof(feature_t));
	cudaMallocHost((void **)&level_h, sizeof(feature_t));
	H_ERR(cudaDeviceSynchronize());
	double time=wtime();
	push_pull_opt
		(level, ggraph, mdata, compute_mapper, worklist_gather, global_barrier);
	
	//for(int levels=0;;levels++)
	//{
	//	//H_ERR(cudaMemcpy(mdata.sa_chk, mdata.vert_status_prev, 
	//	//			sizeof(feature_t)*ggraph.vert_count, cudaMemcpyDeviceToHost));
	//	//for(int i = 0; i < 10; i ++)
	//	//	std::cout<<mdata.sa_chk[i] * (ginst->beg_pos[i+1] - ginst->beg_pos[i])<<" ";
	//	//std::cout<<"\n";
	//	
	//	/* mapper */
	//	tm_map=wtime();
	//	mapper_pull(level, ggraph, mdata, compute_mapper);
	//	tm_map=wtime()-tm_map;
	//	
	//	feature_t *tmp = compute_mapper.vert_status;
	//	compute_mapper.vert_status = compute_mapper.vert_status_prev;
	//	compute_mapper.vert_status_prev = tmp;
	//	
	//	//H_ERR(cudaMemcpy(sml, mdata.worklist_sz_sml, sizeof(vertex_t), cudaMemcpyDeviceToHost));
	//	//H_ERR(cudaMemcpy(mid, mdata.worklist_sz_mid, sizeof(vertex_t), cudaMemcpyDeviceToHost));
	//	//H_ERR(cudaMemcpy(lrg, mdata.worklist_sz_lrg, sizeof(vertex_t), cudaMemcpyDeviceToHost));
	//	//
	//	//printf("level-%d: %d\n", levels, sml[0]+mid[0]+lrg[0]);
	//		
	//		
	//	/*monitoring*/
	//	std::cout<<"Level: "<<(int)levels<<" " 
	//		<<"Time (map, reduce): "<<tm_map<<" "<<tm_red<<"\n";

	//	if(levels == 10)break;
	//}
	std::cout<<"Total time: "<<wtime()-time<<" second(s).\n";
	//dumper(ggraph,mdata);
}
