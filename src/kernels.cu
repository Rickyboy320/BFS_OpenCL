#include "kernels.cuh"

__global__ void BFS_1(const Node* g_graph_nodes,
                    const int* g_graph_edges,
                    char* g_graph_mask, 
                    char* g_updating_graph_mask, 
                    char* g_graph_visited, 
                    int* g_cost, 
                    const int no_of_nodes){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < no_of_nodes && g_graph_mask[tid]) 
    {
        g_graph_mask[tid]=false;
        for(int i = g_graph_nodes[tid].starting; i < g_graph_nodes[tid].starting + g_graph_nodes[tid].no_of_edges; i++) 
        {
            int id = g_graph_edges[i];
            if(!g_graph_visited[id])
            {
                g_cost[id] = g_cost[tid] + 1;
                g_updating_graph_mask[id] = true;
            }
        }
    }	
}

__global__ void BFS_2(char* g_graph_mask, 
                    char* g_updating_graph_mask, 
                    char* g_graph_visited, 
                    char* g_over,
                    const int no_of_nodes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < no_of_nodes && g_updating_graph_mask[tid])
    {
        g_graph_mask[tid]=true;
        g_graph_visited[tid]=true;
        *g_over=true;
        g_updating_graph_mask[tid]=false;
    }
}

void run_Bfs1(dim3 grid, dim3 threads, 
    const Node* g_graph_nodes,
    const int* g_graph_edges,
    char* g_graph_mask, 
    char* g_updating_graph_mask, 
    char* g_graph_visited, 
    int* g_cost, 
    const int no_of_nodes) {
        BFS_1<<<grid, threads>>>(g_graph_nodes, g_graph_edges, g_graph_mask, g_updating_graph_mask, g_graph_visited, g_cost, no_of_nodes);
    }

void run_Bfs2(dim3 grid, dim3 threads,
    char* g_graph_mask, 
    char* g_updating_graph_mask, 
    char* g_graph_visited, 
    char* g_over,
    const int no_of_nodes) {
        BFS_2<<<grid, threads>>>(g_graph_mask, g_updating_graph_mask, g_graph_visited, g_over, no_of_nodes);
    }


