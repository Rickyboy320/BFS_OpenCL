#include "kernels.cuh"

__global__ void BFS_1(const Node* g_nodes,
                    const int* g_edges,
                    char* g_mask, 
                    char* g_new_mask, 
                    char* g_visited, 
                    int* g_cost, 
                    char* done,
                    const int no_of_nodes){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < no_of_nodes && g_mask[tid]) 
    {
        g_mask[tid]=false;
        for(int i = g_nodes[tid].starting; i < g_nodes[tid].starting + g_nodes[tid].no_of_edges; i++) 
        {
            int id = g_edges[i];
            if(!g_visited[id])
            {
                g_cost[id] = g_cost[tid] + 1;
                g_new_mask[id] = true;
                g_visited[id] = true;
                *done = false;
            }
        }
    }	
}

void run_Bfs1(dim3 grid, dim3 threads, 
    const Node* g_nodes,
    const int* g_edges,
    char* g_mask, 
    char* g_new_mask, 
    char* g_visited, 
    int* g_cost, 
    char* done,
    const int no_of_nodes) {
        BFS_1<<<grid, threads>>>(g_nodes, g_edges, g_mask, g_new_mask, g_visited, g_cost, done, no_of_nodes);
    }
