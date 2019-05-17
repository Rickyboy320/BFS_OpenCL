#include "kernels.cuh"

__global__ void BFS_TD(const Node* g_graph_nodes,
                    int* g_graph_frontier,
                    const int* g_graph_frontier_size,
                    const int* g_graph_edges,
                    int* g_new_frontier,
                    int* g_new_frontier_size, 
                    int* g_graph_visited, 
                    int* g_amount_frontier_edges,
                    int* g_cost, 
                    const int no_of_nodes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < *g_graph_frontier_size) 
    {
        int nodeId = g_graph_frontier[tid];
        Node node = g_graph_nodes[nodeId];
        for(int i = node.starting; i < node.starting + node.no_of_edges; i++) 
        {
            int id = g_graph_edges[i];
            if(atomicExch(&g_graph_visited[id], 1) == 0)
            {    
                int old = atomicAdd(g_new_frontier_size, 1);
                g_cost[id] = g_cost[nodeId] + 1;

                g_new_frontier[old] = id;
                g_graph_visited[id] = true;

                Node target = g_graph_nodes[id];
                atomicAdd(g_amount_frontier_edges, target.no_of_edges); 
            }
        }
    }	
}

__global__ void BFS_BU(const Node* g_graph_nodes,
                    const char* g_graph_mask,
                    const int* g_graph_edges,
                    char* g_new_mask,
                    int* g_graph_visited, 
                    int* g_cost, 
                    int* g_new_frontier_size,
                    int* g_amount_frontier_edges,
                    const int no_of_nodes){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if this node has not been visited yet (untraversed)
    if(tid < no_of_nodes && !g_graph_visited[tid]) 
    {
        // Loop over its edges
        Node node = g_graph_nodes[tid];
        for(int i = node.starting; i < node.starting + node.no_of_edges; i++) 
        {
            // If a neighbour is part of the current frontier (mask)
            int id = g_graph_edges[i];
            if(g_graph_mask[id])
            {
                // Increment cost based on parent, and set this as to-be-updated.
                g_cost[tid] = g_cost[id] + 1;

                g_new_mask[tid] = true;
                g_graph_visited[tid] = true;
                atomicAdd(g_new_frontier_size, 1);
                atomicAdd(g_amount_frontier_edges, node.no_of_edges); 
                break;
            }
        }
    }	
}

__global__ void BFS_CONVERT_BU(char* g_graph_mask,
                            int* g_new_frontier,
                            int* g_new_frontier_size,
                            const int no_of_nodes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < no_of_nodes && g_graph_mask[tid]) {
        int old = atomicAdd(g_new_frontier_size, 1);
        g_new_frontier[old] = tid;
    }
}

__global__ void BFS_CONVERT_TD(int* g_frontier,
                            int* g_frontier_size,
                            char* g_new_graph_mask) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < *g_frontier_size) {
        g_new_graph_mask[g_frontier[tid]] = true;
    }
}

__global__ void ZERO(char* g_new_graph_mask,
                    const int no_of_nodes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < no_of_nodes) {
        g_new_graph_mask[tid] = false;
    }
}

void run_TD(dim3 grid, dim3 threads, 
                    const Node* g_graph_nodes,
                    int* g_graph_frontier,
                    const int* g_graph_frontier_size,
                    const int* g_graph_edges,
                    int* g_new_frontier,
                    int* g_new_frontier_size, 
                    int* g_graph_visited, 
                    int* g_amount_frontier_edges,
                    int* g_cost, 
                    const int no_of_nodes) {
    BFS_TD<<<grid, threads>>>(g_graph_nodes, g_graph_frontier, g_graph_frontier_size, g_graph_edges, g_new_frontier, g_new_frontier_size, g_graph_visited, g_amount_frontier_edges, g_cost, no_of_nodes);
}

void run_BU(dim3 grid, dim3 threads,
                    const Node* g_graph_nodes,
                    const char* g_graph_mask,
                    const int* g_graph_edges,
                    char* g_new_mask,
                    int* g_graph_visited, 
                    int* g_cost, 
                    int* g_new_frontier_size,
                    int* g_amount_frontier_edges,
                    const int no_of_nodes) {
    BFS_BU<<<grid, threads>>>(g_graph_nodes, g_graph_mask, g_graph_edges, g_new_mask, g_graph_visited, g_cost, g_new_frontier_size, g_amount_frontier_edges, no_of_nodes);
}

void run_convert_BU(dim3 grid, dim3 threads, char* g_graph_mask, int* g_new_frontier, int* g_new_frontier_size, const int no_of_nodes) {
    BFS_CONVERT_BU<<<grid, threads>>>(g_graph_mask, g_new_frontier, g_new_frontier_size, no_of_nodes);
}

void run_convert_TD(dim3 grid, dim3 threads, int* g_frontier, int* g_frontier_size, char* g_new_graph_mask) {
    BFS_CONVERT_TD<<<grid, threads>>>(g_frontier, g_frontier_size, g_new_graph_mask);
}

void run_zero(dim3 grid, dim3 threads, char* g_new_graph_mask,const int no_of_nodes) {
    ZERO<<<grid, threads>>>(g_new_graph_mask, no_of_nodes);
}