/* ============================================================
//--cambine: kernel funtion of Breadth-First-Search
//--author:	created by Jianbin Fang
//--date:	06/12/2010
============================================================ */
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store: enable
//Structure to hold a node information
typedef struct{
    int starting;
    int no_of_edges;
} Node;

__kernel void BFS_TD(const __global Node* g_graph_nodes,
                    __global int* g_graph_frontier,
                    const __global int* g_graph_frontier_size,
                    const __global int* g_graph_edges,
                    __global int* g_new_frontier,
                    volatile __global int* g_new_frontier_size, 
                    volatile __global int* g_graph_visited, 
                    __global int* g_cost, 
                    const int no_of_nodes) {
    int tid = get_global_id(0);
    if(tid < *g_graph_frontier_size) 
    {
        int nodeId = g_graph_frontier[tid];
        Node node = g_graph_nodes[nodeId];
        for(int i = node.starting; i < node.starting + node.no_of_edges; i++) 
        {
            int id = g_graph_edges[i];
            if(atomic_xchg(&g_graph_visited[id], 1) == 0)
            {
                int old = atomic_inc(g_new_frontier_size);
                g_cost[id] = g_cost[nodeId] + 1;

                g_new_frontier[old] = id;
                g_graph_visited[id] = true;
            }
        }
    }	
}

__kernel void BFS_BU(const __global Node* g_graph_nodes,
                    const __global char* g_graph_mask,
                    const __global int* g_graph_edges,
                    __global char* g_new_mask,
                    __global int* g_graph_visited, 
                    __global int* g_cost, 
                    __global int* g_new_frontier_size,
                    const int no_of_nodes){
    int tid = get_global_id(0);
    
    // Check if this node has not been visited yet (untraversed)
    if(tid < no_of_nodes && !g_graph_visited[tid]) 
    {
        // Loop over its edges
        for(int i = g_graph_nodes[tid].starting; i < g_graph_nodes[tid].starting + g_graph_nodes[tid].no_of_edges; i++) 
        {
            // If a neighbour is part of the current frontier (mask)
            int id = g_graph_edges[i];
            if(g_graph_mask[id])
            {
                // Increment cost based on parent, and set this as to-be-updated.
                g_cost[tid] = g_cost[id] + 1;

                g_new_mask[tid] = true;
                g_graph_visited[tid] = true;
                atomic_inc(g_new_frontier_size);
                break;
            }
        }
    }	
}

__kernel void BFS_CONVERT_BU(__global char* g_graph_mask,
                             __global int* g_new_frontier,
                            volatile __global int* g_new_frontier_size,
                            const int no_of_nodes) {
    int tid = get_global_id(0);
    if(tid < no_of_nodes && g_graph_mask[tid]) {
        int old = atomic_inc(g_new_frontier_size);
        g_new_frontier[old] = tid;
    }
}

__kernel void BFS_CONVERT_TD(__global int* g_frontier,
                            volatile __global int* g_frontier_size,
                            __global char* g_new_graph_mask) {
    int tid = get_global_id(0);
    if(tid < *g_frontier_size) {
        g_new_graph_mask[g_frontier[tid]] = true;
    }
}

__kernel void ZERO(__global char* g_new_graph_mask,
                   const int no_of_nodes) {
    int tid = get_global_id(0);
    if(tid < no_of_nodes) {
        g_new_graph_mask[tid] = false;
    }
}