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
//--7 parameters
__kernel void BFS_1( const __global Node* g_graph_nodes,
                    __global int* g_graph_frontier,
                    const __global int* g_graph_frontier_size,
                    const __global int* g_graph_edges,
                    __global int* g_new_frontier,
                    volatile __global int* g_new_frontier_size, 
                    volatile __global int* g_graph_visited, 
                    __global int* g_cost, 
                    const int no_of_nodes){
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
                g_cost[id] = g_cost[nodeId] + 1;

                int old = atomic_inc(g_new_frontier_size);
                g_new_frontier[old] = id;
                g_graph_visited[id] = true;
            }
        }
    }	
}


