#pragma OPENCL EXTENSION cl_khr_byte_addressable_store: enable

typedef struct{
    int starting;
    int no_of_edges;
} Node;

__kernel void BFS_1( const __global Node* g_nodes,
                    __global int* g_frontier,
                    const __global int* g_frontier_size,
                    const __global int* g_edges,
                    __global int* g_new_frontier,
                    volatile __global int* g_new_frontier_size, 
                    volatile __global int* g_visited, 
                    __global int* g_cost, 
                    const int no_of_nodes){
    int tid = get_global_id(0);
    if(tid < *g_frontier_size) 
    {
        int nodeId = g_frontier[tid];
        Node node = g_nodes[nodeId];
        for(int i = node.starting; i < node.starting + node.no_of_edges; i++) 
        {
            int id = g_edges[i];
            if(atomic_xchg(&g_visited[id], 1) == 0)
            {
                int old = atomic_inc(g_new_frontier_size);
                g_cost[id] = g_cost[nodeId] + 1;

                g_new_frontier[old] = id;
                g_visited[id] = true;
            }
        }
    }	
}


