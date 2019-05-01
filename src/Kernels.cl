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
                    __global int* g_new_frontier_size, 
                    __global char* g_graph_visited, 
                    __global int* g_cost, 
                    const int no_of_nodes){
    int tid = get_global_id(0);
    if(tid < g_graph_frontier_size) 
    {
        int nodeId = g_graph_frontier[tid];
        Node node = g_graph_nodes[nodeId];
        printf("%p\n", node);
        //for(int i = node.starting; i < node.starting + node.no_of_edges; i++) 
        //{
        //    printf("%d\n", i);
            // int id = g_graph_edges[i];
            // if(!g_graph_visited[id])
            // {
            //     g_cost[id] = g_cost[tid] + 1;
            //     g_new_frontier[*g_new_frontier_size] = id;
            //     *g_new_frontier_size += 1;
            // }
        //}
    }	
}


