#pragma OPENCL EXTENSION cl_khr_byte_addressable_store: enable

typedef struct{
    int starting;
    int no_of_edges;
} Node;

__kernel void BFS_TD(const __global Node* g_graph_nodes,
                    const __global int* g_graph_edges,
                    __global char* g_graph_mask, 
                    __global char* g_updating_graph_mask, 
                    __global char* g_graph_visited, 
                    __global int* g_cost, 
                    const int no_of_nodes){
    int tid = get_global_id(0);
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

__kernel void BFS_BU(const __global Node* g_graph_nodes,
                    const __global int* g_graph_edges,
                    __global char* g_graph_mask, 
                    __global char* g_updating_graph_mask, 
                    __global char* g_graph_visited, 
                    __global int* g_cost, 
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
                g_updating_graph_mask[tid] = true;
                break;
            }
        }
    }	
}

/**
 * Update the graph. All nodes that have been visited are now marked as such.
 */
__kernel void BFS_UPDATE(const __global Node* g_graph_nodes,
                        __global char* g_graph_mask, 
                        __global char* g_updating_graph_mask, 
                        __global char* g_graph_visited, 
                        const int no_of_nodes,
                        __global int* frontier_vertices,
                        __global int* frontier_edges) {
    int tid = get_global_id(0);
    if(tid < no_of_nodes)
    {
        if(g_graph_mask[tid])
        {
            g_graph_mask[tid] = false;
            return;
        }
        
        if(g_updating_graph_mask[tid])
        {
            g_graph_mask[tid]=true;
            g_graph_visited[tid]=true;
            g_updating_graph_mask[tid]=false;

            atomic_add(frontier_edges, g_graph_nodes[tid].no_of_edges);
            atomic_inc(frontier_vertices);
        }
    }
    //TODO: potential optimization: no longer compute frontier_edges and frontier_vertices  after BU has been initiated.
}