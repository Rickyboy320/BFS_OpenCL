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
__kernel void BFS_TD(const __global Node* g_graph_nodes,
                    const __global int* g_graph_edges,
                    __global char* g_graph_mask, 
                    __global char* g_new_graph_mask, 
                    __global char* g_graph_visited, 
                    __global int* g_cost, 
                    const int no_of_nodes){
    int tid = get_global_id(0);
    if(tid < no_of_nodes && g_graph_mask[tid]) 
    {
        g_graph_mask[tid]=false;
        //TODO: Vectorize
        for(int i = g_graph_nodes[tid].starting; i < g_graph_nodes[tid].starting + g_graph_nodes[tid].no_of_edges; i++) 
        {
            int id = g_graph_edges[i];
            if(!g_graph_visited[id] && !g_updating_graph_mask[id])
            {
                g_cost[id] = g_cost[tid] + 1;
                g_new_graph_mask[id] = true;
            }
        }

        //Restoration
        for(int i = 0; i < no_of_nodes; i++)
        {
        }
    }	
}

//--7 parameters
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
            }
        }
    }	
}

//--7 parameters
/**
 * Update the graph. All nodes that have been visited are now marked as such.
 */
__kernel void BFS_UPDATE(const __global Node* g_graph_nodes,
                        __global char* g_graph_mask, 
                        __global char* g_updating_graph_mask, 
                        __global char* g_graph_visited, 
                        __global char* g_over,
                        const int no_of_nodes,
                        __global int* frontier_vertices,
                        __global int* frontier_edges) {
    int tid = get_global_id(0);
    if(tid < no_of_nodes && g_updating_graph_mask[tid])
    {
        g_graph_mask[tid]=true;
        g_graph_visited[tid]=true;
        *g_over=true;
        g_updating_graph_mask[tid]=false;

        *frontier_edges += g_graph_nodes[tid].no_of_edges;
        *frontier_vertices += 1;
    }
}


 