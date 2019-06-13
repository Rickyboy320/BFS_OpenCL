#pragma OPENCL EXTENSION cl_khr_byte_addressable_store: enable

typedef struct{
    int starting;
    int no_of_edges;
} Node;

__kernel void BFS_1(const __global Node* g_nodes,
                    const __global int* g_edges,
                    __global char* g_mask, 
                    __global char* g_new_mask, 
                    __global char* g_visited, 
                    __global int* g_cost, 
                    __global char* done,
                    const int no_of_nodes){
    int tid = get_global_id(0);
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

__kernel void EMP(const __global Node* g_nodes,
                    const __global int* g_edges,
                    __global char* g_mask, 
                    __global char* g_new_mask, 
                    __global char* g_visited, 
                    __global int* g_cost, 
                    __global char* done,
                    const int no_of_nodes)
{
    if(*done == false) {
        *done = true;
        return;
    }
    *done = false;
}