#ifndef _KERNELS_H
#define _KERNELS_H

typedef struct{
    int starting;
    int no_of_edges;
} Node;

void run_Bfs1(dim3 grid, dim3 threads, 
    const Node* g_nodes,
    const int* g_edges,
    char* g_mask, 
    char* g_new_mask, 
    char* g_visited, 
    int* g_cost, 
    char* done,
    const int no_of_nodes);       
#endif