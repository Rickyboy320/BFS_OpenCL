#ifndef _KERNELS_H
#define _KERNELS_H

typedef struct{
    int starting;
    int no_of_edges;
} Node;

void run_Bfs1(dim3 grid, dim3 threads, 
    const Node* g_graph_nodes,
    const int* g_graph_edges,
    char* g_graph_mask, 
    char* g_updating_graph_mask, 
    char* g_graph_visited, 
    int* g_cost, 
    const int no_of_nodes);

void run_Bfs2(dim3 grid, dim3 threads,
    char* g_graph_mask, 
    char* g_updating_graph_mask, 
    char* g_graph_visited, 
    char* g_over,
    const int no_of_nodes);        
#endif