#ifndef _KERNELS_H
#define _KERNELS_H

typedef struct{
    int starting;
    int no_of_edges;
} Node;

void run_TD(dim3 grid, dim3 threads, const Node* g_graph_nodes, int* g_graph_frontier, const int* g_graph_frontier_size, const int* g_graph_edges, int* g_new_frontier, int* g_new_frontier_size, int* g_graph_visited, int* g_amount_frontier_edges, int* g_cost, const int no_of_nodes);

void run_BU(dim3 grid, dim3 threads, const Node* g_graph_nodes, const char* g_graph_mask, const int* g_graph_edges, char* g_new_mask, int* g_graph_visited, int* g_cost, int* g_new_frontier_size, int* g_amount_frontier_edges, const int no_of_nodes);

void run_convert_BU(dim3 grid, dim3 threads, char* g_graph_mask, int* g_new_frontier, int* g_new_frontier_size, const int no_of_nodes);

void run_convert_TD(dim3 grid, dim3 threads, int* g_frontier, int* g_frontier_size, char* g_new_graph_mask);

void run_zero(dim3 grid, dim3 threads, char* g_new_graph_mask,const int no_of_nodes);   
#endif