#include <cstdlib>
#include <iostream>
#include <string>
#include <unordered_set>

#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

#include "kernels.cuh"

// Helper
#include "util.h"
#include "matrixmarket/mmio.h"
#include <sys/time.h>

#define MAX_THREADS_PER_BLOCK 256

#define ALPHA 15
#define BETA 18
int iterations = 1;
int source = 0;
int deviceid = 0;
int num_of_blocks = 1;
int num_of_threads_per_block = MAX_THREADS_PER_BLOCK;
bool cpu = false;

typedef unsigned long long timestamp_t;

static timestamp_t get_timestamp ()
{
    struct timeval now;
    gettimeofday(&now, NULL);
    return now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
}

//----------------------------------------------------------
//--Reference bfs on cpu
//--programmer:	jianbin
//--date:	26/01/2011
//--note: width is changed to the new_width
//----------------------------------------------------------
void run_bfs_cpu(int no_of_nodes, Node *h_graph_nodes, int edge_list_size, int *h_graph_edges, char* h_graph_mask, char* h_updating_graph_mask, int* h_graph_visited, int *h_cost_ref)
{
#ifdef PROFILING
    timestamp_t t0 = get_timestamp();
#endif

    int amtloops = 0;
    char shouldContinue;
    do
    {
        amtloops++;
        shouldContinue = false;
        for (int tid = 0; tid < no_of_nodes; tid++)
        {
            if (h_graph_mask[tid])
            {
                h_graph_mask[tid] = false;
                for (int i = h_graph_nodes[tid].starting; i < h_graph_nodes[tid].starting + h_graph_nodes[tid].no_of_edges; i++)
                {
                    int id = h_graph_edges[i]; //--cambine: node id is connected with node tid
                    if (!h_graph_visited[id])
                    {
                        h_cost_ref[id] = h_cost_ref[tid] + 1;
                        h_updating_graph_mask[id] = true;
                    }
                }
            }
        }

        for (int tid = 0; tid < no_of_nodes; tid++)
        {
            if (h_updating_graph_mask[tid] == true)
            {
                h_graph_mask[tid] = true;
                h_graph_visited[tid] = true;
                shouldContinue = true;
                h_updating_graph_mask[tid] = false;
            }
        }
    } while (shouldContinue);

#ifdef VERBOSE
    printf("Took %d loops\n", amtloops);
#endif
#ifdef PROFILING
    timestamp_t t1 = get_timestamp();
    double secs = (t1 - t0) / 1000.0L;
    std::cout << "\treference time (sequential)(ms):" << secs << std::endl;
#endif
}
//----------------------------------------------------------
//--breadth first search on the CUDA device
//----------------------------------------------------------
void run_bfs_cuda(int no_of_nodes, 
                  Node *h_graph_nodes, 
                  int* h_new_graph_frontier, 
                  int h_new_frontier_size, 
                  int edge_list_size, 
                  int *h_graph_edges, 
                  int *h_graph_visited, 
                  int *h_cost)
{
    unsigned int amtloops = 0;
    unsigned int old_frontier_vertices = 1;
    unsigned int frontier_edges = h_graph_nodes[source].no_of_edges;
    unsigned int unexplored_edges = edge_list_size;
    bool has_bottom_upped = false;

    Node* d_graph_nodes;
    int* d_graph_frontier;
    char* d_graph_mask;
    char* d_new_mask;
    int* d_graph_frontier_size;
    int* d_graph_edges;
    int* d_new_frontier;
    int* d_new_frontier_size;
    int* d_graph_visited;
    int* d_amount_frontier_edges;
    int* d_cost;


#ifdef PROFILING
    float kernel_timer = 0;
    float h2d_timer = 0;
    float d2h_timer = 0;

    cudaEvent_t start;
    checkCudaErrors(cudaEventCreate(&start));

    cudaEvent_t stop;
    checkCudaErrors(cudaEventCreate(&stop));
#endif

    try
    {
        // checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_graph_nodes), no_of_nodes * sizeof(Node)));
        //         checkCudaErrors(cudaMemcpy(d_graph_nodes, h_graph_nodes, no_of_nodes * sizeof(Node), cudaMemcpyHostToDevice));


        //--1 transfer data from host to device
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_graph_nodes), no_of_nodes * sizeof(Node)));
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_graph_edges), edge_list_size * sizeof(int)));
        
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_new_frontier), no_of_nodes * sizeof(int)));
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_graph_frontier), no_of_nodes * sizeof(int)));
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_graph_frontier_size), no_of_nodes * sizeof(int)));
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_graph_frontier_size), sizeof(int)));
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_new_frontier_size), sizeof(int)));

        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_graph_visited), no_of_nodes * sizeof(int)));        
 
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_graph_mask), no_of_nodes * sizeof(char)));
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_new_mask), no_of_nodes * sizeof(char)));

        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_amount_frontier_edges), sizeof(int)));    
 
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_cost), no_of_nodes * sizeof(int)));

#ifdef PROFILING
        checkCudaErrors(cudaEventRecord(start, NULL));
#endif
        checkCudaErrors(cudaMemcpy(d_graph_nodes, h_graph_nodes, no_of_nodes * sizeof(Node), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_graph_edges, h_graph_edges, edge_list_size * sizeof(int), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_new_frontier, h_new_graph_frontier, no_of_nodes * sizeof(int), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_new_frontier_size, &h_new_frontier_size, sizeof(int), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_graph_visited, h_graph_visited, no_of_nodes * sizeof(int), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_cost, h_cost, no_of_nodes * sizeof(int), cudaMemcpyHostToDevice));

#ifdef PROFILING
        checkCudaErrors(cudaEventRecord(stop, NULL));
        checkCudaErrors(cudaEventSynchronize(stop));

        float msecTotal = 0.0f;
        checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
        h2d_timer += msecTotal;
#endif
        //--2 invoke kernel
        bool top_down = true;
        dim3 grid(num_of_blocks, 1, 1);
        dim3 threads(num_of_threads_per_block, 1, 1);

        do
        {
            amtloops++;
            
            // 'Swap' buffers
            int* tmp = d_new_frontier;
            d_new_frontier = d_graph_frontier;
            d_graph_frontier = tmp;

            tmp = d_new_frontier_size;
            d_new_frontier_size = d_graph_frontier_size;
            d_graph_frontier_size = tmp;

            char* tmp2 = d_new_mask;
            d_new_mask = d_graph_mask;
            d_graph_mask = tmp2;

            int zero = 0;
            int zero2 = 0;
#ifdef PROFILING
            checkCudaErrors(cudaEventRecord(start, NULL));
#endif
            checkCudaErrors(cudaMemcpy(d_new_frontier_size, &zero, sizeof(int), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_amount_frontier_edges, &zero2, sizeof(int), cudaMemcpyHostToDevice));
#ifdef PROFILING
            checkCudaErrors(cudaEventRecord(stop, NULL));
            checkCudaErrors(cudaEventSynchronize(stop));

            float msecTotal = 0.0f;
            checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
            h2d_timer += msecTotal;
#endif

            bool shrinking = h_new_frontier_size < old_frontier_vertices;

            if (!has_bottom_upped && top_down && frontier_edges > unexplored_edges / ALPHA && !shrinking) {
                top_down = false;
                has_bottom_upped = true;
#ifdef VERBOSE
                printf("Switching to BU\n");
#endif
                // Reset new bitmap
                run_zero(grid, threads, d_graph_mask, no_of_nodes);
                run_convert_TD(grid, threads, d_graph_frontier, d_graph_frontier_size, d_graph_mask);
            } else if(!top_down && h_new_frontier_size < no_of_nodes / BETA && shrinking) {
                top_down = true;
#ifdef VERBOSE
                printf("Switching to TD\n");
#endif
                run_convert_BU(grid, threads, d_graph_mask, d_graph_frontier, d_graph_frontier_size, no_of_nodes);
            }

            unexplored_edges -= frontier_edges;

            //--kernel 0 or 1 (topdown / bottom-up)
            if(top_down) {
                run_TD(grid, threads, d_graph_nodes, d_graph_frontier, d_graph_frontier_size, d_graph_edges, d_new_frontier, d_new_frontier_size, d_graph_visited, d_amount_frontier_edges, d_cost, no_of_nodes);
            } else {
                // Reset new bitmap
                run_zero(grid, threads, d_new_mask, no_of_nodes);
                run_BU(grid, threads, d_graph_nodes, d_graph_mask, d_graph_edges, d_new_mask, d_graph_visited, d_cost, d_new_frontier_size, d_amount_frontier_edges, no_of_nodes);
           }            
            // TODO: no_of_nodes should be frontier size;

#ifdef PROFILING
            checkCudaErrors(cudaEventRecord(stop, NULL));
            checkCudaErrors(cudaEventSynchronize(stop));

            msecTotal = 0.0f;
            checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
            kernel_timer += msecTotal;

            checkCudaErrors(cudaEventRecord(start, NULL));
#endif
            old_frontier_vertices = h_new_frontier_size;
            checkCudaErrors(cudaMemcpy(&h_new_frontier_size, d_new_frontier_size, sizeof(int), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(&frontier_edges, d_amount_frontier_edges, sizeof(int), cudaMemcpyDeviceToHost));

#ifdef PROFILING
            checkCudaErrors(cudaEventRecord(stop, NULL));
            checkCudaErrors(cudaEventSynchronize(stop));

            msecTotal = 0.0f;
            checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
            d2h_timer += msecTotal;
#endif
        } while (h_new_frontier_size > 0);

#ifdef VERBOSE
        printf("Took %d loops\n", amtloops);
#endif

#ifdef PROFILING
        checkCudaErrors(cudaEventRecord(start, NULL));
#endif
        checkCudaErrors(cudaMemcpy(h_cost, d_cost, no_of_nodes * sizeof(int), cudaMemcpyDeviceToHost));

#ifdef PROFILING
        checkCudaErrors(cudaEventRecord(stop, NULL));
        checkCudaErrors(cudaEventSynchronize(stop));

        msecTotal = 0.0f;
        checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
        d2h_timer += msecTotal;
#endif
    }
    catch (std::string msg)
    {
        throw("in run_bfs_cuda -> " + msg);
    }

    //--4 release cuda resources.
    checkCudaErrors(cudaFree(d_graph_nodes));
    checkCudaErrors(cudaFree(d_graph_frontier));
    checkCudaErrors(cudaFree(d_graph_frontier_size));
    checkCudaErrors(cudaFree(d_graph_edges));
    checkCudaErrors(cudaFree(d_new_frontier));
    checkCudaErrors(cudaFree(d_new_frontier_size));
    checkCudaErrors(cudaFree(d_graph_visited));
    checkCudaErrors(cudaFree(d_cost));

#ifdef PROFILING    
    #ifdef VERBOSE
    printf("\tTotal h2d time is: %0.3f milliseconds \n", h2d_timer);
    printf("\tTotal kernel time is: %0.3f milliseconds \n", kernel_timer);
    printf("\tTotal d2h time is: %0.3f milliseconds \n", d2h_timer);
    printf("\tTotal time: %0.3f milliseconds \n", (h2d_timer + kernel_timer + d2h_timer));
    #else
    printf("%0.3f %0.3f %0.3f %0.3f\n", (h2d_timer), (kernel_timer), (d2h_timer), (h2d_timer + kernel_timer + d2h_timer));
    #endif
#endif
}
//----------------------------------------------------------
//--cambine:	main function
//--author:		created by Jianbin Fang
//--date:		25/01/2011
//----------------------------------------------------------
int main(int argc, char *argv[])
{
    MM_typecode matcode;

    int no_of_nodes;

    Node *h_graph_nodes;
    int *h_graph_frontier;
    char *h_graph_mask;
    char *h_updating_graph_mask;
    int *h_graph_visited;
    int *h_graph_edges;
    double *val;

    try
    {
        if (argc < 2)
        {
            fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
            fprintf(stderr, "Flags:\n");
            fprintf(stderr, "\t-g <int>: work group size.\n");
            fprintf(stderr, "\t-d <int>: device id to use.\n");
            fprintf(stderr, "\t-c: use cpu instead of gpu.\n");
            fprintf(stderr, "\t-s <int>: use value as source node (def 0).\n");
            fprintf(stderr, "\t-i <int>: use amount of iterations (def 1).\n");
            exit(0);
        }

        //TODO
        int work_group_size = 0;
        cmdParams(argc, argv, &source, &iterations, &work_group_size, &deviceid, &cpu);

        //Read in Graph from a file
        char *input_f = argv[1];
        printf("%s\n", input_f);

        FILE *fp = fopen(input_f, "r");
        if (!fp)
        {
            printf("Error Reading graph file\n");
            return 1;
        }

        if (mm_read_banner(fp, &matcode) != 0)
        {
            printf("Could not process Matrix Market banner.\n");
            exit(1);
        }

        // Only supports a subset of the Matrix Market data types.
        if (mm_is_complex(matcode) && mm_is_matrix(matcode) && 
                mm_is_sparse(matcode) )
        {
            printf("Sorry, this application does not support ");
            printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
            exit(1);
        }

        // find out size of sparse matrix ....
        int N, nz;   
        if (mm_read_mtx_crd_size(fp, &no_of_nodes, &N, &nz) != 0)
            exit(1);

        if(no_of_nodes != N) {
            printf("[WARNING] Not sure if non-square matrices work properly...");
        }

        /* reserve memory for matrices */
        val = (double *) malloc(nz * sizeof(double));

        /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
        /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
        /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

#ifdef VERBOSE
        printf("Amt nodes: %d, non-zeroes: %d\n", no_of_nodes, nz);
#endif
        std::unordered_set<int>* construction_set = new std::unordered_set<int>[no_of_nodes];

        // Reserve a rough estimate.
        for(int i = 0; i < no_of_nodes; i++) {
            construction_set[i].reserve((int) (nz / no_of_nodes));
        }
        
        if(mm_is_pattern(matcode))
        {
            for (int i = 0; i < nz; i++)
            {
                int x, y;
                if(fscanf(fp, "%d %d\n", &x, &y) != 2) {
                    printf("Failed to read line %d\n", i);
                }
                x--;  /* adjust from 1-based to 0-based */
                y--;

                construction_set[x].insert(y);
                construction_set[y].insert(x);
            }
        }
        else
        {
            for (int i = 0; i < nz; i++)
            {
                int x, y;
                if(fscanf(fp, "%d %d %lg\n", &x, &y, &val[i]) != 3) {
                    printf("Failed to read line %d\n", i);
                }
                x--;  /* adjust from 1-based to 0-based */
                y--;

                construction_set[x].insert(y);
                construction_set[y].insert(x);
            }
        }
        
#ifdef VERBOSE
        printf("Done reading nodes. Converting edges...\n");
#endif

        if (fp !=stdin) fclose(fp);

        int edge_list_size = nz * 2;
        h_graph_edges = (int*) malloc(sizeof(int) * edge_list_size);

        // Distribute threads across multiple Blocks if necessary
        if (no_of_nodes > MAX_THREADS_PER_BLOCK) {
            num_of_blocks = (int)ceil(no_of_nodes / (double)MAX_THREADS_PER_BLOCK);
            num_of_threads_per_block = MAX_THREADS_PER_BLOCK;
        }


        // Allocate host memory
        h_graph_nodes = (Node *)malloc(sizeof(Node) * no_of_nodes);
        h_graph_mask = (char *)malloc(sizeof(char) * no_of_nodes);
        h_graph_frontier = (int *)malloc(sizeof(int) * no_of_nodes);
        h_updating_graph_mask = (char *)malloc(sizeof(char) * no_of_nodes);
        h_graph_visited = (int *)malloc(sizeof(int) * no_of_nodes);

        int index = 0;
        for (int i = 0; i < no_of_nodes; i++)
        {
            h_graph_nodes[i].starting = index;
            h_graph_nodes[i].no_of_edges = construction_set[i].size();
            std::copy(construction_set[i].begin(), construction_set[i].end(), &h_graph_edges[index]);
            index += construction_set[i].size();
            construction_set[i].clear();

            h_graph_mask[i] = false;
            h_updating_graph_mask[i] = false;
            h_graph_visited[i] = 0;   
        }

        //cudaInit();

        // Allocate mem for the result on host side and run bfs
        int **h_cost;
        h_cost = (int**) malloc(iterations * sizeof(int*));    
        for(int i = 0; i < iterations; i++)
        {    
            h_cost[i] = (int*) malloc(no_of_nodes * sizeof(int));
            for (int j = 0; j < no_of_nodes; j++)
            {
                h_cost[i][j] = -1;
            }

    #ifdef VERBOSE
            printf("Running cuda...\n");
    #endif

            //---------------------------------------------------------
            //--cuda entry
            h_cost[i][source] = 0;
            h_graph_frontier[0] = source;
            h_graph_visited[source] = 1;
            run_bfs_cuda(no_of_nodes, h_graph_nodes, h_graph_frontier, 1, edge_list_size, h_graph_edges, h_graph_visited, h_cost[i]);
        }

        //cudaRelease();

#ifndef NO_CHECK
        //---------------------------------------------------------
        //--cpu entry
        // Initialize the memory again

        int *h_cost_ref = (int *)malloc(sizeof(int) * no_of_nodes);
        int *h_ref_graph_visited = (int *)malloc(sizeof(int) * no_of_nodes);

        for (int i = 0; i < no_of_nodes; i++)
        {
            h_cost_ref[i] = -1;
            h_graph_mask[i] = false;
            h_updating_graph_mask[i] = false;
            h_ref_graph_visited[i] = false;
        }

        #ifdef VERBOSE
            printf("Running cpu...\n");
        #endif
        
        // Set the source node as true in the mask
        h_cost_ref[source] = 0;
        h_graph_mask[source] = true;
        h_graph_visited[source] = true;
        run_bfs_cpu(no_of_nodes, h_graph_nodes, edge_list_size, h_graph_edges, h_graph_mask, h_updating_graph_mask, h_graph_visited, h_cost_ref);
        //---------------------------------------------------------
        //--result verification
        for(int i = 0; i < iterations; i++) {
            compare_results<int>(h_cost_ref, h_cost[i], no_of_nodes);
            free(h_cost[i]);
        }
        free(h_cost);
        free(h_ref_graph_visited);
#endif
    }
    catch (std::string msg)
    {
        std::cout << "--cambine: exception in main ->" << msg << std::endl;
    }

    // Release host memory
    free(val);
    free(h_graph_nodes);
    
    free(h_graph_mask);
    free(h_updating_graph_mask);
    free(h_graph_visited);
    free(h_graph_edges);

    return 0;
}
