#define __CL_ENABLE_EXCEPTIONS
#include <cstdlib>
#include <iostream>
#include <string>
#include <cstring>
#include <unordered_set>

#include "CLHelper.h"
#include "util.h"
#include "matrixmarket/mmio.h"
#include <sys/time.h>

#define MAX_THREADS_PER_BLOCK 256

#define ALPHA 15
#define BETA 18
int iterations = 1;
int source = 0;

typedef unsigned long long timestamp_t;

static timestamp_t get_timestamp ()
{
    struct timeval now;
    gettimeofday(&now, NULL);
    return  now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
}

struct Node
{
    int starting;
    int no_of_edges;
};

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
    double secs = (t1 - t0) / 1000000.0L;
    std::cout << "\treference time (sequential)(s):" << secs << std::endl;
#endif
}
//----------------------------------------------------------
//--breadth first search on the OpenCL device
//----------------------------------------------------------
void run_bfs_opencl(int no_of_nodes, 
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

    cl_mem d_graph_nodes;
    cl_mem d_graph_frontier;
    cl_mem d_graph_mask;
    cl_mem d_new_mask;
    cl_mem d_graph_frontier_size;
    cl_mem d_graph_edges;
    cl_mem d_new_frontier;
    cl_mem d_new_frontier_size;
    cl_mem d_graph_visited;
    cl_mem d_amount_frontier_edges;
    cl_mem d_cost;

#ifdef PROFILING
    cl_ulong h2d_timer = 0;
    cl_ulong kernel_timer = 0;
    cl_ulong d2h_timer = 0;
#endif

    try
    {
        //--1 transfer data from host to device
        d_graph_nodes = _clCreateAndCpyMem(no_of_nodes * sizeof(Node), h_graph_nodes);
        d_graph_edges = _clCreateAndCpyMem(edge_list_size * sizeof(int), h_graph_edges);
        
        d_new_frontier = _clMallocRW(no_of_nodes * sizeof(int), h_new_graph_frontier);
        d_graph_frontier = _clCreateBuffer(CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, no_of_nodes * sizeof(int), NULL);
        d_graph_frontier_size = _clCreateBuffer(CL_MEM_READ_WRITE, sizeof(int), NULL);
        d_new_frontier_size = _clMallocRW(sizeof(int), &h_new_frontier_size);
        
        d_graph_visited = _clMallocRW(no_of_nodes * sizeof(int), h_graph_visited);

        d_graph_mask = _clCreateBuffer(CL_MEM_HOST_NO_ACCESS, no_of_nodes * sizeof(int), NULL);
        d_new_mask = _clCreateBuffer(CL_MEM_HOST_NO_ACCESS, no_of_nodes * sizeof(int), NULL);
    
        d_amount_frontier_edges = _clCreateBuffer(CL_MEM_READ_WRITE, sizeof(int), NULL);

        d_cost = _clMallocRW(no_of_nodes * sizeof(int), h_cost);

        cl_event h2dpreevents[5];
        h2dpreevents[0] = _clMemcpyH2D(d_graph_nodes, no_of_nodes * sizeof(Node), h_graph_nodes);
        h2dpreevents[1] = _clMemcpyH2D(d_new_frontier, no_of_nodes * sizeof(int), h_new_graph_frontier);
        h2dpreevents[2] = _clMemcpyH2D(d_new_frontier_size, sizeof(int), &h_new_frontier_size);
        h2dpreevents[3] = _clMemcpyH2D(d_graph_visited, no_of_nodes * sizeof(int), h_graph_visited);
        h2dpreevents[4] = _clMemcpyH2D(d_cost, no_of_nodes * sizeof(int), h_cost);
        
#ifdef PROFILING
        _clWait(5, h2dpreevents);
        _clFinish();
       
        for(int i = 0; i < 5; i++) {
            cl_ulong time_start;
            cl_ulong time_end;

            clGetEventProfilingInfo(h2dpreevents[i], CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
            clGetEventProfilingInfo(h2dpreevents[i], CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
            h2d_timer += time_end - time_start;
        }
#endif
        //--2 invoke kernel
        bool top_down = true;

        cl_event h2devents[2];
        cl_event kernelevents[5];
        cl_event d2hevents[2];
        do
        {
            amtloops++;
            
            // 'Swap' buffers
            cl_mem tmp = d_new_frontier;
            d_new_frontier = d_graph_frontier;
            d_graph_frontier = tmp;

            tmp = d_new_frontier_size;
            d_new_frontier_size = d_graph_frontier_size;
            d_graph_frontier_size = tmp;

            tmp = d_new_mask;
            d_new_mask = d_graph_mask;
            d_graph_mask = tmp;

            int zero = 0;
            int zero2 = 0;
            h2devents[0] = _clMemcpyH2D(d_new_frontier_size, sizeof(int), &zero);
            h2devents[1] = _clMemcpyH2D(d_amount_frontier_edges, sizeof(int), &zero2);

#ifdef PROFILING
            _clWait(2, h2devents);

            _clFinish();

            for(int i = 0; i < 2; i++) {
                cl_ulong time_start;
                cl_ulong time_end;

                clGetEventProfilingInfo(h2devents[i], CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
                clGetEventProfilingInfo(h2devents[i], CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

                h2d_timer += time_end - time_start;
            }
#endif
            clReleaseEvent(h2devents[0]);
            clReleaseEvent(h2devents[1]);

            bool shrinking = h_new_frontier_size < old_frontier_vertices;
            int eventindex = 0;


            if (!has_bottom_upped && top_down && frontier_edges > unexplored_edges / ALPHA && !shrinking) {
                top_down = false;
                has_bottom_upped = true;
#ifdef VERBOSE
                printf("Switching to BU\n");
#endif
                // Reset new bitmap
                _clSetArgs(4, 0, d_graph_mask);
                _clSetArgs(4, 1, &no_of_nodes, sizeof(int));
                kernelevents[eventindex++] =_clInvokeKernel(4, no_of_nodes, work_group_size);

                int kernel_idx = 0;
                _clSetArgs(3, kernel_idx++, d_graph_frontier);
                _clSetArgs(3, kernel_idx++, d_graph_frontier_size);
                _clSetArgs(3, kernel_idx++, d_graph_mask);
                kernelevents[eventindex++] = _clInvokeKernel(3, no_of_nodes, work_group_size);

            } else if(!top_down && h_new_frontier_size < no_of_nodes / BETA && shrinking) {
                top_down = true;
#ifdef VERBOSE
                printf("Switching to TD\n");
#endif
                int kernel_idx = 0;
                _clSetArgs(2, kernel_idx++, d_graph_mask);
                _clSetArgs(2, kernel_idx++, d_graph_frontier);
                _clSetArgs(2, kernel_idx++, d_graph_frontier_size);
                _clSetArgs(2, kernel_idx++, &no_of_nodes, sizeof(int));
                kernelevents[eventindex++] = _clInvokeKernel(2, no_of_nodes, work_group_size);
            }

            unexplored_edges -= frontier_edges;

            //--kernel 0 or 1 (topdown / bottom-up)
            if(top_down) {
                int kernel_idx = 0;
                _clSetArgs(0, kernel_idx++, d_graph_nodes);
                _clSetArgs(0, kernel_idx++, d_graph_frontier);
                _clSetArgs(0, kernel_idx++, d_graph_frontier_size);
                _clSetArgs(0, kernel_idx++, d_graph_edges);
                _clSetArgs(0, kernel_idx++, d_new_frontier);
                _clSetArgs(0, kernel_idx++, d_new_frontier_size);
                _clSetArgs(0, kernel_idx++, d_graph_visited);
                _clSetArgs(0, kernel_idx++, d_amount_frontier_edges);
                _clSetArgs(0, kernel_idx++, d_cost);
                _clSetArgs(0, kernel_idx++, &no_of_nodes, sizeof(int));

                kernelevents[eventindex++] = _clInvokeKernel(0, no_of_nodes, work_group_size);
            } else {
                // Reset new bitmap
                _clSetArgs(4, 0, d_new_mask);
                _clSetArgs(4, 1, &no_of_nodes, sizeof(int));
                kernelevents[eventindex++] = _clInvokeKernel(4, no_of_nodes, work_group_size);

                int kernel_idx = 0;
                _clSetArgs(1, kernel_idx++, d_graph_nodes);
                _clSetArgs(1, kernel_idx++, d_graph_mask);
                _clSetArgs(1, kernel_idx++, d_graph_edges);
                _clSetArgs(1, kernel_idx++, d_new_mask);
                _clSetArgs(1, kernel_idx++, d_graph_visited);
                _clSetArgs(1, kernel_idx++, d_cost);
                _clSetArgs(1, kernel_idx++, d_new_frontier_size);
                _clSetArgs(1, kernel_idx++, d_amount_frontier_edges);
                _clSetArgs(1, kernel_idx++, &no_of_nodes, sizeof(int));

                kernelevents[eventindex++] = _clInvokeKernel(1, no_of_nodes, work_group_size);
            }            
            //int work_items = no_of_nodes;
            // TODO: no_of_nodes should be frontier size;
#ifdef PROFILING
            //Force waiting for kernels to finish.
            _clWait(eventindex, kernelevents);
            _clFinish();

            for(int i = 0; i < eventindex; i++) {            
                cl_ulong time_start;
                cl_ulong time_end;

                clGetEventProfilingInfo(kernelevents[i], CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
                clGetEventProfilingInfo(kernelevents[i], CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

                kernel_timer += time_end-time_start;
            }

#endif
            clReleaseEvent(kernelevents[0]);

            old_frontier_vertices = h_new_frontier_size;
            d2hevents[0] = _clMemcpyD2H(d_new_frontier_size, sizeof(int), &h_new_frontier_size);
            d2hevents[1] = _clMemcpyD2H(d_amount_frontier_edges, sizeof(int), &frontier_edges);
#ifdef PROFILING
            _clWait(2, d2hevents);
            _clFinish();

            for(int i = 0; i < 2; i++) {
                cl_ulong time_start;
                cl_ulong time_end;

                clGetEventProfilingInfo(d2hevents[i], CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
                clGetEventProfilingInfo(d2hevents[i], CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

                d2h_timer += time_end - time_start;
            }

#endif
            clReleaseEvent(d2hevents[0]);
            clReleaseEvent(d2hevents[1]);
        } while (h_new_frontier_size > 0);

        _clFinish();

        //--3 transfer data from device to host
        cl_event d2hevent[1];
        d2hevent[0] = _clMemcpyD2H(d_cost, no_of_nodes * sizeof(int), h_cost);

#ifdef PROFILING
        _clWait(1, d2hevent);
        _clFinish();

        cl_ulong time_start;
        cl_ulong time_end;

        clGetEventProfilingInfo(d2hevent[0], CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
        clGetEventProfilingInfo(d2hevent[0], CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

        d2h_timer += time_end - time_start;
#endif
        clReleaseEvent(d2hevent[0]);
    }
    catch (std::string msg)
    {
        throw("in run_bfs_opencl -> " + msg);
    }

    //--4 release cl resources.
    _clFree(d_graph_nodes);
    _clFree(d_graph_frontier);
    _clFree(d_graph_frontier_size);
    _clFree(d_graph_edges);
    _clFree(d_new_frontier);
    _clFree(d_new_frontier_size);
    _clFree(d_graph_visited);
    _clFree(d_cost);

#ifdef PROFILING
    
    #ifdef VERBOSE
    printf("Took %d loops\n", amtloops);
    printf("\tTotal h2d time is: %0.3f milliseconds \n", (h2d_timer) / 1000000.0);
    printf("\tTotal kernel time is: %0.3f milliseconds \n", (kernel_timer) / 1000000.0);
    printf("\tTotal d2h time is: %0.3f milliseconds \n", (d2h_timer) / 1000000.0);
    printf("\tTotal time: %0.3f milliseconds \n", (h2d_timer + kernel_timer + d2h_timer) / 1000000.0);
    #else
    printf("%0.3f %0.3f %0.3f %0.3f\n", (h2d_timer) / 1000000.0, (kernel_timer) / 1000000.0, (d2h_timer) / 1000000.0, (h2d_timer + kernel_timer + d2h_timer) / 1000000.0);
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

        _clCmdParams(argc, argv, &source, &iterations);

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
        work_group_size = no_of_nodes > MAX_THREADS_PER_BLOCK ? MAX_THREADS_PER_BLOCK : no_of_nodes;

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

        _clInit();

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
            printf("Running opencl...\n");
    #endif

            //---------------------------------------------------------
            //--opencl entry
            h_cost[i][source] = 0;
            h_graph_frontier[0] = source;
            h_graph_visited[source] = 1;
            run_bfs_opencl(no_of_nodes, h_graph_nodes, h_graph_frontier, 1, edge_list_size, h_graph_edges, h_graph_visited, h_cost[i]);
        }

        _clRelease();

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
