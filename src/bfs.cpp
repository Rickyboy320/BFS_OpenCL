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
#define SOURCE 9

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
void run_bfs_cpu(int no_of_nodes, Node *h_graph_nodes, int edge_list_size, int *h_graph_edges, char* h_graph_mask, char* h_updating_graph_mask, char* h_graph_visited, int *h_cost_ref)
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

    printf("Took %d loops\n", amtloops);

#ifdef PROFILING
    timestamp_t t1 = get_timestamp();
    double secs = (t1 - t0) / 1000000.0L;
    std::cout << "\treference time (sequential)(s):" << secs << std::endl;
#endif
}
//----------------------------------------------------------
//--breadth first search on the OpenCL device
//----------------------------------------------------------
void run_bfs_opencl(int no_of_nodes, Node *h_graph_nodes, int edge_list_size, int *h_graph_edges, char *h_graph_mask, char *h_updating_graph_mask, char *h_graph_visited, int *h_cost)
{
    char h_over = false;
    cl_mem d_graph_nodes, d_graph_edges, d_graph_mask, d_updating_graph_mask, d_graph_visited, d_cost, d_over, d_frontier_edges, d_frontier_vertices;

#ifdef PROFILING
    cl_ulong kernel1_timer = 0;
    cl_ulong kernel2_timer = 0;
    cl_ulong h2d_timer = 0;
    cl_ulong d2h_timer = 0;
#endif


    try
    {
        int frontier_edges = h_graph_nodes[SOURCE].no_of_edges;
        int frontier_vertices = 1;
        int old_frontier_vertices = 1;
        int unexplored_edges = edge_list_size;

        //--1 transfer data from host to device
        _clInit();

        d_graph_nodes = _clMalloc(no_of_nodes * sizeof(Node), h_graph_nodes);
        d_graph_edges = _clMalloc(edge_list_size * sizeof(int), h_graph_edges);
        d_graph_mask = _clMallocRW(no_of_nodes * sizeof(char), h_graph_mask);
        d_updating_graph_mask = _clMallocRW(no_of_nodes * sizeof(char), h_updating_graph_mask);
        d_graph_visited = _clMallocRW(no_of_nodes * sizeof(char), h_graph_visited);
        d_cost = _clMallocRW(no_of_nodes * sizeof(int), h_cost);
        d_over = _clMallocRW(sizeof(char), &h_over);
        d_frontier_edges = _clMallocRW(sizeof(int), &frontier_edges);
        d_frontier_vertices = _clMallocRW(sizeof(int), &frontier_vertices);

        cl_event h2dpreevents[6];
        h2dpreevents[0] = _clMemcpyH2D(d_graph_nodes, no_of_nodes * sizeof(Node), h_graph_nodes);
        h2dpreevents[1] = _clMemcpyH2D(d_graph_edges, edge_list_size * sizeof(int), h_graph_edges);
        h2dpreevents[2] = _clMemcpyH2D(d_graph_mask, no_of_nodes * sizeof(char), h_graph_mask);
        h2dpreevents[3] = _clMemcpyH2D(d_updating_graph_mask, no_of_nodes * sizeof(char), h_updating_graph_mask);
        h2dpreevents[4] = _clMemcpyH2D(d_graph_visited, no_of_nodes * sizeof(char), h_graph_visited);
        h2dpreevents[5] = _clMemcpyH2D(d_cost, no_of_nodes * sizeof(int), h_cost);

#ifdef PROFILING
        clWaitForEvents(6, h2dpreevents);
        _clFinish();

        for(int i = 0; i < 6; i++) {
            cl_ulong time_start;
            cl_ulong time_end;

            clGetEventProfilingInfo(h2dpreevents[i], CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
            clGetEventProfilingInfo(h2dpreevents[i], CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

            h2d_timer += time_end - time_start;
        }
#endif
        //--2 invoke kernel
        int amtloops = 0;
        bool top_down = true;

        cl_event h2devents[3];
        cl_event kernelevents[2];
        cl_event d2hevents[3];
        do
        {
            amtloops++;

            h_over = false; 

            int zero = 0;
            h2devents[0] = _clMemcpyH2D(d_over, sizeof(char), &h_over);
            h2devents[1] = _clMemcpyH2D(d_frontier_edges, sizeof(int), &zero);
            h2devents[2] = _clMemcpyH2D(d_frontier_vertices, sizeof(int), &zero);

#ifdef PROFILING
            clWaitForEvents(3, h2devents);
            _clFinish();

            for(int i = 0; i < 3; i++) {
                cl_ulong time_start;
                cl_ulong time_end;

                clGetEventProfilingInfo(h2devents[i], CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
                clGetEventProfilingInfo(h2devents[i], CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

                h2d_timer += time_end - time_start;
            }
#endif

            bool shrinking = frontier_vertices < old_frontier_vertices;

            if (top_down && frontier_edges > unexplored_edges / ALPHA) { //&& !shrinking) {
                //printf("Switching to BU\n");
                top_down = false;
            } else if(!top_down && frontier_vertices < no_of_nodes / BETA) { //&& shrinking) {
                top_down = true;
                //printf("Switching to TD\n");
            }

            unexplored_edges -= frontier_edges;

            //printf("shrinking: %s, topdown: %s, frontier_edges: %d, unexplored_edges: %d, frontier_vertices: %d, no_of_nodes: %d\n", shrinking ? "true" : "false", top_down ? "true" : "false", frontier_edges, unexplored_edges, frontier_vertices, no_of_nodes);

            //--kernel 0 or 1 (topdown / bottom-up)
            int kernel_id = top_down ? 0 : 1;
            int kernel_idx = 0;
            _clSetArgs(kernel_id, kernel_idx++, d_graph_nodes);
            _clSetArgs(kernel_id, kernel_idx++, d_graph_edges);
            _clSetArgs(kernel_id, kernel_idx++, d_graph_mask);
            _clSetArgs(kernel_id, kernel_idx++, d_updating_graph_mask);
            _clSetArgs(kernel_id, kernel_idx++, d_graph_visited);
            _clSetArgs(kernel_id, kernel_idx++, d_cost);
            _clSetArgs(kernel_id, kernel_idx++, &no_of_nodes, sizeof(int));

            //int work_items = no_of_nodes;
            kernelevents[0] = _clInvokeKernel(kernel_id, no_of_nodes, work_group_size);
            
            //--kernel 2 (update)
            kernel_id = 2;
            kernel_idx = 0;
            _clSetArgs(kernel_id, kernel_idx++, d_graph_nodes);
            _clSetArgs(kernel_id, kernel_idx++, d_graph_mask);
            _clSetArgs(kernel_id, kernel_idx++, d_updating_graph_mask);
            _clSetArgs(kernel_id, kernel_idx++, d_graph_visited);
            _clSetArgs(kernel_id, kernel_idx++, d_over);
            _clSetArgs(kernel_id, kernel_idx++, &no_of_nodes, sizeof(int));
            _clSetArgs(kernel_id, kernel_idx++, d_frontier_vertices);
            _clSetArgs(kernel_id, kernel_idx++, d_frontier_edges);

            //work_items = no_of_nodes;
            kernelevents[1] = _clInvokeKernel(kernel_id, no_of_nodes, work_group_size);
#ifdef PROFILING
            //Force waiting for kernel to finish.
            clWaitForEvents(2, kernelevents);
            _clFinish();

            cl_ulong time_start;
            cl_ulong time_end;

            clGetEventProfilingInfo(kernelevents[0], CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
            clGetEventProfilingInfo(kernelevents[0], CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

            kernel1_timer += time_end-time_start;

            clGetEventProfilingInfo(kernelevents[1], CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
            clGetEventProfilingInfo(kernelevents[1], CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

            kernel2_timer = time_end-time_start;
            
#endif
            old_frontier_vertices = frontier_vertices;
            d2hevents[0] = _clMemcpyD2H(d_over, sizeof(char), &h_over);
            d2hevents[1] = _clMemcpyD2H(d_frontier_vertices, sizeof(int), &frontier_vertices);
            d2hevents[2] = _clMemcpyD2H(d_frontier_edges, sizeof(int), &frontier_edges);
#ifdef PROFILING
            clWaitForEvents(3, d2hevents);
            _clFinish();

            for(int i = 0; i < 3; i++) {
                cl_ulong time_start;
                cl_ulong time_end;

                clGetEventProfilingInfo(d2hevents[i], CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
                clGetEventProfilingInfo(d2hevents[i], CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

                d2h_timer += time_end - time_start;
            }
#endif
        } while (h_over);


        printf("Took %d loops\n", amtloops);
        _clFinish();

        //--3 transfer data from device to host
        _clMemcpyD2H(d_cost, no_of_nodes * sizeof(int), h_cost);
    }
    catch (std::string msg)
    {
        throw("in run_bfs_opencl -> " + msg);
    }

    //--4 release cl resources.
    _clFree(d_graph_nodes);
    _clFree(d_graph_edges);
    _clFree(d_graph_mask);
    _clFree(d_updating_graph_mask);
    _clFree(d_graph_visited);
    _clFree(d_cost);
    _clFree(d_over);
    _clRelease();

#ifdef PROFILING
    //printf("Kernel1 time is: %0.3f milliseconds \n", kernel1_timer / 1000000.0);
    //printf("Kernel2 time is: %0.3f milliseconds \n", kernel2_timer / 1000000.0);
    printf("\tTotal h2d time is: %0.3f milliseconds \n", (h2d_timer) / 1000000.0);
    printf("\tTotal kernel time is: %0.3f milliseconds \n", (kernel1_timer + kernel2_timer) / 1000000.0);
    printf("\tTotal d2h time is: %0.3f milliseconds \n", (d2h_timer) / 1000000.0);
    printf("\tTotal time: %0.3f milliseconds \n", (h2d_timer + kernel1_timer + kernel2_timer + d2h_timer) / 1000000.0);
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
    char *h_graph_mask;
    char *h_updating_graph_mask;
    char *h_graph_visited;
    int *h_graph_edges;
    int *I;
    int *J;
    double *val;

    try
    {
        if (argc < 2)
        {
            fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
            fprintf(stderr, "Flags:\n");
            fprintf(stderr, "\t-g <int>: work group size.\n");
            fprintf(stderr, "\t-d <int>: device id to use.\n");

            exit(0);
        }

        _clCmdParams(argc, argv);

        //Read in Graph from a file
        char *input_f = argv[1];
        printf("Reading File\n");
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
        I = (int *) malloc(nz * sizeof(int));
        J = (int *) malloc(nz * sizeof(int));
        val = (double *) malloc(nz * sizeof(double));

        /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
        /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
        /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

        printf("Amt nodes: %d, non-zeroes: %d\n", no_of_nodes, nz);
        std::unordered_set<int>* construction_set = new std::unordered_set<int>[no_of_nodes];

        // Reserve a rough estimate.
        for(int i = 0; i < no_of_nodes; i++) {
            construction_set[i].reserve((int) (nz / no_of_nodes));
        }
        
        if(mm_is_pattern(matcode))
        {
            for (int i = 0; i < nz; i++)
            {
                if(fscanf(fp, "%d %d\n", &I[i], &J[i]) != 2) {
                    printf("Failed to read line %d\n", i);
                }
                I[i]--;  /* adjust from 1-based to 0-based */
                J[i]--;

                construction_set[I[i]].insert(J[i]);
                construction_set[J[i]].insert(I[i]);
            }
        }
        else
        {
            for (int i = 0; i < nz; i++)
            {
                if(fscanf(fp, "%d %d %lg\n", &I[i], &J[i], &val[i]) != 3) {
                    printf("Failed to read line %d\n", i);
                }
                I[i]--;  /* adjust from 1-based to 0-based */
                J[i]--;

                construction_set[I[i]].insert(J[i]);
                construction_set[J[i]].insert(I[i]);
            }
        }
        
        printf("Done reading nodes. Converting edges...\n");
        
        if (fp !=stdin) fclose(fp);

        int edge_list_size = nz * 2;
        h_graph_edges = (int*) malloc(sizeof(int) * edge_list_size);

        // Distribute threads across multiple Blocks if necessary
        work_group_size = no_of_nodes > MAX_THREADS_PER_BLOCK ? MAX_THREADS_PER_BLOCK : no_of_nodes;

        // Allocate host memory
        h_graph_nodes = (Node *)malloc(sizeof(Node) * no_of_nodes);
        h_graph_mask = (char *)malloc(sizeof(char) * no_of_nodes);
        h_updating_graph_mask = (char *)malloc(sizeof(char) * no_of_nodes);
        h_graph_visited = (char *)malloc(sizeof(char) * no_of_nodes);

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
            h_graph_visited[i] = false;   
        }

        // Allocate mem for the result on host side
        int *h_cost = (int *)malloc(sizeof(int) * no_of_nodes);
        int *h_cost_ref = (int *)malloc(sizeof(int) * no_of_nodes);
        for (int i = 0; i < no_of_nodes; i++)
        {
            h_cost[i] = -1;
            h_cost_ref[i] = -1;
        }
        h_cost[SOURCE] = 0;
        h_cost_ref[SOURCE] = 0;

        printf("Running opencl...\n");

        //---------------------------------------------------------
        //--opencl entry
        h_graph_mask[SOURCE] = true;
        h_graph_visited[SOURCE] = true;
        run_bfs_opencl(no_of_nodes, h_graph_nodes, edge_list_size, h_graph_edges, h_graph_mask, h_updating_graph_mask, h_graph_visited, h_cost);

        //---------------------------------------------------------
        //--cpu entry
        // Initialize the memory again
        for (int i = 0; i < no_of_nodes; i++)
        {
            h_graph_mask[i] = false;
            h_updating_graph_mask[i] = false;
            h_graph_visited[i] = false;
        }

        printf("Running cpu...\n");

        // Set the SOURCE node as true in the mask
        h_graph_mask[SOURCE] = true;
        h_graph_visited[SOURCE] = true;
        run_bfs_cpu(no_of_nodes, h_graph_nodes, edge_list_size, h_graph_edges, h_graph_mask, h_updating_graph_mask, h_graph_visited, h_cost_ref);

        //---------------------------------------------------------
        //--result verification
        compare_results<int>(h_cost_ref, h_cost, no_of_nodes);
    }
    catch (std::string msg)
    {
        std::cout << "--cambine: exception in main ->" << msg << std::endl;
    }

    // Release host memory
    free(I);
    free(J);
    free(val);
    free(h_graph_nodes);
    
    free(h_graph_mask);
    free(h_updating_graph_mask);
    free(h_graph_visited);
    free(h_graph_edges);

    return 0;
}
