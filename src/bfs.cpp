#define __CL_ENABLE_EXCEPTIONS
#include <cstdlib>
#include <iostream>
#include <string>
#include <cstring>
#include <set>

#ifdef PROFILING
#include "timer.h"
#endif

#include "CLHelper.h"
#include "util.h"
#include "matrixmarket/mmio.h"


#define MAX_THREADS_PER_BLOCK 256

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
        timer cpu_timer;
        double cpu_time = 0.0;
        cpu_timer.reset();
        cpu_timer.start();
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
        cpu_timer.stop();
        cpu_time = cpu_timer.getTimeInSeconds();
        std::cout << "reference time (sequential)(s):" << cpu_time << std::endl;
    #endif
}
//----------------------------------------------------------
//--breadth first search on the OpenCL device
//----------------------------------------------------------
void run_bfs_opencl(int no_of_nodes, Node *h_graph_nodes, int edge_list_size, int *h_graph_edges, char *h_graph_mask, char *h_updating_graph_mask, char *h_graph_visited, int *h_cost)
{
    char h_over = false;
    cl_mem d_graph_nodes, d_graph_edges, d_graph_mask, d_updating_graph_mask, d_graph_visited, d_cost, d_over;

#ifdef PROFILING
    timer full_timer = timer("full");
    timer alloc_timer = timer("alloc");
    timer h2d_timer = timer("h2d");
    timer d2h_timer = timer("d2h");
    timer kernel_timer = timer("kernel");
    full_timer.reset();
    alloc_timer.reset();
    h2d_timer.reset();
    d2h_timer.reset();
    kernel_timer.reset();
        
    full_timer.start();
#endif

    try
    {
        //--1 transfer data from host to device
        _clInit();

#ifdef PROFILING
        alloc_timer.start();
#endif
        d_graph_nodes = _clMalloc(no_of_nodes * sizeof(Node), h_graph_nodes);
        d_graph_edges = _clMalloc(edge_list_size * sizeof(int), h_graph_edges);
        d_graph_mask = _clMallocRW(no_of_nodes * sizeof(char), h_graph_mask);
        d_updating_graph_mask = _clMallocRW(no_of_nodes * sizeof(char), h_updating_graph_mask);
        d_graph_visited = _clMallocRW(no_of_nodes * sizeof(char), h_graph_visited);

        d_cost = _clMallocRW(no_of_nodes * sizeof(int), h_cost);
        d_over = _clMallocRW(sizeof(char), &h_over);

#ifdef PROFILING
        alloc_timer.stop();
        h2d_timer.start();
#endif
        _clMemcpyH2D(d_graph_nodes, no_of_nodes * sizeof(Node), h_graph_nodes);
        _clMemcpyH2D(d_graph_edges, edge_list_size * sizeof(int), h_graph_edges);
        _clMemcpyH2D(d_graph_mask, no_of_nodes * sizeof(char), h_graph_mask);
        _clMemcpyH2D(d_updating_graph_mask, no_of_nodes * sizeof(char), h_updating_graph_mask);
        _clMemcpyH2D(d_graph_visited, no_of_nodes * sizeof(char), h_graph_visited);
        _clMemcpyH2D(d_cost, no_of_nodes * sizeof(int), h_cost);

#ifdef PROFILING
        h2d_timer.stop();
#endif

        //--2 invoke kernel

        int amtloops = 0;
        do
        {
            amtloops++;

            h_over = false; 
#ifdef PROFILING
            h2d_timer.start();
#endif
            _clMemcpyH2D(d_over, sizeof(char), &h_over);

#ifdef PROFILING
            h2d_timer.stop();
            kernel_timer.start();
#endif

            //--kernel 0
            int kernel_id = 0;
            int kernel_idx = 0;
            _clSetArgs(kernel_id, kernel_idx++, d_graph_nodes);
            _clSetArgs(kernel_id, kernel_idx++, d_graph_edges);
            _clSetArgs(kernel_id, kernel_idx++, d_graph_mask);
            _clSetArgs(kernel_id, kernel_idx++, d_updating_graph_mask);
            _clSetArgs(kernel_id, kernel_idx++, d_graph_visited);
            _clSetArgs(kernel_id, kernel_idx++, d_cost);
            _clSetArgs(kernel_id, kernel_idx++, &no_of_nodes, sizeof(int));

            //int work_items = no_of_nodes;
            _clInvokeKernel(kernel_id, no_of_nodes, work_group_size);

            //--kernel 1
            kernel_id = 1;
            kernel_idx = 0;
            _clSetArgs(kernel_id, kernel_idx++, d_graph_mask);
            _clSetArgs(kernel_id, kernel_idx++, d_updating_graph_mask);
            _clSetArgs(kernel_id, kernel_idx++, d_graph_visited);
            _clSetArgs(kernel_id, kernel_idx++, d_over);
            _clSetArgs(kernel_id, kernel_idx++, &no_of_nodes, sizeof(int));

            //work_items = no_of_nodes;
            _clInvokeKernel(kernel_id, no_of_nodes, work_group_size);
#ifdef PROFILING
            //Force awiting for kernel to finish.
            _clFinish();

            kernel_timer.stop();
            d2h_timer.start();
#endif
            _clMemcpyD2H(d_over, sizeof(char), &h_over);

#ifdef PROFILING
            d2h_timer.stop();
#endif
        } while (h_over);


        printf("Took %d loops\n", amtloops);
        _clFinish();

#ifdef PROFILING
        d2h_timer.start();
#endif

        //--3 transfer data from device to host
        _clMemcpyD2H(d_cost, no_of_nodes * sizeof(int), h_cost);

#ifdef PROFILING
        d2h_timer.stop();
#endif
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
    full_timer.stop();

    //--statistics
    std::cout << full_timer;
    std::cout << alloc_timer;
    std::cout << d2h_timer;
    std::cout << h2d_timer;
    std::cout << kernel_timer;
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

        printf("Amt nodes: %d\n", no_of_nodes);
        std::set<int>* construction_set = new std::set<int>[no_of_nodes];

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

            h_graph_mask[i] = false;
            h_updating_graph_mask[i] = false;
            h_graph_visited[i] = false;   
        }

        int source = 0;

        // Allocate mem for the result on host side
        int *h_cost = (int *)malloc(sizeof(int) * no_of_nodes);
        int *h_cost_ref = (int *)malloc(sizeof(int) * no_of_nodes);
        for (int i = 0; i < no_of_nodes; i++)
        {
            h_cost[i] = -1;
            h_cost_ref[i] = -1;
        }
        h_cost[source] = 0;
        h_cost_ref[source] = 0;

        printf("Running opencl...\n");

        //---------------------------------------------------------
        //--opencl entry
        h_graph_mask[source] = true;
        h_graph_visited[source] = true;
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

        // Set the source node as true in the mask
        h_graph_mask[source] = true;
        h_graph_visited[source] = true;
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
