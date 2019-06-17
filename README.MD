# Breadth-First Search (BFS) in OpenCL
Various implementations of BFS in OpenCL. Results can be found in PDF format in the charts/ directory.  
Naming conventions for the results are as follows:  
'analysis22_xxxx.pdf' is the breakdown of the iterations for the unmodified and direction-optimizing implementations on the graph500-22 graph (where xxxx is the platform).  
'appeff_xxxx.pdf' is the Application Efficiency for all implementations and platforms processing a specific graph.  
'exec22_xxxx.pdf' is the comparison of the execution time of an implementation on graph500-22 and graph500-23.  
'exec_xxxx.pdf' is either: a) the execution times of all implementations on one graph, or b) the execution times of one implementation on all graphs.  
'kernel_sizes_xxxx.pdf' shows the influence of differing kernel sizes on the performance of a specific graph.  
'load_balance.pdf' shows the unhalted runtime of all cores when running graph500-10 50000 times.  
'perfport.pdf' shows the performance portability of all implementations on all graphs.  
'platform_xxxx.pdf' shows the 'Programming Model Efficiency' of CUDA vs OpenCL for a specific implementation (and its respective port).  
'sizes_xxxx.pdf' shows the influence of differing work-group sizes on the performance of a specific graph.  
'zero_xxxx.pdf' shows a breakdown of H2D, kernel and D2H times for the GPU and the CPU for a specific graph.  

Source code for each implementation is on its own branch:
'Unmodified' is the master branch;  
'Datastructure' is the datastruct branch;  
'Direction-optimizing' is the direction-optimized branch;  
'Datastructure + Direction-optimizing' is the better_direction branch;  

Additionally, the CUDA ports are in the 'cuda_master' and 'cuda_better_direction' branches.

The 'zero-copy' and 'zero-copy-2' branches are for the zero-copy specialization. The former breaks portability with the GPU, but the latter recovers this.
Finally, the 'kernel-items' branch allows for differing the kernel workload.
