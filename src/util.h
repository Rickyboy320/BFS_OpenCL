#ifndef _C_UTIL_
#define _C_UTIL_
#include <math.h>
#include <iostream>
#include <omp.h>
//-------------------------------------------------------------------
//--initialize array with maximum limit
//-------------------------------------------------------------------
template<typename datatype>
void fill(datatype *A, const int n, const datatype maxi){
    for (int j = 0; j < n; j++) 
    {
        A[j] = ((datatype) maxi * (rand() / (RAND_MAX + 1.0f)));
    }
}

//--print matrix
template<typename datatype>
void print_matrix(datatype *A, int height, int width){
    for(int i=0; i<height; i++){
        for(int j=0; j<width; j++){
            int idx = i*width + j;
            std::cout<<A[idx]<<" ";
        }
        std::cout<<std::endl;
    }

    return;
}
//-------------------------------------------------------------------
//--verify results
//-------------------------------------------------------------------
#define MAX_RELATIVE_ERROR  .002
template<typename datatype>
void verify_array(const datatype *cpuResults, const datatype *clResults, const int size){

    char passed = true; 
#pragma omp parallel for
    for (int i=0; i<size; i++){
      if (fabs(cpuResults[i] - clResults[i]) / cpuResults[i] > MAX_RELATIVE_ERROR){
         passed = false; 
      }
    }
    if (passed){
        std::cout << "--cambine:passed:-)" << std::endl;
    }
    else{
        std::cout << "--cambine: failed:-(" << std::endl;
    }
    return ;
}
template<typename datatype>
void compare_results(const datatype *cpuResults, const datatype *clResults, const int size){

    int not_traversed = 0;
    char passed = true; 

    for (int i=0; i<size; i++){
      if (cpuResults[i]!=clResults[i]){
         printf("Diff@%d: %d != %d\n", i, clResults[i], cpuResults[i]);
         passed = false; 
      }

      if (clResults[i] == -1) {
          not_traversed++;
      }
    }
    if (passed){
        std::cout << "--cambine:passed:-)" << std::endl;
    }
    else{
        std::cout << "--cambine: failed:-(" << std::endl;
    }

    if(not_traversed == 0) {
        std::cout << "--cambine:traversal completed :-)" << std::endl;
    } else {
        std::cout << "--cambine:traversed (" << size - not_traversed << " / " << size << ") :-(" << std::endl;
    }
    return ;
}

void cmdParams(int argc, char *argv[], int* source, int* iterations, int* work_group_size, int* device_id_inuse, bool* cpu)
{
    for (int i = 2; i < argc; i++)
    {
        if(argv[i][0] != '-') {
            continue;
        }

        switch (argv[i][1])
        {
        case 'g': //--g stands for size of work group
            if (++i < argc)
            {
                sscanf(argv[i], "%d", work_group_size);
#ifdef VERBOSE
                printf("Setting work group size to %d\n", *work_group_size);
#endif
            }
            else
            {
                std::cerr << "Could not read argument after option " << argv[i - 1] << std::endl;
                throw;
            }
            break;
        case 'd': //--d stands for device id used in computaion
            if (++i < argc)
            {
                sscanf(argv[i], "%d", device_id_inuse);
#ifdef VERBOSE
                printf("Setting device id to %d\n", *device_id_inuse);
#endif
            }
            else
            {
                std::cerr << "Could not read argument after option " << argv[i - 1] << std::endl;
                throw;
            }
            break;
        case 'c':
            *cpu = true;
#ifdef VERBOSE
            printf("Attempting to use CPU instead of GPU.\n");
#endif
            break;
        case 's':
             if (++i < argc)
            {
                sscanf(argv[i], "%d", source);
#ifdef VERBOSE
                printf("Setting source to %d\n", *source);
#endif
            }
            else
            {
                std::cerr << "Could not read argument after option " << argv[i - 1] << std::endl;
                throw;
            }
            break;
        case 'i':
             if (++i < argc)
            {
                sscanf(argv[i], "%d", iterations);
#ifdef VERBOSE
                printf("Setting iterations to %d\n", *iterations);
#endif
            }
            else
            {
                std::cerr << "Could not read argument after option " << argv[i - 1] << std::endl;
                throw;
            }
            break;
        default:;
        }
    }
}

#endif

