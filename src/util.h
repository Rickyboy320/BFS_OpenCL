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
        std::cout << "--cambine:passed:-)" << endl;
    }
    else{
        std::cout << "--cambine: failed:-(" << endl;
    }
    return ;
}
template<typename datatype>
void compare_results(const datatype *cpuResults, const datatype *clResults, const int size){

    int not_traversed = 0;
    char passed = true; 
    #pragma omp parallel for
    for (int i=0; i<size; i++){
      if (cpuResults[i] != clResults[i]) {
         printf("i: %d Diff: %d != %d\n", i, clResults[i], cpuResults[i]);
         passed = false; 
      }

      if (clResults[i] == -1) {
          not_traversed++;
      }
    }
    if (passed){
        std::cout << "--cambine:passed:-)" << endl;
    }
    else{
        std::cout << "--cambine: failed:-(" << endl;
    }

    if(not_traversed == 0) {
        std::cout << "--cambine:traversal completed :-)" << endl;
    } else {
        std::cout << "--cambine:traversed (" << size - not_traversed << " / " << size << ") :-(" << endl;
    }
    return ;
}

#endif

