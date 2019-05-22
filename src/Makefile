SRC = matrixmarket/mmio.cpp bfs.cpp kernels.cu

EXE = ../bin/bfs

CUDA_PATH ?= /usr/local/cuda
HOST_COMPILER ?= g++
CC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

CC_FLAGS    = -O3

INCLUDES  := -I../inc -I./matrixmarket

profile: $(SRC)
	$(CC) $(INCLUDES) $(CC_FLAGS) $(SRC) -o $(EXE) -D PROFILING

debug: $(SRC)
	$(CC) $(INCLUDES) $(CC_FLAGS) -g $(SRC) -o $(EXE) -D ERRMSG -D PTX_MSG -D RES_MSG -D PROFILING -D VERBOSE

results: $(SRC)
	$(CC) $(INCLUDES) $(CC_FLAGS) $(SRC) -o $(EXE) -D NO_CHECK -D PROFILING

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result*
