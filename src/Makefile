CC = g++

SRC = bfs.cpp timer.cc matrixmarket/mmio.c

CC_FLAGS = -g -O3

EXE = ../bin/bfs

release:$(SRC)
	$(CC) $(CC_FLAGS) $(SRC) -o $(EXE) -lOpenCL 

errmsg:$(SRC)
	$(CC) $(CC_FLAGS) $(SRC) -o $(EXE) -lOpenCL -D ERRMSG

ptx:$(SRC)
	$(CC) $(CC_FLAGS) $(SRC) -o $(EXE) -lOpenCL -D PTX_MSG

profile:$(SRC)
	$(CC) $(CC_FLAGS) $(SRC) -o $(EXE) -lOpenCL -D PROFILING

res:$(SRC)
	$(CC) $(CC_FLAGS) $(SRC) -o $(EXE) -lOpenCL -D RES_MSG

debug: $(SRC)
	$(CC) $(CC_FLAGS) $(SRC) -o $(EXE) -lOpenCL -D ERRMSG -D PTX_MSG -D RES_MSG -D PROFILING

run-gpu: debug
	./$(EXE) ../data/webbase-1M.mtx

run-cpu: debug
	./$(EXE) ../data/webbase-1M.mtx -c

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result*
