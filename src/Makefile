CC = g++

SRC = bfs.cpp matrixmarket/mmio.c

CC_FLAGS = -O3

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
	$(CC) $(CC_FLAGS) -g $(SRC) -o $(EXE) -lOpenCL -D ERRMSG -D PTX_MSG -D RES_MSG -D PROFILING -D VERBOSE

results: $(SRC)
	$(CC) $(CC_FLAGS) $(SRC) -o $(EXE) -lOpenCL -D NO_CHECK -D PROFILING

run-gpu: debug
	./$(EXE) ../data/road_usa.mtx

run-cpu: debug
	./$(EXE) ../data/road_usa.mtx -c

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result*
