CC = g++

SRC = bfs.cpp timer.cc

CC_FLAGS = -g -O3

EXE = bfs

release:$(SRC)
	$(CC) $(CC_FLAGS) $(SRC) -o $(EXE) -lOpenCL 

errmsg:$(SRC)
	$(CC) $(CC_FLAGS) $(SRC) -o $(EXE) -lOpenCL -D ERRMSG PTX_MSG

ptx:$(SRC)
	$(CC) $(CC_FLAGS) $(SRC) -o $(EXE) -lOpenCL -D PTX_MSG

profile:$(SRC)
	$(CC) $(CC_FLAGS) $(SRC) -o $(EXE) -lOpenCL -D PROFILING

res:$(SRC)
	$(CC) $(CC_FLAGS) $(SRC) -o $(EXE) -lOpenCL -D RES_MSG

debug: $(SRC)
	$(CC) $(CC_FLAGS) $(SRC) -o $(EXE) -lOpenCL 

run:
	./$(EXE)

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result*
