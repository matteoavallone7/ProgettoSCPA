# ===== Compilatori =====
CC = mpicc
NVCC  = nvcc
CUDA_ROOT ?= /opt/cuda/12.6

# ===== Flag =====
CFLAGS  = -O3 -Wall -std=c11 -I$(CUDA_DIR) -I$(CUDA_ROOT)/include
NVFLAGS = -O3 -arch=sm_75 -Xcompiler -fPIC -I$(CUDA_DIR)

LDFLAGS = -L$(CUDA_ROOT)/lib64 -lm -lcudart -lcublas

# ===== Directory =====
CUDA_DIR = cuda

# ===== Target =====
TARGET = matmul_mpi_cuda

# ===== File sorgenti C =====
SRC_C = \
	main.c \
	process_grid.c \
	matrixgen.c \
	distribution.c \
	matmul_serial.c

# ===== File CUDA =====
SRC_CU = $(CUDA_DIR)/matmul_cuda.cu

# ===== Oggetti =====
OBJ_C  = $(SRC_C:.c=.o)
OBJ_CU = $(SRC_CU:.cu=.o)

OBJS = $(OBJ_C) $(OBJ_CU)

# ===== Header =====
HEADERS = \
	matrixgen.h \
	matmul_serial.h \
	utils.h \
	$(CUDA_DIR)/matmul_cuda.h

# ===== Build =====
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Build completata: $(TARGET)"

# ===== Compilazione C =====
%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

# ===== Compilazione CUDA =====
$(CUDA_DIR)/%.o: $(CUDA_DIR)/%.cu $(HEADERS)
	$(NVCC) $(NVFLAGS) -c $< -o $@

# ===== Test rapido =====
test: $(TARGET)
	mpirun -np 4 ./$(TARGET) -M 8192 -N 8192 -k 96 -P 2 -Q 2 -verify -n_iter 10
	mpirun -np 4 ./$(TARGET) -M 8000 -N 8000 -k 64 -P 2 -Q 2 -verify -n_iter 10
	mpirun -np 4 ./$(TARGET) -M 4000 -N 8000 -k 32 -P 2 -Q 2 -verify -n_iter 10
	mpirun -np 4 ./$(TARGET) -M 4000 -N 8000 -k 16 -P 2 -Q 2 -verify -n_iter 10

# ===== Benchmark completo =====
benchmark: $(TARGET)
	@for k in 3 6 8 16 20 32 64 96 100 128; do \
		echo ""; \
		echo "===== k=$$k | M=N ====="; \
		mpirun -np 4 ./$(TARGET) -M 8000 -N 8000 -k $$k -P 2 -Q 2 -verify -n_iter 10; \
		echo ""; \
		echo "===== k=$$k | M=3N ====="; \
		mpirun -np 4 ./$(TARGET) -M 12000 -N 4000 -k $$k -P 2 -Q 2 -verify -n_iter 10; \
		echo ""; \
		echo "===== k=$$k | N=2M ====="; \
		mpirun -np 4 ./$(TARGET) -M 4000 -N 8000 -k $$k -P 2 -Q 2 -verify -n_iter 10; \
		echo ""; \
		echo "===== k=$$k | M=N (large) ====="; \
		mpirun -np 4 ./$(TARGET) -M 12000 -N 12000 -k $$k -P 2 -Q 2 -verify -n_iter 10; \
		echo ""; \
		echo "===== k=$$k | M=N (power of 2) ====="; \
		mpirun -np 4 ./$(TARGET) -M 8192 -N 8192 -k $$k -P 2 -Q 2 -verify -n_iter 10; \
		echo ""; \
		echo "===== k=$$k | M=3N (power of 2) ====="; \
		mpirun -np 4 ./$(TARGET) -M 24576 -N 8192 -k $$k -P 2 -Q 2 -verify -n_iter 10; \
		echo ""; \
		echo "===== k=$$k | M=N (small) ====="; \
		mpirun -np 4 ./$(TARGET) -M 1024 -N 1024 -k $$k -P 2 -Q 2 -verify -n_iter 10; \
		echo ""; \
	done

# ===== Pulizia =====
clean:
	rm -f *.o
	rm -f $(CUDA_DIR)/*.o
	rm -f $(TARGET)
	rm -f results/benchmark_results.csv

.PHONY: all test benchmark clean
