# cpu parts build anywhere with a c++17 compiler; the cuda parts need nvcc.
CXX ?= c++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Iinclude
NVCC ?= nvcc
NVFLAGS ?= -O3 -std=c++17 -Iinclude -arch=sm_75 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_75,code=sm_75

build/test_gradients: tests/test_gradients.cpp src/lstm_cpu.cpp include/lstm.h | build
	$(CXX) $(CXXFLAGS) tests/test_gradients.cpp src/lstm_cpu.cpp -o $@

build/liblstm_cpu.so: src/c_api.cpp src/lstm_cpu.cpp include/lstm.h | build
	$(CXX) $(CXXFLAGS) -fPIC -shared src/c_api.cpp src/lstm_cpu.cpp -o $@

build/liblstm_cuda.so: src/lstm_cuda.cu | build
	$(NVCC) $(NVFLAGS) -Xcompiler -fPIC -shared src/lstm_cuda.cu -lcublas -o $@

cuda: build/liblstm_cuda.so

build:
	mkdir -p build

test: build/test_gradients build/liblstm_cpu.so
	./build/test_gradients

clean:
	rm -rf build

.PHONY: test clean cuda
