# cpu parts build anywhere with a c++17 compiler; the cuda parts need nvcc.
CXX ?= c++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Iinclude
NVCC ?= nvcc
NVFLAGS ?= -O3 -std=c++17 -Iinclude -arch=sm_75 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_75,code=sm_75

build/test_gradients: tests/test_gradients.cpp src/lstm_cpu.cpp include/lstm.h | build
	$(CXX) $(CXXFLAGS) tests/test_gradients.cpp src/lstm_cpu.cpp -o $@

build:
	mkdir -p build

test: build/test_gradients
	./build/test_gradients

clean:
	rm -rf build

.PHONY: test clean
