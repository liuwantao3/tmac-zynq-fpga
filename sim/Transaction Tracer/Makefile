CXX      = g++
CXXFLAGS = -std=c++17 -pthread -O2 -Wall -Wextra -I..

all: demo

demo: demo.cpp ../fpga_sim.hpp axitrace.hpp fpga_profiler.hpp
	$(CXX) $(CXXFLAGS) -o $@ demo.cpp

test: demo
	./demo

clean:
	rm -f demo

.PHONY: all test clean
