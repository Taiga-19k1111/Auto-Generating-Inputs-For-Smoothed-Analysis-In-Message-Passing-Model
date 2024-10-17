CXX = g++
CXXFLAGS = -std=gnu++11 -O2
.PHONY: all clean

all: solvers/clique solvers/coloring solvers/vertex_cover solvers/vertex_cover_approx

solvers/clique: solvers/src/clique.cpp
	$(CXX) $(CXXFLAGS) solvers/src/clique.cpp -o solvers/clique

solvers/coloring: solvers/src/coloring.cpp
	$(CXX) $(CXXFLAGS) solvers/src/coloring.cpp -o solvers/coloring

solvers/vertex_cover: solvers/src/vertex_cover.cpp
	$(CXX) $(CXXFLAGS) solvers/src/vertex_cover.cpp -o solvers/vertex_cover

solvers/vertex_cover_approx: solvers/src/vertex_cover_approx.cpp
	$(CXX) $(CXXFLAGS) solvers/src/vertex_cover_approx.cpp -o solvers/vertex_cover_approx

solvers/quicksort_r: solvers/src/quicksort.cpp
	$(CXX) $(CXXFLAGS) solvers/src/quicksort.cpp -o solvers/quicksort

solvers/quicksort_l: solvers/src/quicksort_left.cpp
	$(CXX) $(CXXFLAGS) solvers/src/quicksort_left.cpp -o solvers/quicksort_left

solvers/quicksort_c: solvers/src/quicksort_center.cpp
	$(CXX) $(CXXFLAGS) solvers/src/quicksort_center.cpp -o solvers/quicksort_center

solvers/quicksort_rnd: solvers/src/quicksort_random.cpp
	$(CXX) $(CXXFLAGS) solvers/src/quicksort_random.cpp -o solvers/quicksort_random

solvers/quicksort_check: solvers/src/quicksort_check.cpp
	$(CXX) $(CXXFLAGS) solvers/src/quicksort_check.cpp -o solvers/quicksort_check

clean:
	rm -r solvers
	mkdir solvers
	git checkout solvers
