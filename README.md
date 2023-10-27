# CUDA clique counting
High Performance Computing task, University of Warsaw 2022/23

### Introduction

A _clique_ is a subset of vertices of an undirected graph such that every two distinct vertices in the clique are adjacent.

Clique related problems require lot of computational power in most variants due to NP-hardness. They are also well suited for parallelisation, which makes them perfect candidates for massive parallel GPU environment. In this task you will face one of those problems.

In the _k-clique problem_ you are given an undirected graph _G_ and a number _k_. The output is a list of numbers of cliques of size _i_ in _G_ for _i_ ∈ \[1, _k_\]. This problem is _NP-hard_ for _k_ being a variable, but is in _P_ when _k_ is a constant.

### Task

The goal of this task is to implement _k-clique problem_ solution in CUDA. We may assume that _k_ < 12.

### Used resources

*   Parallel K-clique Counting on GPUs. In 2022 International Conference on Supercomputing (ICS ’22), Mohammad Almasri, Izzat El Hajj, Rakesh Nagi, Jinjun Xiong, and Wen- mei Hwu. 2022. Available at: [https://arxiv.org/pdf/2104.13209.pdf](https://arxiv.org/pdf/2104.13209.pdf)

*   Video tutorial for the article above:

    [https://www.youtube.com/watch?v=l-8s7Aiotvo](https://www.youtube.com/watch?v=l-8s7Aiotvo)

### Compilation, input and output

The program should be compiled using the following instructions performed in the main catalogue:

`make clean; make`

The program should be then launched as follows:

`./kcliques <graph input file> <k value> <output file>`

#### Input

Graph input file consists of arbitrary number of lines describing graph _G_ in a form of edge list. Each line contains two space separated numbers, _a_ and _b_, 0 ≤ _a_, _b_ < 232. There is an undirected edge between _a_ and _b_ in the graph.

Value of _k_ is provided as a command line argument.

##### Additional constraints

Each edge should appear at most once in the list.

Let _G_′ be a directed graph constructed from _G_. _G_′ has the same edges as _G_, directed from the node of lower degree to the one with higher degree. We guarantee that the maximum out-degree of a node in _G_′ does not exceed 1024. This note may be helpful during optimisations.

#### Output

The program should write _k_ numbers in the specified output file. _i_\-th number describes count of cliques of size _i_ in the input graph, modulo 109.

#### Examples

When called with `./kcliques graph_input 4 output`

and the `graph_input` file content is:

    0 1
    0 2
    0 3
    1 2
    3 4
    3 5
    4 5
    2 3
    1 3
    2 4

![image](graph.png)

Then the program should put the following in the `output` file:

    6 10 6 1

#### Correctness

Catalogues `my_tests` and `correctness_fixed` contain example input and output files. The program can be tested with `./script.sh <test_dir> <k>`.

#### Report

`Report.md` contains brief implementation description, used techniques and optimisations (in Polish).
