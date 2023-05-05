#include <cuda.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include "errors.h"

#define MAX_DEG 1024
#define MAX_STACK MAX_DEG * MAX_DEG

#define BLOCK_SIZE 64
#define NUM_BLOCKS 16

std::vector<std::pair<int, int>> edges;
std::map<int, int> degree;

std::map<int, int> id_to_number;
std::map<int, int> number_to_id;


__global__ void kcliques(std::pair<int, int>* edges, std::pair<int, int>* intervals, int N, bool* intersect, int* stackVertex, int* stackDepth, int* cliques, int K, bool* inducedSubgraph) {
    // TODO: Zakodować binarnie i zwiększyć MAX_DEG
    //__shared__ bool inducedSubgraph[MAX_DEG][MAX_DEG];
    __shared__ int neighbours[MAX_DEG];

    __shared__ int stackTop;

    __shared__ int pref[BLOCK_SIZE];

    int part = N / gridDim.x;
    int rest = N % gridDim.x;
    // Maksymalna różnica liczby przetwarzanych wierzchołków między blokami wynosi 1
    // TODO: Być może lepiej, żeby ostatni blok robił mniej
    int firstVertexIncl = blockIdx.x * part + min(blockIdx.x, rest);
    int lastVertexExcl = (blockIdx.x + 1) * part + min(blockIdx.x + 1, rest);

    for (int v = firstVertexIncl; v < lastVertexExcl; ++v) { // Rozważamy graf indukowany zbiorem sąsiadów v
        int graphSize = intervals[v].second - intervals[v].first;
        int graphPart = graphSize / blockDim.x;
        int graphRest = graphSize % blockDim.x;

        // if (threadIdx.x == 0) {
        //     printf("v = %d graphSize = %d\n", v, graphSize);
        // }

        // Maksymalna różnica liczby przetwarzanych sąsiadów v między wątkami wynosi 1
        // TODO: Być może lepiej, żeby ostatni wątek robił mniej
        int firstNeighbourIncl = threadIdx.x * graphPart + min(threadIdx.x, graphRest);
        int lastNeighbourExcl = (threadIdx.x + 1) * graphPart + min(threadIdx.x + 1, graphRest);

        for (int i = firstNeighbourIncl; i < lastNeighbourExcl; ++i) {
            neighbours[i] = edges[i + intervals[v].first].second;
        }

        __syncthreads();

        for (int i = firstNeighbourIncl; i < lastNeighbourExcl; ++i) {
            int u = neighbours[i]; // Kopiujemy listę sąsiedztwa tego sąsiada
            // i = numer odpowiadający u w indukowanym podgrafie

            // Czyścimy pamięć shared
            for (int j = 0; j < graphSize; ++j) {
                inducedSubgraph[blockIdx.x * MAX_DEG * MAX_DEG + i * MAX_DEG + j] = false;
            }

            for (int j = intervals[u].first; j < intervals[u].second; j++) {
                int w = edges[j].second;
                // Sprawdzamy, czy w jest w indukowanym podgrafie
                int left = 0, right = graphSize - 1, mid;
                while (left < right) {
                    mid = (left + right) / 2;
                    if (neighbours[mid] < w) {
                        left = mid + 1;
                    }
                    else {
                        right = mid;
                    }
                }
                if (neighbours[left] == w) {
                    // left = numer odpowiadający w w indukowanym podgrafie
                    inducedSubgraph[blockIdx.x * MAX_DEG * MAX_DEG + i * MAX_DEG + left] = true;
                }
            }
        }

        for (int i = firstNeighbourIncl; i < lastNeighbourExcl; ++i) {
            // intersect[0][i] = true;
            intersect[blockIdx.x * MAX_STACK + 0 * MAX_DEG + i] = true;
            stackVertex[blockIdx.x * MAX_STACK + i] = i;
            stackDepth[blockIdx.x * MAX_STACK + i] = 1;
        }

        if (threadIdx.x == blockDim.x - 1) {
            stackTop = graphSize - 1;
            cliques[0 * NUM_BLOCKS + blockIdx.x]++;
            cliques[1 * NUM_BLOCKS + blockIdx.x] += graphSize; // Odpowiada wszystkim tym wrzuconym na stos wierzchołkom
        }

        __syncthreads();

        // if (threadIdx.x == 0 && blockIdx.x == 0) {
        //     printf("subgaph induced by %d\n", v);
        //     for (int i = 0; i < graphSize; i++) {
        //         for (int j = 0; j < graphSize; j++) {
        //             printf("%d ", inducedSubgraph[i][j]);
        //         }
        //         printf("\n");
        //     }
        // }

        while(true) {
            // if (threadIdx.x == 0 && blockIdx.x == 0) {
            //     printf("stackTop %d\n", stackTop);
            //     for (int i = 0; i <= stackTop; i++) {
            //         printf("%d (%d)\n", stackVertex[blockIdx.x * MAX_STACK + i], stackDepth[blockIdx.x * MAX_STACK + i]);
            //     }
            // }
            if (stackTop < 0) {
                break;
            }

            int u = stackVertex[blockIdx.x * MAX_STACK + stackTop];
            int depth = stackDepth[blockIdx.x * MAX_STACK + stackTop];
            // if (threadIdx.x == 0 && blockIdx.x == 0) {
            //     printf("vertex %d of depth %d\n", u, depth);
            // }

            int children = 0;
            for (int i = firstNeighbourIncl; i < lastNeighbourExcl; ++i) {
                intersect[blockIdx.x * MAX_STACK + depth * MAX_DEG + i] = intersect[blockIdx.x * MAX_STACK + (depth - 1) * MAX_DEG + i]  && inducedSubgraph[blockIdx.x * MAX_DEG * MAX_DEG + u * MAX_DEG + i];
                children += (int)(intersect[blockIdx.x * MAX_STACK + depth * MAX_DEG + i]);
            }

            pref[threadIdx.x] = children;
            __syncthreads();

            for (int i = 1; i < BLOCK_SIZE; i *= 2) {
                pref[threadIdx.x] += (threadIdx.x >= i ? pref[threadIdx.x - i] : 0);
                __syncthreads();
            }

            if (depth + 1 < K - 1) {
                int pos = stackTop + pref[threadIdx.x] - children;
                for (int i = firstNeighbourIncl; i < lastNeighbourExcl; ++i) {
                    if (intersect[blockIdx.x * MAX_STACK + depth * MAX_DEG + i]) {
                        stackVertex[blockIdx.x * MAX_STACK + pos] = i;
                        stackDepth[blockIdx.x * MAX_STACK + pos] = depth + 1;
                        pos++;
                    }
                }
            }

            // teraz pos = stackTop + pref[threadId.x]
            __syncthreads();

            if (threadIdx.x == blockDim.x - 1) {
                cliques[(depth + 1) * NUM_BLOCKS + blockIdx.x] += pref[threadIdx.x];
                stackTop = (depth + 1 < K - 1 ? stackTop + pref[threadIdx.x] - 1 : stackTop - 1);
            }

            __syncthreads();
        }

    }
}

int main(int argc, char* argv[]) {

    if (argc != 4) {
        std::cerr << "Usage: ./kcliques <graph input file> <k value> <output file>\n";
        return 0;
    }

    int K;
    try {
        K = std::stoi(argv[2]);
    }
    catch(std::exception) {
        std::cerr << "Usage: ./kcliques <graph input file> <k value> <output file>\n";
        return 0;
    }

    std::ifstream input (argv[1]);

    if (input.is_open()) {

        std::string line;
        size_t idx;

        while ( getline (input,line) ) {
            try {
                int a = std::stoi(line, &idx);
                int b = std::stoi(&(line[idx]));
                if (a != b) { // Ignorujemy pętle
                    edges.push_back({a, b});
                    degree[a]++;
                    degree[b]++;
                }
            }
            catch(std::exception) {
                std::cerr << "Error: invalid input data format\n";
                input.close();
                return 0;
            }
        }
        input.close();
    }
    else {
        std::cerr << "Error: unable to open input file\n";
        return 0;
    }

    int N = degree.size(); // Liczba wierzchołków

    // Przenumerowanie id na liczby z przedziału od 0 do N
    int num = 0;
    for (std::map<int, int>::iterator it = degree.begin(); it != degree.end(); ++it) {
        id_to_number[it->first] = num;
        number_to_id[num] = it->first;
        num++;
    }

    // Skierowanie krawędzi
    // TODO: zrobić to na GPU
    for (int i = 0; i < edges.size(); i++) {
        int a = edges[i].first;
        int b = edges[i].second;
        if (degree[b] > degree[a] || (degree[b] == degree[a] && id_to_number[b] > id_to_number[a])) {
            // graph[id_to_number[a]].push_back(id_to_number[b]);
            edges[i].first = id_to_number[a];
            edges[i].second = id_to_number[b];
        }
        else {
            // graph[id_to_number[b]].push_back(id_to_number[a]);
            edges[i].first = id_to_number[b];
            edges[i].second = id_to_number[a];
        }
    }

    // Sortowanie krawędzi wychodzących po numerze sąsiada
    // TODO: zrobić to na GPU
    sort(edges.begin(), edges.end());

    // Sprawdzenie poprawności wejścia
    for (int i = 1; i < edges.size(); ++i) {
        if (edges[i - 1] == edges[i]) {
            std::cerr << "Error: each edge should appear at most once in the list\n";
            return 0;
        }
    }    

    // Wyznaczenie przedziału krawędzi wychodzących dla każdego wierzchołka
    std::vector<std::pair<int, int>> intervals(N, {0, 0});
    int l = 0, r = 1;
    while (l < edges.size()) {
        while (r < edges.size() && edges[r].first == edges[l].first) {
            r++;
        }
        intervals[edges[l].first] = {l, r};
        l = r;
        r = l + 1;
    }

    // for (int i = 0; i < edges.size(); i++) {
    //     std::cout << edges[i].first << " " << edges[i].second << "\n";
    // }
    // for (int i = 0; i < N; i++) {
    //     std::cout << i << " {"<<intervals[i].first << ", " << intervals[i].second << "}\n";
    // }

    int cliques[NUM_BLOCKS * K];

    cudaEvent_t start, stop;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, 0));

    std::pair<int, int>* devEdges;
    std::pair<int, int>* devIntervals;
    HANDLE_ERROR(cudaMalloc((void**)&devEdges, sizeof(std::pair<int, int>) * edges.size()));
    HANDLE_ERROR(cudaMalloc((void**)&devIntervals, sizeof(std::pair<int, int>) * intervals.size()));

    HANDLE_ERROR(cudaMemcpy(devEdges, &edges.front(), sizeof(std::pair<int, int>) * edges.size(), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(devIntervals, &intervals.front(), sizeof(std::pair<int, int>) * intervals.size(), cudaMemcpyHostToDevice));

    bool* devIntersect;
    int* devStackVertex;
    int* devStackDepth;
    HANDLE_ERROR(cudaMalloc((void**)&devIntersect, sizeof(bool) * MAX_DEG * MAX_DEG * NUM_BLOCKS));
    HANDLE_ERROR(cudaMalloc((void**)&devStackVertex, sizeof(int) * MAX_STACK * NUM_BLOCKS));
    HANDLE_ERROR(cudaMalloc((void**)&devStackDepth, sizeof(int) * MAX_STACK * NUM_BLOCKS));

    int* devCliques;
    HANDLE_ERROR(cudaMalloc((void**)&devCliques, sizeof(int) * NUM_BLOCKS * K));
    HANDLE_ERROR(cudaMemset(devCliques, 0, sizeof(int) * NUM_BLOCKS * K));
    // HANDLE_ERROR(cudaMemcpy(devCliques, cliques, sizeof(int) * NUM_BLOCKS * K, cudaMemcpyHostToDevice));

    bool* devInducedSubrgaph;
    HANDLE_ERROR(cudaMalloc((void**)&devInducedSubrgaph, sizeof(bool) * NUM_BLOCKS * MAX_DEG * MAX_DEG));

    kcliques<<<NUM_BLOCKS, BLOCK_SIZE>>>(devEdges, devIntervals, N, devIntersect, devStackVertex, devStackDepth, devCliques, K, devInducedSubrgaph);

    HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));

	float elapsedTime;
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
	printf("Total GPU execution time: %3.1f ms\n", elapsedTime);

	HANDLE_ERROR(cudaEventDestroy(start));
	HANDLE_ERROR(cudaEventDestroy(stop));

    HANDLE_ERROR(cudaMemcpy(cliques, devCliques, sizeof(int) * NUM_BLOCKS * K, cudaMemcpyDeviceToHost));

    cudaFree(devEdges);
    cudaFree(devIntervals);
    cudaFree(devIntersect);
    cudaFree(devStackVertex);
    cudaFree(devStackDepth);
    cudaFree(devCliques);

    std::ofstream output (argv[3]);
    if (output.is_open()) {
        for (int i = 0; i < K; i++) {
            int sum = 0;
            for (int j = 0; j < NUM_BLOCKS; j++) {
                sum += cliques[i * NUM_BLOCKS + j];
                // std::cout << cliques[i * NUM_BLOCKS + j] << " ";
            }
            output << sum;
            if (i < K - 1) output << " ";
        }
        output.close();
    }
    else {
        std::cerr << "Unable to open output file\n";
        return 0;
    }
}
