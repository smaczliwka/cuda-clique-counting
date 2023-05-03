#include <cuda.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include "errors.h"

#define MAX_DEG 512
#define MAX_STACK MAX_DEG * MAX_DEG

#define BLOCK_SIZE 16
#define NUM_BLOCKS 32

std::vector<std::pair<int, int>> edges;
std::map<int, int> degree;

std::map<int, int> id_to_number;
std::map<int, int> number_to_id;


__global__ void kcliques(std::pair<int, int>* edges, std::pair<int, int>* intervals, int N, unsigned int* intersect, int* stackVertex, int* stackDepth, int* cliques, int K) {
    // TODO: Zakodować binarnie i zwiększyć MAX_DEG
    __shared__ unsigned int inducedSubgraph[MAX_DEG * MAX_DEG / 32];
    // __shared__ int neighbours[MAX_DEG];

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

        // Maksymalna różnica liczby przetwarzanych sąsiadów v między wątkami wynosi 1
        // TODO: Być może lepiej, żeby ostatni wątek robił mniej
        int firstNeighbourIncl = threadIdx.x * graphPart + min(threadIdx.x, graphRest);
        int lastNeighbourExcl = (threadIdx.x + 1) * graphPart + min(threadIdx.x + 1, graphRest);

        int codeSize = (graphSize + 31) / 32; // Na tylu liczbach kodujemy wiersz macierzy sąsiedztwa;
        int codePart = codeSize / blockDim.x;
        int codeRest = codeSize % blockDim.x;

        int firstIntersectionIncl = threadIdx.x * codePart + min(threadIdx.x, codeRest);
        int lastIntersectionExcl = (threadIdx.x + 1) * codePart + min(threadIdx.x + 1, codeRest);

        // if (threadIdx.x == 0) {
        //     printf("v = %d graphSize = %d\n", v, graphSize);
        // }

        // for (int i = firstNeighbourIncl; i < lastNeighbourExcl; ++i) {
        //     neighbours[i] = edges[i + intervals[v].first].second;
        // }

        __syncthreads();

        for (int i = firstNeighbourIncl; i < lastNeighbourExcl; ++i) {
            int u = edges[i + intervals[v].first].second; // Kopiujemy listę sąsiedztwa tego sąsiada
            // i = numer odpowiadający u w indukowanym podgrafie

            // Czyścimy wiersz odpowiadający u w inducedSubgraph
            for (int j = 0; j < codeSize; ++j) {
                inducedSubgraph[i * (MAX_DEG / 32) + j] = 0;
            }

            for (int j = intervals[u].first; j < intervals[u].second; j++) {
                int w = edges[j].second;
                // Sprawdzamy, czy w jest w indukowanym podgrafie
                int left = 0, right = graphSize - 1, mid;
                while (left < right) {
                    mid = (left + right) / 2;
                    if (edges[mid + intervals[v].first].second < w) {
                        left = mid + 1;
                    }
                    else {
                        right = mid;
                    }
                }
                if (edges[left + intervals[v].first].second == w) {
                    // left = numer odpowiadający w w indukowanym podgrafie
                    int number = left / 32;
                    int bit = left % 32;
                    inducedSubgraph[i * (MAX_DEG / 32) + number] |= (1 << bit);
                }
            }
        }

        for (int i = firstNeighbourIncl; i < lastNeighbourExcl; ++i) {
            stackVertex[blockIdx.x * MAX_STACK + i] = i;
            stackDepth[blockIdx.x * MAX_STACK + i] = 1;
        }

        // Więcej jednynek niż graphSize. Czy to nie przeszkadza?
        for (int i = firstIntersectionIncl; i < lastIntersectionExcl; ++i) {
            intersect[blockIdx.x * MAX_DEG * (MAX_DEG / 32) + 0 * (MAX_DEG / 32) + i] = ~0;
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
        //         for (int number = 0; number < codeSize; number++) {
        //             for (int bit = 0; bit < 32 && number * 32 + bit < graphSize; bit++) {
        //                 printf("%d ", (inducedSubgraph[i][number] >> bit) & 1);
        //             }
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
            for (int i = firstIntersectionIncl; i < lastIntersectionExcl; ++i) {
                intersect[blockIdx.x * MAX_DEG * (MAX_DEG / 32) + depth * (MAX_DEG / 32) + i] = intersect[blockIdx.x * MAX_DEG * (MAX_DEG / 32) + (depth - 1) * (MAX_DEG / 32) + i] & inducedSubgraph[u * (MAX_DEG / 32) + i];
                for (int bit = 0; bit < 32; bit++) {
                    children += ((intersect[blockIdx.x * MAX_DEG * (MAX_DEG / 32) + depth * (MAX_DEG / 32) + i] >> bit) & 1);
                }
            }

            pref[threadIdx.x] = children;
            __syncthreads();

            for (int i = 1; i < BLOCK_SIZE; i *= 2) {
                pref[threadIdx.x] += (threadIdx.x >= i ? pref[threadIdx.x - i] : 0);
                __syncthreads();
            }

            if (depth + 1 < K - 1) {
                int pos = stackTop + pref[threadIdx.x] - children;
                for (int i = firstIntersectionIncl; i < lastIntersectionExcl; ++i) {
                    for (int bit = 0; bit < 32; bit++) {
                        if ((intersect[blockIdx.x * MAX_DEG * (MAX_DEG / 32) + depth * (MAX_DEG / 32) + i] >> bit) & 1 == 1) {
                            stackVertex[blockIdx.x * MAX_STACK + pos] = i * 32 + bit;
                            stackDepth[blockIdx.x * MAX_STACK + pos] = depth + 1;
                            pos++;
                        }
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

    cudaFuncSetCacheConfig(kcliques, cudaFuncCachePreferShared);

    if (MAX_DEG % 32 != 0) {
        std::cerr << "MAX_DEG must be a multiple of 32";
        return 1;
    }

    if (argc != 4) {
        std::cerr << "Usage: ./kcliques <graph input file> <k value> <output file>\n";
        return 1;
    }

    int K;
    try {
        K = std::stoi(argv[2]);
    }
    catch(std::exception) {
        std::cerr << "Usage: ./kcliques <graph input file> <k value> <output file>\n";
        return 1;
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
            return 1;
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

    std::pair<int, int>* devEdges;
    std::pair<int, int>* devIntervals;
    HANDLE_ERROR(cudaMalloc((void**)&devEdges, sizeof(std::pair<int, int>) * edges.size()));
    HANDLE_ERROR(cudaMalloc((void**)&devIntervals, sizeof(std::pair<int, int>) * intervals.size()));

    HANDLE_ERROR(cudaMemcpy(devEdges, &edges.front(), sizeof(std::pair<int, int>) * edges.size(), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(devIntervals, &intervals.front(), sizeof(std::pair<int, int>) * intervals.size(), cudaMemcpyHostToDevice));
  
    unsigned int* devIntersect;
    int* devStackVertex;
    int* devStackDepth;
    HANDLE_ERROR(cudaMalloc((void**)&devIntersect, sizeof(unsigned int) * NUM_BLOCKS * MAX_DEG * (MAX_DEG / 32)));
    HANDLE_ERROR(cudaMalloc((void**)&devStackVertex, sizeof(int) * MAX_STACK * NUM_BLOCKS));
    HANDLE_ERROR(cudaMalloc((void**)&devStackDepth, sizeof(int) * MAX_STACK * NUM_BLOCKS));

    int cliques[NUM_BLOCKS * K];
    for (int i = 0; i < NUM_BLOCKS * K; i++) {
        cliques[i] = 0;
    }

    int* devCliques;
    HANDLE_ERROR(cudaMalloc((void**)&devCliques, sizeof(int) * NUM_BLOCKS * K));
    HANDLE_ERROR(cudaMemcpy(devCliques, cliques, sizeof(int) * NUM_BLOCKS * K, cudaMemcpyHostToDevice));

    kcliques<<<NUM_BLOCKS, BLOCK_SIZE>>>(devEdges, devIntervals, N, devIntersect, devStackVertex, devStackDepth, devCliques, K);

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
            output << sum << " ";
        }
        output.close();
    }
    else {
        std::cerr << "Unable to open output file\n";
        return 0;
    }
}