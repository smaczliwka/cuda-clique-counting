#include <cuda.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include "errors.h"

#define MAX_DEG 1024
#define MAX_STACK MAX_DEG * MAX_DEG / 2
#define MAX_DEPTH K

#define BLOCK_SIZE 32
#define NUM_BLOCKS 128
#define GROUP_SIZE 8
#define GROUPS_PER_BLOCK (int)(BLOCK_SIZE / GROUP_SIZE)

#define MOD 1000000000

// #define FULL_MASK 0xffffffff

std::vector<std::pair<uint, uint>> edges;
std::map<uint, uint> degree;

std::map<uint, uint> id_to_number;
std::map<uint, uint> number_to_id;


__global__ void kcliques(std::pair<uint, uint>* edges, std::pair<int, int>* intervals, int N, unsigned int* intersect, uint* stackVertex, int* stackDepth, int* cliques, int K, unsigned int* inducedSubgraph) {
    // TODO: Zakodować binarnie i zwiększyć MAX_DEG
    // __shared__ unsigned int inducedSubgraph[MAX_DEG * MAX_DEG / 32];
    // __shared__ uint neighbours[MAX_DEG];

    uint mask = ((~(uint)0) >> (32 - GROUP_SIZE));
    int groupInWarp = (threadIdx.x % 32) / GROUP_SIZE;
    mask = mask << (groupInWarp * GROUP_SIZE);

    // if (threadIdx.x == GROUP_SIZE - 1)
    // for (int bit = 31; bit >= 0; bit--) {
    //     printf("%d", (mask >> bit) & 1);   
    // }

    __shared__ int maxStackTop[GROUPS_PER_BLOCK];

    int stackTop;
    int pref;

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

        // if (threadIdx.x == 0 && blockIdx.x == 0) {
        //     printf("v = %d graphSize = %d codeSize = %d\n", v, graphSize, codeSize);
        // }

        // for (int i = firstNeighbourIncl; i < lastNeighbourExcl; ++i) {
        //     neighbours[i] = edges[i + intervals[v].first].second;
        // }

        // __syncthreads();

        for (int i = firstNeighbourIncl; i < lastNeighbourExcl; ++i) {
            uint u = edges[i + intervals[v].first].second; // Kopiujemy listę sąsiedztwa tego sąsiada
            // i = numer odpowiadający u w indukowanym podgrafie

            // Czyścimy wiersz odpowiadający u w inducedSubgraph
            for (int j = 0; j < codeSize; ++j) {
                inducedSubgraph[blockIdx.x * MAX_DEG * (MAX_DEG / 32) + i * (MAX_DEG / 32) + j] = 0;
            }

            for (int j = intervals[u].first; j < intervals[u].second; j++) {
                uint w = edges[j].second;
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
                    inducedSubgraph[blockIdx.x * MAX_DEG * (MAX_DEG / 32) + i * (MAX_DEG / 32) + number] |= (1 << bit);
                }
            }
        }

        // Wykonujemy DFS

        for (int i = firstNeighbourIncl; i < lastNeighbourExcl; ++i) {
            // i-ty sąsiad trafia na stos grupy i % GROUPS_PER_BLOCK na pozycję i / GROUPS_PER_BLOCK
            stackVertex[blockIdx.x * GROUPS_PER_BLOCK * MAX_STACK + (i % GROUPS_PER_BLOCK) * MAX_STACK + (i / GROUPS_PER_BLOCK)] = i;
            stackDepth[blockIdx.x * GROUPS_PER_BLOCK * MAX_STACK + (i % GROUPS_PER_BLOCK) * MAX_STACK + (i / GROUPS_PER_BLOCK)] = 1;
        }

        int codePart = codeSize / GROUP_SIZE;
        int codeRest = codeSize % GROUP_SIZE;

        int threadInGroup = threadIdx.x % GROUP_SIZE; // Którym jestem wątkiem w swojej grupie.
        int groupId = threadIdx.x / GROUP_SIZE; // Numer mojej grupy.

        int firstIntersectionIncl = threadInGroup * codePart + min(threadInGroup, codeRest);
        int lastIntersectionExcl = (threadInGroup + 1) * codePart + min(threadInGroup + 1, codeRest);


        // Więcej jednynek niż graphSize. Czy to nie przeszkadza?
        for (int i = firstIntersectionIncl; i < lastIntersectionExcl; ++i) {
            intersect[blockIdx.x * GROUPS_PER_BLOCK * MAX_DEPTH * (MAX_DEG / 32) + groupId * MAX_DEPTH * (MAX_DEG / 32) + 0 * (MAX_DEG / 32) + i] = ~0;
        }

        stackTop = (graphSize / GROUPS_PER_BLOCK) + (graphSize % GROUPS_PER_BLOCK > groupId ? 1 : 0) - 1;
        // if (threadInGroup == 0) {
        //     // printf("graphSize %d, GROUPS PER BLOCK %d\n", graphSize, GROUPS_PER_BLOCK);
        //     // printf("graphSize / GROUPS_PER_BLOCK %d\n", (graphSize / GROUPS_PER_BLOCK));
        //     // printf("(graphSize mod GROUPS_PER_BLOCK) %d\n", (graphSize % GROUPS_PER_BLOCK));
        //     printf("groupId = %d, initial stackTop = %d\n", groupId, stackTop);
        // }

        if (threadIdx.x == 0) {
            cliques[0 * NUM_BLOCKS * GROUPS_PER_BLOCK + blockIdx.x * GROUPS_PER_BLOCK + groupId]++;
            cliques[0 * NUM_BLOCKS * GROUPS_PER_BLOCK + blockIdx.x * GROUPS_PER_BLOCK + groupId] %= MOD;            
        }
        if (threadInGroup == 0) {
            maxStackTop[groupId] = stackTop;
            cliques[1 * NUM_BLOCKS * GROUPS_PER_BLOCK + blockIdx.x * GROUPS_PER_BLOCK + groupId] += (graphSize / GROUPS_PER_BLOCK) + (graphSize % GROUPS_PER_BLOCK > groupId ? 1 : 0); // Odpowiada wszystkim tym wrzuconym na stos wierzchołkom
            cliques[1 * NUM_BLOCKS * GROUPS_PER_BLOCK + blockIdx.x * GROUPS_PER_BLOCK + groupId] %= MOD;
        }

        // __syncthreads();

        // if (threadIdx.x == 0) {
        //     for (int i = 0; i < GROUPS_PER_BLOCK; i++) {
        //         printf("%d ", maxStackTop[i]);
        //     }
        //     printf("\n");
        // }

        __syncthreads();

        for (int i = 1; i < GROUPS_PER_BLOCK; i *= 2) {
            if (threadInGroup == 0) {
                maxStackTop[groupId] = max(maxStackTop[(groupId + i) % GROUPS_PER_BLOCK], maxStackTop[groupId]);
            }
            __syncthreads();
        }

        // __syncthreads();

        // if (threadIdx.x == 0 && blockIdx.x == 0) {
        //     printf("subgaph induced by %d\n", v);
        //     for (int i = 0; i < graphSize; i++) {
        //         for (int number = 0; number < codeSize; number++) {
        //             for (int bit = 0; bit < 32 && number * 32 + bit < graphSize; bit++) {
        //                 printf("%d ", (inducedSubgraph[blockIdx.x * MAX_DEG * (MAX_DEG / 32) + i * (MAX_DEG / 32) + number] >> bit) & 1);
        //             }
        //         }
        //         printf("\n");
        //     }
        // }

        while(maxStackTop[groupId] >= 0) {

            // if (threadIdx.x == 0) {
            //     printf("after reduce\n");
            //     for (int i = 0; i < GROUPS_PER_BLOCK; i++) {
            //         printf("%d ", maxStackTop[i]);
            //     }
            //     printf("\n");
            // }

            // __syncthreads();
            // if (threadInGroup == 0) {
            //     printf("groupId = %d: stackTop %d\n", groupId, stackTop);
            //     for (int i = 0; i <= stackTop; i++) {
            //         printf("groupId = %d: %d (%d)\n", groupId, stackVertex[blockIdx.x * GROUPS_PER_BLOCK * MAX_STACK + groupId * MAX_STACK + i], stackDepth[blockIdx.x * GROUPS_PER_BLOCK * MAX_STACK + groupId * MAX_STACK + i]);
            //     }
            // }

            if (stackTop >= 0) {
                uint u = stackVertex[blockIdx.x * GROUPS_PER_BLOCK * MAX_STACK + groupId * MAX_STACK + stackTop];
                int depth = stackDepth[blockIdx.x * GROUPS_PER_BLOCK * MAX_STACK + groupId * MAX_STACK + stackTop];
                // if (threadInGroup == 0) {
                //     printf("groupId = %d: vertex %d of depth %d\n", groupId, u, depth);
                // }

                int children = 0;
                for (int i = firstIntersectionIncl; i < lastIntersectionExcl; ++i) {
                    intersect[blockIdx.x * GROUPS_PER_BLOCK * MAX_DEPTH * (MAX_DEG / 32) + groupId * MAX_DEPTH * (MAX_DEG / 32) + depth * (MAX_DEG / 32) + i] = intersect[blockIdx.x * GROUPS_PER_BLOCK * MAX_DEPTH * (MAX_DEG / 32) + groupId * MAX_DEPTH * (MAX_DEG / 32) + (depth - 1) * (MAX_DEG / 32) + i] & inducedSubgraph[blockIdx.x * MAX_DEG * (MAX_DEG / 32) + u * (MAX_DEG / 32) + i];
                    for (int bit = 0; bit < 32; bit++) {
                        children += ((intersect[blockIdx.x * GROUPS_PER_BLOCK * MAX_DEPTH * (MAX_DEG / 32) + groupId * MAX_DEPTH * (MAX_DEG / 32) + depth * (MAX_DEG / 32) + i] >> bit) & 1);
                    }
                }

                // if (threadIdx.x == 0) {
                //     for (int i = 0; i < codeSize; i++) {
                //         for (int bit = 0; bit < 32; bit++) {
                //             //printf("%d", (inducedSubgraph[blockIdx.x * MAX_DEG * (MAX_DEG / 32) + u * (MAX_DEG / 32) + i] >> bit) & 1);
                //             printf("%d", (intersect[blockIdx.x * GROUPS_PER_BLOCK * MAX_DEPTH * (MAX_DEG / 32) + groupId * MAX_DEPTH * (MAX_DEG / 32) + (depth) * (MAX_DEG / 32) + i] >> bit) & 1);
                //         }
                //     }
                //     printf("\n");
                // }

                pref = children;
                // printf("threadInGroup %d, pref before = %d\n", threadInGroup, pref);
                // for (int i = 1; i < GROUP_SIZE; i *= 2) {
                //     pref += __shfl_up_sync(FULL_MASK, pref, i);
                // }
                for (int i = 1; i < GROUP_SIZE; i *= 2) {
                    int tmp = __shfl_sync(mask, pref, threadInGroup - i, GROUP_SIZE);
                    pref += (threadInGroup >= i ? tmp : 0);
                }
                // printf("threadInGroup %d, pref = %d\n", threadInGroup, pref);

                // for (int i = 1; i < GROUP_SIZE; i *= 2) {
                //     pref[threadIdx.x] += (threadInGroup >= i ? pref[threadIdx.x - i] : 0);
                //     __syncthreads();
                // }

                
                if (depth + 1 < K - 1) {
                    int pos = stackTop + pref - children;
                    for (int i = firstIntersectionIncl; i < lastIntersectionExcl; ++i) {
                        for (int bit = 0; bit < 32; bit++) {
                            if ((intersect[blockIdx.x * GROUPS_PER_BLOCK * MAX_DEPTH * (MAX_DEG / 32) + groupId * MAX_DEPTH * (MAX_DEG / 32) + depth * (MAX_DEG / 32) + i] >> bit) & 1 == 1) {
                                stackVertex[blockIdx.x * GROUPS_PER_BLOCK * MAX_STACK + groupId * MAX_STACK + pos] = i * 32 + bit;
                                stackDepth[blockIdx.x * GROUPS_PER_BLOCK * MAX_STACK + groupId * MAX_STACK + pos] = depth + 1;
                                // if (groupId == 0) printf("threadInGroup %d: wrzucam %d na pos %d\n", threadInGroup, i * 32 + bit, pos);
                                pos++;
                            }
                        }
                    }
                }

                if (threadInGroup == GROUP_SIZE - 1) {
                    // printf("threadInGroup %d, pref %d, depth + 1 = %d, K - 1 = %d\n stack top %d\n", threadInGroup, pref, depth + 1, K - 1, stackTop);
                    cliques[(depth + 1) * NUM_BLOCKS * GROUPS_PER_BLOCK + blockIdx.x * GROUPS_PER_BLOCK + groupId] += pref;
                    cliques[(depth + 1) * NUM_BLOCKS * GROUPS_PER_BLOCK + blockIdx.x * GROUPS_PER_BLOCK + groupId] %= MOD;
                    // if (blockIdx.x == 0) printf("cliques %d")
                    stackTop = (depth + 1 < K - 1 ? stackTop + pref - 1 : stackTop - 1);
                    //maxStackTop[groupId] = stackTop;
                    // printf("new stack top %d\n", stackTop);
                }

                stackTop = __shfl_sync(mask, stackTop, groupId * GROUP_SIZE + (GROUP_SIZE - 1), GROUP_SIZE);            
            }

            if (threadInGroup == 0) {
                maxStackTop[groupId] = stackTop;
            }

            __syncthreads();

            // if (threadInGroup == 0) {
            //     printf("groupId %d: stackTop = %d, maxStackTop = %d\n", groupId, stackTop, maxStackTop[groupId]);
            // }

            // __syncthreads();

            for (int i = 1; i < GROUPS_PER_BLOCK; i *= 2) {
                if (threadInGroup == 0) {
                    maxStackTop[groupId] = max(maxStackTop[(groupId + i) % GROUPS_PER_BLOCK], maxStackTop[groupId]);
                }
                __syncthreads();
            }

        }

    }
}

int main(int argc, char* argv[]) {

    if (32 % GROUP_SIZE != 0) {
        std::cerr << "Warp size must be a multiple of GROUP_SIZE";
        return 1;       
    }
    if (BLOCK_SIZE % GROUP_SIZE != 0) {
        std::cerr << "BLOCK_SIZE must be a multiple of GROUP_SIZE";
        return 1;  
    }

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
                uint a = std::stoul(line, &idx);
                uint b = std::stoul(&(line[idx]));
                if (a != b) { // Ignorujemy pętle
                    edges.push_back({a, b});
                    degree[a]++;
                    degree[b]++;
                }
            }
            catch(std::exception) {
                std::cerr << "Error: invalid input data format\n";
                input.close();
                return 1;
            }
        }
        input.close();
    }
    else {
        std::cerr << "Error: unable to open input file\n";
        return 1;
    }

    int N = degree.size(); // Liczba wierzchołków

    // Przenumerowanie id na liczby z przedziału od 0 do N
    uint num = 0;
    for (std::map<uint, uint>::iterator it = degree.begin(); it != degree.end(); ++it) {
        id_to_number[it->first] = num;
        number_to_id[num] = it->first;
        num++;
    }

    // Skierowanie krawędzi
    // TODO: zrobić to na GPU
    for (int i = 0; i < edges.size(); i++) {
        uint a = edges[i].first;
        uint b = edges[i].second;
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
        if (r - l > MAX_DEG) {
            std::cerr << "Error: maximal degree after orienting greater than MAX_DEG\n";
            return 1;
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

    int cliques[NUM_BLOCKS * GROUPS_PER_BLOCK * K];

    cudaEvent_t start, stop;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, 0));

    std::pair<uint, uint>* devEdges;
    std::pair<int, int>* devIntervals;
    HANDLE_ERROR(cudaMalloc((void**)&devEdges, sizeof(std::pair<uint, uint>) * edges.size()));
    HANDLE_ERROR(cudaMalloc((void**)&devIntervals, sizeof(std::pair<int, int>) * intervals.size()));

    HANDLE_ERROR(cudaMemcpy(devEdges, &edges.front(), sizeof(std::pair<uint, uint>) * edges.size(), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(devIntervals, &intervals.front(), sizeof(std::pair<int, int>) * intervals.size(), cudaMemcpyHostToDevice));
  
    unsigned int* devIntersect;
    uint* devStackVertex;
    int* devStackDepth;
    HANDLE_ERROR(cudaMalloc((void**)&devIntersect, sizeof(unsigned int) * NUM_BLOCKS * GROUPS_PER_BLOCK * MAX_DEPTH * (MAX_DEG / 32)));
    HANDLE_ERROR(cudaMalloc((void**)&devStackVertex, sizeof(uint) * MAX_STACK * NUM_BLOCKS * GROUPS_PER_BLOCK));
    HANDLE_ERROR(cudaMalloc((void**)&devStackDepth, sizeof(int) * MAX_STACK * NUM_BLOCKS * GROUPS_PER_BLOCK));

    int* devCliques;
    HANDLE_ERROR(cudaMalloc((void**)&devCliques, sizeof(int) * NUM_BLOCKS * GROUPS_PER_BLOCK * K));
    HANDLE_ERROR(cudaMemset(devCliques, 0, sizeof(int) * NUM_BLOCKS * GROUPS_PER_BLOCK * K));

    unsigned int* devInducedSubrgaph;
    HANDLE_ERROR(cudaMalloc((void**)&devInducedSubrgaph, sizeof(unsigned int) * NUM_BLOCKS * MAX_DEG * (MAX_DEG / 32)));

    kcliques<<<NUM_BLOCKS, BLOCK_SIZE>>>(devEdges, devIntervals, N, devIntersect, devStackVertex, devStackDepth, devCliques, K, devInducedSubrgaph);

    HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));

	float elapsedTime;
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
	printf("Total GPU execution time: %3.1f ms\n", elapsedTime);

	HANDLE_ERROR(cudaEventDestroy(start));
	HANDLE_ERROR(cudaEventDestroy(stop));

    HANDLE_ERROR(cudaMemcpy(cliques, devCliques, sizeof(int) * NUM_BLOCKS * GROUPS_PER_BLOCK * K, cudaMemcpyDeviceToHost));

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
            for (int j = 0; j < NUM_BLOCKS * GROUPS_PER_BLOCK; j++) {
                sum += cliques[i * NUM_BLOCKS * GROUPS_PER_BLOCK + j];
                sum %= MOD;
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
