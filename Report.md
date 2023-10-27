# Clique counting - zadanie CUDA

### OPIS ROZWIĄZANIA
Podejście **vertex-center, graph-orientitanion**. Rozwiązanie identyczne z opisanym w artykule.

Po wczytaniu opisu grafu id wierzchołków są zamieniane na liczby od 0 do N-1, gdzie N to liczba wierzchołków, a krawędzie orientowane w stronę wierzchołka o większym stopniu. Preprocessing ten wykonywany jest na CPU, ponieważ jest mało kosztowny w porównaniu z przeszukiwaniem grafu. Graf w postaci listy krawędzi oraz przedziałów odpowiadających krawędziom incydentnym z danym wierzchołkiem jest przekazywany do kernela.

W podejściu vertex-center blok jest odpowiedzialny za przeszukanie podgrafu indukowanego listą sąsiadów wierzchołka. Jeśli bloków jest mniej niż wierzchołków, to blok zajmuje się kolejno wierzchołkami z przydzielonego sobie przedziału. Najpierw wszystkie wątki bloku współpracują, aby wyekstrahować podgraf indukowany i zapisać go jako macierz sąsiedztwa, a następnie dzielą się na grupy i każda z nich niezależnie przeszukuje poddrzewo jednego dziecka korzenia algorytmem DFS. Algorytm DFS jest wykonywany sekwencyjnie przy użyciu stosu. Dla każdej grupy każdego bloku należy wcześniej zaalokować miejsce na ten stos. Ale ponieważ po zorientowaniu krawędzi działamy na DAGu, rozmiar ten będzie maksymalnie kwadratowy względem liczby wierzchołków indukowanego podgrafu. Dodatkowo w algorytmie DFS musimy utrzymywać przecięcie list sąsiedztwa na każdym poziomie na ścieżce do aktualnie przetwarzanego wierzchołka. Pamięć na to również alokujemy wcześniej dla każdej grupy każdego bloku. W każdym kroku wątki w grupie dzielą się pracą przy wykonywaniu przecięcia listy sąsiedztwa aktualnie przetwarzanego wierzchołka z przecięciem na poprzednim poziomie, a następnie komunikują ze sobą, aby ustalić, w którym miejscu na stosie każdy wątek ma umieścić przeglądanych przez siebie sąsiadów. Po umieszczeniu sąsiadów na stosie wszystkie grupy wątków synchronizują się przy użyciu pamięci shared, aby ustalić, czy któraś z nich ma jeszcze jakiś wierzchołek na stosie. Jeśli maksymalny rozmiar stosu będzie zerowy, to przeszukiwanie tego indukowanego podgrafu się kończy.

Wyniki zapisywane są w tablicy, przekazywanej jako parametr wywołania kernela. Dla każdej grupy każdego bloku, dla każdego i od 1 do k istnieje miejsce w tej tablicy, w którym zapisywana jest liczba klik wielkości i zliczonych przez tę grupę. Po wykonaniu kernela zliczającego kliki kolejny kernel wykonuje redukcję na tej tablicy uzyskując sumaryczne wyniki dla każdego i od 1 do k.

### OPTYMALIZACJE
- **Binary encoding** - kodowanie binarne macierzy sąsiedztwa indukowanego podgrafu pozwala użyć do tego 8 razy mniej pamięci (1 bit zamiast 8-bitowego boola), przez co można zaalokować pamięć na więcej współbieżnych wykonań DFSa.
- **Sub-warp partitioning** - podział bloku na grupy o rozmiarze mniejszym niż rozmiar warpa i synchronizacja wątków w grupach przy użyciu low cost warp-level primitives (https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)

### ZAPOŻYCZENIA
- **`Makefile`** - identyczny z dostępnymi w Cuda Samples (https://github.com/NVIDIA/cuda-samples) i używanymi na laboratoriach
- **`errors.h`** - biblioteczka używana na laboratoriach, pozwalająca na np. ładne wypisywanie komunikatów o błędzie w przypadku próby zaalokowania pamięci na GPU ponad limit

