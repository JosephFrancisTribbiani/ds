from importlib.resources import path
from pathlib import Path


# считываем исходные данные из файла
input_file_path = Path(__file__).parent.resolve() / 'task_6_input.txt'
with open(input_file_path, 'r') as file:
    n, start_vertex, finish_vertex = list(map(int, file.readline().split()))
    start_vertex, finish_vertex = start_vertex - 1, finish_vertex - 1
    graph = [list(map(int, file.readline().split())) for _ in range(n)]

# # считывание исходных данных через терминал (для informatics)
# n, start_vertex, finish_vertex = list(map(int, input().split()))
# start_vertex, finish_vertex = start_vertex - 1, finish_vertex - 1
# graph = [list(map(int, input().split())) for _ in range(n)]

# формируем список посещенных вершин
visited = [False for _ in range(n)]

# формируем вектор расстояний до стартовой вершины
distances = [float('inf') for _ in range(n)]
distances[start_vertex] = 0
pathes = [list() for _ in range(n)]
pathes[start_vertex] = [start_vertex + 1]

for _ in range(n):
    # находим минимальное значение расстояния
    curr_vertex, dist_to_start = None, float('inf')
    for vertex, distance in enumerate(distances):
        if not visited[vertex] and (distance <= dist_to_start):
            curr_vertex, dist_to_start = vertex, distance
    
    # делаем текущую вершину посещенной
    visited[curr_vertex] = True
    
    # смотрим на непосещенных соседей выбраной вершины
    for neighbor, dist_to_neighbor in enumerate(graph[curr_vertex]):
        if not visited[neighbor] and (dist_to_neighbor != -1) and \
            ((dist_to_start + dist_to_neighbor) < distances[neighbor]):
            distances[neighbor] = dist_to_start + dist_to_neighbor
            pathes[neighbor] = pathes[curr_vertex] + [neighbor + 1]

if len(pathes[finish_vertex]) > 0:
    print(*pathes[finish_vertex])
else:
    print(-1)
