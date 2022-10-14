from pathlib import Path


# считываем исходные данные из файла
input_file_path = Path(__file__).parent.resolve() / 'task_5_input.txt'
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
distances = [float('inf') for v in range(n)]
distances[start_vertex] = 0

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
        if not visited[neighbor] and (dist_to_neighbor != -1):
            distances[neighbor] = min(distances[neighbor], dist_to_start + dist_to_neighbor)

print(distances[finish_vertex] if distances[finish_vertex] != float('inf') else -1)
