from pathlib import Path


# # считываем данные из файла и задаём граф в виде матрицы смежности
# input_file_path = Path(__file__).parent.resolve() / 'task_7_input_1.txt'
# with open(input_file_path, 'r') as file:
#     n = int(file.readline())  # количество городов
#     prices = list(map(int, file.readline().split()))  # стоимость бензина в каждом городе
#     graph = [[0 if i == j else -1 for j in range(n)] for i in range(n)]
#     m = int(file.readline())  # количество дорог в стране
#     for _ in range(m):
#         city_from, city_to = list(map(int, file.readline().split()))
#         city_from, city_to = city_from - 1, city_to - 1
#         graph[city_from][city_to] = prices[city_from]
#         graph[city_to][city_from] = prices[city_to]

# считываем данные через терминал (для informatics) и задаём граф в виде матрицы смежности
n = int(input())  # количество городов
prices = list(map(int, input().split()))  # стоимость бензина в каждом городе
graph = [[0 if i == j else -1 for j in range(n)] for i in range(n)]
m = int(input())  # количество дорог в стране
for _ in range(m):
    city_from, city_to = list(map(int, input().split()))
    city_from, city_to = city_from - 1, city_to - 1
    graph[city_from][city_to] = prices[city_from]
    graph[city_to][city_from] = prices[city_to]

# решаем задачу нахождения оптимального пути с помощтю наивного алгоритма Дейкстры

# задаем вектор посещенных городов
visited = [False for _ in range(n)]

# задаем вектор расстояний от старта
dist = [float('inf') for _ in range(n)]
dist[0] = 0

# итерируемся по всем городам
for _ in range(n):
    # отметим, что город пока не выбран
    u = -1

    # итерируемся по всем городам и ищем непосещенный, с минимальным значением дистанции
    for i in range(n):
        if not visited[i] and (u == -1 or dist[i] < dist[u]):
            u = i

    # если все города посещены или мы не можем посетить этот город, то прерываем цикл
    if dist[u] == float('inf'):
        break

    # отмечаем выбранный город как посещенный
    visited[u] = True

    for v, l in enumerate(graph[u]):
        if not visited[v] and (l != -1) and (dist[u] + l < dist[v]):
            dist[v] = dist[u] + l

print(dist[-1] if dist[-1] != float('inf') else -1)
