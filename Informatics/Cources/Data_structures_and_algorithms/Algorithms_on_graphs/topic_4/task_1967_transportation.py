from multiprocessing import allow_connection_pickling
from pathlib import Path
from collections import deque


N_CUPS =       1e7   # всего количество для доставки
MAX_TIME =     1440  # максимальное время на доставку груза в минутах (т.е. 24 часа)
CUP_WEIGHT =   100   # вес одной кружки в граммах
TRUCK_WEIGHT = 3e6   # вес грузовика в граммах
LOCAL_FILE =   True

# считываем данные из файла и задаём граф в виде матрицы смежности
if LOCAL_FILE:
    input_file_path = Path(__file__).parent.resolve() / "task_1967_input.txt"
    with open(input_file_path, 'r') as file:
        n, m = list(map(int, file.readline().split()))  # количество узловых пунктов и количество дорог соответственно
        # собираем информацию о дорогах
        graph = {k: list() for k in range(n)}
        for _ in range(m):
            n_from, n_to, t, w = list(map(int, file.readline().split()))
            n_from, n_to = n_from - 1, n_to - 1
            # где
            #   n_from, n_to - узловые пункты, соединенные дорогой
            #   t - время прохождения участка дороги (в минутах)
            #   w - максимальный вес грузовика (в граммах)
            graph[n_from].append((n_to, t, w))
            graph[n_to].append((n_from, t, w))
else:
    n, m = list(map(int, input().split()))  # количество узловых пунктов и количество дорог соответственно
    # собираем информацию о дорогах
    graph = {k: list() for k in range(n)}
    for _ in range(m):
        n_from, n_to, t, w = list(map(int, input().split()))
        n_from, n_to = n_from - 1, n_to - 1
        # где
        #   n_from, n_to - узловые пункты, соединенные дорогой
        #   t - время прохождения участка дороги (в минутах)
        #   w - максимальный вес грузовика (в граммах)
        graph[n_from].append((n_to, t, w))
        graph[n_to].append((n_from, t, w))

# алгоритм Дейкстры позволит найти все города, до которых возможно добраться за время 
