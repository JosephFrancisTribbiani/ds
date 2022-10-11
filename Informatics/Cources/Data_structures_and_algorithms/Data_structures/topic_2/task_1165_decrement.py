from re import L
from typing import Union
from pathlib import Path


class Heap:
    def __init__(self, nums: list) -> None:
        # инициализируем кучу
        self.heap = list()

        # на вход принимаем массив, из которого неоходимо сформировать бинарную кучу
        for num in nums:
            self.add_elem(elem=num)

    def add_elem(self, elem: Union[int, float]) -> None:
        # добавляем элемент в конец массива
        self.heap.append(elem)
        
        # выполняем операцию всплытия элемента
        self.shift_up(node=len(self.heap) - 1)
        return

    def shift_up(self, node: int) -> None:
        # находим родителя элемента
        parent = (node - 1) // 2

        # проверяем, что мы не root (т.е. node != 0) и что значение родителя элемента меньше значения node
        # если условия выполнены, меняем node с родителем и запускаем shift_up из новой вершины
        if node and self.heap[node] > self.heap[parent]:
            self.heap[node], self.heap[parent] = self.heap[parent], self.heap[node]
            node = self.shift_up(node=parent)
        return node

    def shift_down(self, node: int) -> None:
        # находим детей элемента
        l_child = node*2 + 1
        r_child = node*2 + 2

        if r_child < len(self.heap) and self.heap[r_child] > max(self.heap[l_child], self.heap[node]):
            self.heap[r_child], self.heap[node] = self.heap[node], self.heap[r_child]
            node = self.shift_down(node=r_child)
        elif l_child < len(self.heap) and self.heap[l_child] > self.heap[node]:
            self.heap[l_child], self.heap[node] = self.heap[node], self.heap[l_child]
            node = self.shift_down(node=l_child)
        return node

    def increment(self, idx: int, val: Union[int, float]) -> None:
        """
        Метод увеличивает значение элемента кучи с индексом idx на значение val и восстанавливает кучу
        """
        self.heap[idx] += val
        node = self.shift_up(node=idx)
        return node

    def decrement(self, idx: int, val: Union[int, float]) -> None:
        """
        Метод уменьшает значение элемента кучи с индексом idx на значение val и восстанавливает кучу
        """
        self.heap[idx] -= val
        node = self.shift_down(node=idx)
        return node


if __name__ == "__main__":
    # считываем входные данные из фйла
    input_data_path = Path(__file__).parent.resolve() / 'task_1165_input.txt'
    with open(input_data_path, 'r') as file:
        # количество элементов в куче
        n = int(file.readline())

        # инициализируем кучу
        nums = list(map(int, file.readline().split()))

        # выполняем операции increment
        n_requests = int(file.readline())
        requests = [tuple(map(int, file.readline().split())) for _ in range(n_requests)]

    # # считываем данные через терминал (для informatics)
    # # количество элементов в куче
    # n = int(input())

    # # элементы кучи
    # nums = list(map(int, input().split()))

    # # выполняем операции increment
    # n_requests = int(input())
    # requests = [tuple(map(int, input().split())) for _ in range(n_requests)]

    # инициализируем кучу
    heap = Heap(nums=nums)    

    for idx, val in requests:
        node = heap.decrement(idx=idx - 1, val=val)
        print(node + 1)

    print(*heap.heap)
