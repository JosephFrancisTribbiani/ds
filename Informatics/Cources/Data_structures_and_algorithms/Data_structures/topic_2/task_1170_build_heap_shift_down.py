from typing import Union
from pathlib import Path


LOCAL_FILE = False


class Heap:
    def __init__(self, nums: list = []) -> None:
        # инициализируем кучу
        self.heap = nums
        self.heap_size = len(nums)

        # восстанавливаем кучу

        # находим индекс первого листа
        first_leaf = self.heap_size // 2

        # итерируемся по верщинам, которые не являются листовыми, справа налево
        # и выполняем операцию просеивания вниз для этих вершин
        for v in range(first_leaf, 0, -1):
            self.shift_down(node=v - 1)


    def shift_down(self, node: int) -> int:
        # находим детей элемента
        l_child = node*2 + 1
        r_child = node*2 + 2

        if r_child < self.heap_size and self.heap[r_child] > max(self.heap[l_child], self.heap[node]):
            self.heap[r_child], self.heap[node] = self.heap[node], self.heap[r_child]
            node = self.shift_down(node=r_child)
        elif l_child < self.heap_size and self.heap[l_child] > self.heap[node]:
            self.heap[l_child], self.heap[node] = self.heap[node], self.heap[l_child]
            node = self.shift_down(node=l_child)
        return node


if __name__ == "__main__":

    if LOCAL_FILE:
        # считываем входные данные из файла
        input_data_path = Path(__file__).parent.resolve() / 'task_1170_input.txt'
        with open(input_data_path, 'r') as file:
            # количество элементов в куче
            n = int(file.readline())

            # массив значений
            nums = list(map(int, file.readline().split()))
    else:
        # считываем данные через терминал (для informatics)
        # количество элементов в куче
        n = int(input())

        # массив значений
        nums = list(map(int, input().split()))


    heap = Heap(nums=nums)
    print(*heap.heap)