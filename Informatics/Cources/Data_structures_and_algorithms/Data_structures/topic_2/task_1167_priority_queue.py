from typing import Union
from pathlib import Path


LOCAL_FILE = True


class Heap:
    def __init__(self, max_heap_size: int = None, nums: list = []) -> None:
        # инициализируем кучу
        self.heap = list()
        self.max_heap_size = max_heap_size if max_heap_size else len(nums) + 1

        # на вход принимаем массив, из которого неоходимо сформировать бинарную кучу
        for num in nums:
            self.add_elem(elem=num)

    def add_elem(self, elem: Union[int, float]) -> int:
        # добавляем элемент в конец массива
        self.heap.append(elem)
        
        # выполняем операцию всплытия элемента
        node = self.shift_up(node=len(self.heap) - 1)
        return node

    def shift_up(self, node: int) -> int:
        # находим родителя элемента
        parent = (node - 1) // 2

        # проверяем, что мы не root (т.е. node != 0) и что значение родителя элемента меньше значения node
        # если условия выполнены, меняем node с родителем и запускаем shift_up из новой вершины
        if node and self.heap[node] > self.heap[parent]:
            self.heap[node], self.heap[parent] = self.heap[parent], self.heap[node]
            node = self.shift_up(node=parent)
        return node

    def shift_down(self, node: int) -> int:
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

    def extract_max(self) -> int:
        if self.heap:
            # меняем местами root и последнй элемент кучи
            self.heap[-1], self.heap[0] = self.heap[0], self.heap[-1]

            # достаем последний элемент кучи
            val = self.heap.pop()

            # просейваем root вниз
            node = self.shift_down(node=0)

            return node, val

    def request_proccess(self, request: list) -> None:
        operation_code = request[0]
        if operation_code == 1:
            # extract max elem
            resp = self.extract_max()
            if not resp:
                print(-1)
                return
            elif self.heap:
                node, val = resp
            else:
                node, val = -1, resp[1]
            print(node + 1, val)
        elif operation_code == 2:
            value = request[1]
            # add elem
            if len(self.heap) < self.max_heap_size:
                node = self.add_elem(elem=value)
                print(node + 1)
            else:
                print(-1)
        return


if __name__ == "__main__":

    if LOCAL_FILE:
        # считываем входные данные из файла
        input_data_path = Path(__file__).parent.resolve() / 'task_1167_input.txt'
        with open(input_data_path, 'r') as file:
            # количество элементов в куче
            n, m = list(map(int, file.readline().split()))

            # считываем запросы
            requests = list()
            for _ in range(m):
                requests.append(list(map(int, file.readline().split())))
    else:
        # считываем данные через терминал (для informatics)
        # количество элементов в куче
        n, m = list(map(int, input().split()))

        # считываем запросы
        requests = list()
        for _ in range(m):
            requests.append(list(map(int, input().split())))


    heap = Heap(max_heap_size=n)

    for r in requests:
        heap.request_proccess(request=r)
    print(*heap.heap)
