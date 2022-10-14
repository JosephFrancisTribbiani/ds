from typing import Union
from pathlib import Path


LOCAL_FILE = True


class Heap:
    def __init__(self, max_heap_size: int = None, nums: list = []) -> None:
        # инициализируем кучу
        self.heap = list()
        self.heap_size = 0
        self.max_heap_size = max_heap_size if max_heap_size else len(nums) + 1

        # на вход принимаем массив, из которого неоходимо сформировать бинарную кучу
        for num in nums:
            self.add_elem(elem=num)

    def add_elem(self, elem: Union[int, float]) -> int:
        # добавляем элемент в конец массива
        self.heap.append(elem)
        self.heap_size += 1
        
        # выполняем операцию всплытия элемента
        node = self.shift_up(node=len(self.heap) - 1)
        return node

    def remove_elem(self, idx: int) -> Union[None, int, float]:
        # проверяем, есть ли элемент с таким индексом
        if not self.heap or idx >= self.heap_size:
            return
        
        # меняем с последним элементом кучи, удаляем и выполняем операцию просеивания вверх и вниз
        self.heap[-1], self.heap[idx] = self.heap[idx], self.heap[-1]
        val = self.heap.pop()
        self.heap_size -= 1

        # восстанавливаем кучу
        if self.heap_size > idx:
            idx = self.shift_down(node=idx)
            self.shift_up(node=idx)
            
        return val

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

        if r_child < self.heap_size and self.heap[r_child] > max(self.heap[l_child], self.heap[node]):
            self.heap[r_child], self.heap[node] = self.heap[node], self.heap[r_child]
            node = self.shift_down(node=r_child)
        elif l_child < self.heap_size and self.heap[l_child] > self.heap[node]:
            self.heap[l_child], self.heap[node] = self.heap[node], self.heap[l_child]
            node = self.shift_down(node=l_child)
        return node

    def extract_max(self) -> int:
        if self.heap:
            # меняем местами root и последнй элемент кучи
            self.heap[-1], self.heap[0] = self.heap[0], self.heap[-1]

            # достаем последний элемент кучи
            val = self.heap.pop()
            self.heap_size -= 1

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
            if self.heap_size < self.max_heap_size:
                node = self.add_elem(elem=value)
                print(node + 1)
            else:
                print(-1)
        elif operation_code == 3:
            # удаление елемента
            idx = request[1] - 1
            val = self.remove_elem(idx=idx)
            print(val if val else -1)
        return


if __name__ == "__main__":

    if LOCAL_FILE:
        # считываем входные данные из файла
        input_data_path = Path(__file__).parent.resolve() / 'task_1168_input.txt'
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
