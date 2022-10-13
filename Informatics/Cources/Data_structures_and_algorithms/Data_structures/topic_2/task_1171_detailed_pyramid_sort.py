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

        # итерируемся по вершинам, которые не являются листовыми, справа налево
        # и выполняем операцию просеивания вниз для этих вершин
        for v in range(first_leaf, 0, -1):
            self.shift_down(node=v - 1)

    def shift_down(self, node: int, stop: int = None) -> int:
        if stop is None:
            stop = self.heap_size

        # находим детей элемента
        l_child = node*2 + 1
        r_child = node*2 + 2

        if r_child < stop and self.heap[r_child] > max(self.heap[l_child], self.heap[node]):
            self.heap[r_child], self.heap[node] = self.heap[node], self.heap[r_child]
            node = self.shift_down(node=r_child, stop=stop)
        elif l_child < stop and self.heap[l_child] > self.heap[node]:
            self.heap[l_child], self.heap[node] = self.heap[node], self.heap[l_child]
            node = self.shift_down(node=l_child, stop=stop)
        return node

    def pyramid_sort(self, verbose: bool = False) -> None:
        for stop in range(self.heap_size, 0, -1):
            if verbose:
                print(*self.heap[:stop])
            stop -= 1

            # меняем местами root и последнй элемент неотсортированной части кучи
            self.heap[stop], self.heap[0] = self.heap[0], self.heap[stop]

            # выполняем операцию просеивания вниз до последнего элемента, который мы только что поменяли
            self.shift_down(node=0, stop=stop)
        return


if __name__ == "__main__":

    if LOCAL_FILE:
        # считываем входные данные из файла
        input_data_path = Path(__file__).parent.resolve() / 'task_1171_input.txt'
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

    # инициализируем кучу
    heap = Heap(nums=nums)
    heap.pyramid_sort(verbose=True)
    print(*heap.heap)