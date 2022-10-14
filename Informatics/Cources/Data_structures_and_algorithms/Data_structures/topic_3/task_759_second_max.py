from pathlib import Path
from typing import Union


LOCAL_FILE = False


class Node:
    def __init__(self, key=Union[int, float]) -> None:
        self.key = key
        self.left = None
        self.right = None


class Tree:
    def __init__(self) -> None:
        self.root = None

    def add(self, key: Union[int, float]) -> None:
        if self.root is None:
            self.root = Node(key=key)
        else:
            self._add(key=key, node=self.root)
        return
    
    def _add(self, key: Union[int, float], node) -> None:
        if key < node.key:
            if node.left is not None:
                self._add(key=key, node=node.left)
            else:
                node.left = Node(key=key)
        elif key > node.key:
            if node.right is not None:
                self._add(key=key, node=node.right)
            else:
                node.right = Node(key=key)
        else:
            # если элемент уже присутствует в дереве, то добавлять его не нужно (условие задачи)
            pass
        return

    def get_max(self):
        """
        Функция возвращает максимальный элемент дерева
        """
        if self.root is not None:
            return self._get_max(node=self.root)
        return

    def _get_max(self, node):
        if node.right is not None:
            return self._get_max(node=node.right)
        return node.key

    def get_second_max(self):
        """
        Функция возращает второй по величине элемент дерева
        """
        if self.root is not None:
            return self._get_second_max(node=self.root)
        return

    def _get_second_max(self, node):
        if not node.right and node.left:
            return self._get_max(node=node.left)
        elif node.right and (node.right.left or node.right.right):
            return self._get_second_max(node=node.right)
        return node.key


if __name__ == "__main__":

    if LOCAL_FILE:
        # считываем входные данные из файла
        input_data_path = Path(__file__).parent.resolve() / 'topic_3_input.txt'
        with open(input_data_path, 'r') as file:
            # массив значений
            nums = list(map(int, file.readline().split()))

    else:
        # считываем данные через терминал (для informatics)
        # массив значений
        nums = list(map(int, input().split()))

    # инициализируем дерево
    btree = Tree()

    # добавляем вершины в дерево
    for num in nums:
        if num == 0:
            break
        
        btree.add(key=num)
    # необходимо вывести второй по величине элемент бинарного дерева
    # гарантируется, что такой элемент в дереве есть
    # самый максимальный элемент дерева - самый правый в дереве
    # второй по величине - его родитель, если поддерево слева отсутствует,
    # и самый правый элемент поддерева слева, если поддерево слева существует.
    print(btree.get_second_max())
