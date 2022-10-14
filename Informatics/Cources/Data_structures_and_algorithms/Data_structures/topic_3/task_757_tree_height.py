from pathlib import Path
from typing import Union


LOCAL_FILE = False


class Node:
    def __init__(self, key: Union[int, float]) -> None:
        self.key = key
        self.left = None
        self.right = None


class Tree:
    def __init__(self) -> None:
        self.root = None

    def get_root(self):
        return self.root

    def add(self, key: Union[int, float]) -> None:
        # если у дерева нет корня, то добавляем первый элемент в качестве корня дерева
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

    def print_tree(self) -> None:
        """
        Метод печатает дерево в порядке возрастания элементов
        """
        if self.root is not None:
            self._print_tree(node=self.root)
        return

    def _print_tree(self, node) -> None:
        if node is not None:
            self._print_tree(node=node.left)
            print(node.key)
            self._print_tree(node=node.right)
        return

    def get_depth(self) -> int:
        return self._get_depth(node=self.root)

    def _get_depth(self, node) -> int:
        if node is None:
            return 0

        # глубина левого поддерева
        l_depth = self._get_depth(node=node.left)

        # глубина правого поддерева
        r_depth = self._get_depth(node=node.right)
        return max(l_depth, r_depth) + 1
        
        
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
    print(btree.get_depth())
