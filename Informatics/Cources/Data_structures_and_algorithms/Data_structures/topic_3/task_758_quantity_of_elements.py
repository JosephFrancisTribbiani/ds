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

    def get_size(self) -> int:
        return self._get_size(node=self.root)

    def _get_size(self, node) -> int:
        if node is None:
            return 0
        
        # размер левого поддерева
        left_size = self._get_size(node=node.left)  
        
        # размер правого поддерева
        right_size = self._get_size(node=node.right)
        return left_size + right_size + 1


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
    tree_size = btree.get_size()
    print(tree_size)
