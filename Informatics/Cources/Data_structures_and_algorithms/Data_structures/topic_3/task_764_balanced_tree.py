from pathlib import Path
from typing import Union
import sys


LOCAL_FILE = False


class Node:
    def __init__(self, key: Union[int, float]):
        self.key = key
        self.left = None
        self.right = None


class Tree:
    def __init__(self):
        self.root = None

    def add(self, key: Union[int, float]):
        if not self.root:
            self.root = Node(key=key)
            return

        node = self.root

        while node:
            if key < node.key:
                if not node.left:
                    node.left = Node(key=key)
                    return
                else:
                    node = node.left
            elif key > node.key:
                if not node.right:
                    node.right = Node(key=key)
                    return
                else:
                    node = node.right
            else:
                return
        return

    def check_balance(self):
        if self.root:
            self._check_balance(node=self.root)
        return

    def _check_balance(self, node):
        if node:
            l_depth = self._check_balance(node=node.left)
            r_depth = self._check_balance(node=node.right)
            if abs(l_depth - r_depth) > 1:
                print("NO")
                sys.exit(0)
            return max(l_depth, r_depth) + 1
        return 0


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
    btree.check_balance()
    print("YES")