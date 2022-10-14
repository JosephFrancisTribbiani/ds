from pathlib import Path
from typing import Union


LOCAL_FILE = False


class Node:
    def __init__(self, key: Union[int, float]) -> None:
        self.key = key
        self.nunique = 1
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
                node.nunique += 1
                return
        return

    def print_tree(self):
        if self.root:
            self._print_tree(node=self.root)
        return

    def _print_tree(self, node):
        if node:
            self._print_tree(node=node.left)
            print(node.key, node.nunique)
            self._print_tree(node=node.right)
        return


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
    btree.print_tree()
