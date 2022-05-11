import numpy as np



class Tree:
    def __init__(self,capacity):
        self.capacity = capacity # 葉子數量
        self.tree = np.zeros(self.capacity * 2 - 1) # 總節點數量
        self.start = capacity - 1 # 最左邊的葉子節點index

    def add(self,p,data):
        leaf_idx = self.start + p - 1

        if leaf_idx > len(self.tree) - 1:
            raise ValueError('Leaf index big then total leaf capacity.')

        self.tree[leaf_idx] = data
        self._update(leaf_idx)

    def get_tree(self):
        return self.tree

    def get_leaf(self,rand):
        p_idx = 0
        while True:
            L_idx = p_idx * 2 + 1
            R_idx = L_idx + 1

            if L_idx >= len(self.tree):
                break
            else:
                if rand <= self.tree[L_idx]:
                    p_idx = L_idx
                else:
                    rand -= self.tree[R_idx]
                    p_idx = R_idx

        return self.tree[p_idx]

    def get_total(self):
        return self.tree[0]

    def _update(self,leaf_idx):
        if leaf_idx == 0:
            return
        else:
            if leaf_idx % 2 == 0:
                R_idx = leaf_idx
                L_idx = leaf_idx - 1
            else:
                R_idx = leaf_idx + 1
                L_idx = leaf_idx

            p_idx = (leaf_idx - 1) // 2
            self.tree[p_idx] = self.tree[R_idx] + self.tree[L_idx]
            self._update(p_idx)


tree = Tree(4)
tree.add(1,3)
tree.add(2,1)
tree.add(3,4)
tree.add(4,2)

print(tree.get_leaf(4.1))
