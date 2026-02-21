#!/usr/bin/env python3
"""
7-build_decision_tree.py
Decision Tree with random splitting.
"""

import numpy as np


class Node:
    """Internal node."""

    def __init__(
        self,
        feature=None,
        threshold=None,
        left_child=None,
        right_child=None,
        depth=0,
        is_root=False
    ):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.depth = depth
        self.is_root = is_root
        self.is_leaf = False
        self.sub_population = None

    def get_leaves_below(self):
        """Return leaves below."""
        leaves = []
        for child in [self.left_child, self.right_child]:
            leaves.extend(child.get_leaves_below())
        return leaves

    def pred(self, x):
        """Recursive prediction."""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        return self.right_child.pred(x)


class Leaf:
    """Leaf node."""

    def __init__(self, value, depth=0):
        self.value = value
        self.depth = depth
        self.is_leaf = True
        self.sub_population = None

    def get_leaves_below(self):
        """Return itself."""
        return [self]

    def pred(self, x):
        """Return value."""
        return self.value


class Decision_Tree:
    """Decision Tree."""

    def __init__(
        self,
        split_criterion="random",
        max_depth=10,
        min_pop=1,
        seed=0
    ):
        self.rng = np.random.default_rng(seed)
        self.split_criterion = split_criterion
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.root = Node(is_root=True, depth=0)
        self.explanatory = None
        self.target = None
        self.predict = None

    def depth(self):
        """Return max depth."""
        leaves = self.get_leaves()
        return max(leaf.depth for leaf in leaves)

    def count_nodes(self, only_leaves=False):
        """Count nodes."""
        stack = [self.root]
        count = 0
        while stack:
            n = stack.pop()
            if only_leaves:
                if getattr(n, "is_leaf", False):
                    count += 1
            else:
                count += 1
            if not getattr(n, "is_leaf", False):
                stack.append(n.left_child)
                stack.append(n.right_child)
        return count

    def get_leaves(self):
        """Return all leaves."""
        return self.root.get_leaves_below()

    def np_extrema(self, arr):
        """Return min and max."""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Random split."""
        diff = 0
        while diff == 0:
            feature = self.rng.integers(
                0,
                self.explanatory.shape[1]
            )
            vals = self.explanatory[:, feature][
                node.sub_population
            ]
            fmin, fmax = self.np_extrema(vals)
            diff = fmax - fmin
        x = self.rng.uniform()
        threshold = (1 - x) * fmin + x * fmax
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        """Create leaf."""
        y = self.target[sub_population]
        classes, counts = np.unique(
            y,
            return_counts=True
        )
        value = classes[np.argmax(counts)]
        leaf = Leaf(value=value, depth=node.depth + 1)
        leaf.sub_population = sub_population
        return leaf

    def get_node_child(self, node, sub_population):
        """Create internal node."""
        n = Node(depth=node.depth + 1)
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        """Recursive training."""
        node.feature, node.threshold = \
            self.split_criterion(node)

        go_left = (
            self.explanatory[:, node.feature]
            > node.threshold
        )

        left_pop = node.sub_population & go_left
        right_pop = node.sub_population & (~go_left)

        left_size = np.sum(left_pop)
        right_size = np.sum(right_pop)

        left_pure = np.unique(
            self.target[left_pop]
        ).size <= 1

        right_pure = np.unique(
            self.target[right_pop]
        ).size <= 1

        next_depth = node.depth + 1

        left_leaf = (
            left_size < self.min_pop
            or next_depth == self.max_depth
            or left_pure
        )

        right_leaf = (
            right_size < self.min_pop
            or next_depth == self.max_depth
            or right_pure
        )

        if left_leaf:
            node.left_child = \
                self.get_leaf_child(node, left_pop)
        else:
            node.left_child = \
                self.get_node_child(node, left_pop)
            self.fit_node(node.left_child)

        if right_leaf:
            node.right_child = \
                self.get_leaf_child(node, right_pop)
        else:
            node.right_child = \
                self.get_node_child(node, right_pop)
            self.fit_node(node.right_child)

    def update_predict(self):
        """Vectorized predict."""
        leaves = self.get_leaves()
        values = np.array(
            [leaf.value for leaf in leaves]
        )

        def _predict(a):
            preds = []
            for leaf in leaves:
                mask = leaf.sub_population
                preds.append(leaf.value)
            return np.array(
                [self.root.pred(x) for x in a]
            )

        self.predict = _predict

    def fit(self, explanatory, target, verbose=0):
        """Train tree."""
        if self.split_criterion == "random":
            self.split_criterion = \
                self.random_split_criterion

        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = \
            np.ones_like(target, dtype=bool)

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print("Training finished.\n")
            print("Depth :", self.depth())
            print("Number of nodes :",
                  self.count_nodes())
            print("Number of leaves :",
                  self.count_nodes(
                      only_leaves=True
                  ))
            print("Accuracy on training data :",
                  self.accuracy(
                      self.explanatory,
                      self.target
                  ))

    def accuracy(self, test_explanatory, test_target):
        """Return accuracy."""
        pred = self.predict(test_explanatory)
        return np.sum(
            np.equal(pred, test_target)
        ) / test_target.size
