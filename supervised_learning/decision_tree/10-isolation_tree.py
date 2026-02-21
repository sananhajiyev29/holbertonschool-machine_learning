#!/usr/bin/env python3
"""
10-isolation_tree.py

Isolation Random Trees:
- Train using random splits only (no target classes)
- Prediction returns depth of the leaf an individual falls into
- Useful for outlier detection (small mean leaf depth => likely outlier)
"""

Node = __import__("8-build_decision_tree").Node
Leaf = __import__("8-build_decision_tree").Leaf
import numpy as np


class Isolation_Random_Tree:
    """Isolation Random Tree."""

    def __init__(self, max_depth=10, seed=0, root=None):
        """Initialize Isolation_Random_Tree."""
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def __str__(self):
        """String representation (same behavior as Decision_Tree)."""
        return str(self.root)

    def depth(self):
        """Return maximum depth (same behavior as Decision_Tree)."""

        def rec(n):
            if getattr(n, "is_leaf", False):
                return n.depth
            return max(rec(n.left_child), rec(n.right_child))

        return rec(self.root)

    def count_nodes(self, only_leaves=False):
        """Count nodes (same behavior as Decision_Tree)."""

        def rec(n):
            if getattr(n, "is_leaf", False):
                return 1
            return 1 + rec(n.left_child) + rec(n.right_child)

        def rec_leaves(n):
            if getattr(n, "is_leaf", False):
                return 1
            return rec_leaves(n.left_child) + rec_leaves(n.right_child)

        if only_leaves:
            return rec_leaves(self.root)
        return rec(self.root)

    def update_bounds(self):
        """Compute bounds for whole tree (same behavior as Decision_Tree)."""
        self.root.update_bounds_below()

    def get_leaves(self):
        """Return all leaves (same behavior as Decision_Tree)."""
        return self.root.get_leaves_below()

    def update_predict(self):
        """
        Build efficient batch predict (same structure as Decision_Tree),
        but prediction values are leaf.depth.
        """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        def _predict(a):
            preds = np.zeros(a.shape[0], dtype=int)
            for leaf in leaves:
                m = leaf.indicator(a)
                preds[m] = int(leaf.value)
            return preds

        self.predict = _predict

    def np_extrema(self, arr):
        """Return min and max of arr."""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Pick a random valid (feature, threshold) (same as Decision_Tree)."""
        diff = 0.0
        while diff == 0.0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            vals = self.explanatory[:, feature][node.sub_population]
            feature_min, feature_max = self.np_extrema(vals)
            diff = feature_max - feature_min

        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return int(feature), float(threshold)

    def get_leaf_child(self, node, sub_population):
        """
        Create a Leaf child (different from Decision_Tree):
        leaf value is the depth of the leaf.
        """
        leaf_child = Leaf(value=node.depth + 1)
        leaf_child.depth = node.depth + 1
        leaf_child.subpopulation = sub_population
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Create a Node child (same as Decision_Tree)."""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        """Recursively fit a node using random splits."""
        node.feature, node.threshold = self.random_split_criterion(node)

        go_left = self.explanatory[:, node.feature] > node.threshold
        left_population = np.logical_and(node.sub_population, go_left)
        right_population = np.logical_and(node.sub_population, ~go_left)

        left_size = int(np.sum(left_population))
        right_size = int(np.sum(right_population))

        is_left_leaf = (left_size < self.min_pop) or (
            (node.depth + 1) >= self.max_depth
        )
        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        is_right_leaf = (right_size < self.min_pop) or (
            (node.depth + 1) >= self.max_depth
        )
        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        """Train isolation tree on explanatory only."""
        self.split_criterion = self.random_split_criterion
        self.explanatory = explanatory

        n = explanatory.shape[0]
        self.root.sub_population = np.ones(n, dtype="bool")

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(
                "  Training finished.\n"
                f"    - Depth                     : {self.depth()}\n"
                f"    - Number of nodes           : {self.count_nodes()}\n"
                f"    - Number of leaves          : "
                f"{self.count_nodes(only_leaves=True)}"
            )
