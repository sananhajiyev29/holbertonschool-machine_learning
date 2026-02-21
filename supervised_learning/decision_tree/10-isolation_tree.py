#!/usr/bin/env python3
"""
10-isolation_tree.py

Isolation Random Trees:
- Train using random splits only (no target classes)
- Prediction returns the depth of the leaf an individual falls into
- Outliers tend to reach shallow leaves (small depth)
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
        """String representation (same as in Decision_Tree)."""
        return str(self.root)

    def depth(self):
        """Return maximum depth (same as in Decision_Tree)."""

        def rec(n):
            if getattr(n, "is_leaf", False):
                return n.depth
            return max(rec(n.left_child), rec(n.right_child))

        return rec(self.root)

    def count_nodes(self, only_leaves=False):
        """Count nodes (same as in Decision_Tree)."""

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
        """Compute bounds for whole tree (same as in Decision_Tree)."""
        self.root.update_bounds_below()

    def get_leaves(self):
        """Return all leaves (same as in Decision_Tree)."""
        return self.root.get_leaves_below()

    def update_predict(self):
        """
        Build efficient batch prediction function.

        Here, leaf.value is the depth of the leaf.
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
        """
        Pick a random (feature, threshold) valid for node population.

        Same spirit as Decision_Tree, but safe: if no feature has spread,
        return a dummy split (caller should not rely on it if leaf).
        """
        pop = int(np.sum(node.sub_population))
        if pop <= 1:
            return 0, 0.0

        diff = 0.0
        tries = 0
        n_features = self.explanatory.shape[1]

        while diff == 0.0 and tries < 10 * n_features:
            feature = int(self.rng.integers(0, n_features))
            vals = self.explanatory[:, feature][node.sub_population]
            feature_min, feature_max = self.np_extrema(vals)
            diff = float(feature_max - feature_min)
            tries += 1

        if diff == 0.0:
            return 0, 0.0

        x = float(self.rng.uniform())
        threshold = (1 - x) * float(feature_min) + x * float(feature_max)
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        """
        Create a Leaf child (different from Decision_Tree):
        leaf.value is the depth of the leaf.
        """
        leaf_child = Leaf(value=node.depth + 1)
        leaf_child.depth = node.depth + 1
        leaf_child.subpopulation = sub_population
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Create a Node child (same as in Decision_Tree)."""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def _must_be_leaf(self, node, sub_population):
        """
        Leaf condition for isolation tree:
        - population size <= min_pop  (IMPORTANT: prevents infinite loops)
        - OR reached max_depth
        - OR all features have zero spread (cannot split)
        """
        pop = int(np.sum(sub_population))
        if pop <= self.min_pop:
            return True
        if (node.depth + 1) >= self.max_depth:
            return True

        idx = np.where(sub_population)[0]
        x_sub = self.explanatory[idx, :]
        spread = np.max(x_sub, axis=0) - np.min(x_sub, axis=0)
        return np.all(spread == 0)

    def fit_node(self, node):
        """Recursively fit a node using random splits."""
        node.feature, node.threshold = self.random_split_criterion(node)

        go_left = self.explanatory[:, node.feature] > node.threshold
        left_population = np.logical_and(node.sub_population, go_left)
        right_population = np.logical_and(node.sub_population, ~go_left)

        is_left_leaf = self._must_be_leaf(node, left_population)
        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        is_right_leaf = self._must_be_leaf(node, right_population)
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
