#!/usr/bin/env python3
"""
8-build_decision_tree.py

Trainable decision tree:
- Random split criterion
- Gini impurity split criterion
- Recursive fitting
- Bounds + indicators
- Efficient batch prediction
"""

import numpy as np


class Node:
    """Internal decision tree node."""

    def __init__(
        self,
        feature=None,
        threshold=None,
        left_child=None,
        right_child=None,
        depth=0,
        is_root=False
    ):
        """Initialize a Node."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.depth = depth
        self.is_root = is_root

        self.is_leaf = False
        self.lower = {}
        self.upper = {}
        self.indicator = None
        self.sub_population = None

    def get_leaves_below(self):
        """Return all leaves below this node."""
        leaves = []
        for child in [self.left_child, self.right_child]:
            leaves.extend(child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        """
        Recursively compute lower/upper bounds dictionaries.

        Convention:
        left_child  -> feature > threshold   (update lower bound)
        right_child -> feature <= threshold  (update upper bound)
        """
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1 * np.inf}

        f = self.feature
        t = self.threshold

        for child in [self.left_child, self.right_child]:
            child.lower = dict(self.lower)
            child.upper = dict(self.upper)

            if child is self.left_child:
                prev_low = child.lower.get(f, -1 * np.inf)
                child.lower[f] = max(prev_low, t)
            else:
                prev_up = child.upper.get(f, np.inf)
                child.upper[f] = min(prev_up, t)

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def update_indicator(self):
        """Build indicator function from bounds."""

        def is_large_enough(x):
            if not self.lower:
                return np.ones(x.shape[0], dtype=bool)
            checks = np.array([
                np.greater(x[:, key], self.lower[key])
                for key in self.lower
            ])
            return np.all(checks, axis=0)

        def is_small_enough(x):
            if not self.upper:
                return np.ones(x.shape[0], dtype=bool)
            checks = np.array([
                np.less_equal(x[:, key], self.upper[key])
                for key in self.upper
            ])
            return np.all(checks, axis=0)

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]),
            axis=0
        )

    def pred(self, x):
        """Recursive prediction for one individual."""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        return self.right_child.pred(x)


class Leaf:
    """Leaf node."""

    def __init__(self, value, depth=0):
        """Initialize a Leaf."""
        self.value = value
        self.depth = depth
        self.is_leaf = True

        self.lower = {}
        self.upper = {}
        self.indicator = None
        self.sub_population = None
        self.subpopulation = None

    def get_leaves_below(self):
        """Return itself."""
        return [self]

    def update_bounds_below(self):
        """Leaf: stop recursion."""
        pass

    def update_indicator(self):
        """Leaf uses same indicator logic as Node."""
        Node.update_indicator(self)

    def pred(self, x):
        """Return leaf value."""
        return self.value


class Decision_Tree:
    """Trainable decision tree."""

    def __init__(
        self,
        split_criterion="random",
        max_depth=10,
        min_pop=1,
        seed=0,
        root=None
    ):
        """Initialize Decision_Tree."""
        self.split_criterion = split_criterion
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.rng = np.random.default_rng(seed)

        self.root = root if root is not None else Node(is_root=True, depth=0)

        self.explanatory = None
        self.target = None
        self.predict = None

    def np_extrema(self, arr):
        """Return min and max of arr."""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Pick a random (feature, threshold) valid for node population."""
        diff = 0.0
        while diff == 0.0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            vals = self.explanatory[:, feature][node.sub_population]
            feature_min, feature_max = self.np_extrema(vals)
            diff = feature_max - feature_min

        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return int(feature), float(threshold)

    def possible_thresholds(self, node, feature):
        """Return midpoints between sorted unique values in node for feature."""
        values = np.unique(self.explanatory[:, feature][node.sub_population])
        return (values[1:] + values[:-1]) / 2

    def Gini_split_criterion_one_feature(self, node, feature):
        """
        Return (best_threshold, best_gini_avg) for one feature.

        No for/while loops allowed.
        """
        y = self.target[node.sub_population].astype(int)
        x = self.explanatory[:, feature][node.sub_population]

        thresholds = self.possible_thresholds(node, feature)
        if thresholds.size == 0:
            return np.array([0.0, np.inf])

        classes = np.unique(y)
        one_hot = (y[:, None] == classes[None, :])

        left_mask = x[:, None] > thresholds[None, :]
        right_mask = np.logical_not(left_mask)

        left_f = left_mask[:, :, None] & one_hot[:, None, :]
        right_f = right_mask[:, :, None] & one_hot[:, None, :]

        left_counts = np.sum(left_f, axis=0).astype(float)
        right_counts = np.sum(right_f, axis=0).astype(float)

        left_n = np.sum(left_counts, axis=1)
        right_n = np.sum(right_counts, axis=1)
        n_total = float(y.size)

        left_n_safe = np.where(left_n == 0.0, 1.0, left_n)
        right_n_safe = np.where(right_n == 0.0, 1.0, right_n)

        left_probs = left_counts / left_n_safe[:, None]
        right_probs = right_counts / right_n_safe[:, None]

        gini_left = 1.0 - np.sum(left_probs ** 2, axis=1)
        gini_right = 1.0 - np.sum(right_probs ** 2, axis=1)

        gini_avg = (left_n / n_total) * gini_left + (right_n / n_total) * gini_right

        j = int(np.argmin(gini_avg))
        return np.array([float(thresholds[j]), float(gini_avg[j])])

    def Gini_split_criterion(self, node):
        """Return (best_feature, best_threshold) for node using Gini."""
        x = np.array([
            self.Gini_split_criterion_one_feature(node, i)
            for i in range(self.explanatory.shape[1])
        ])
        i = int(np.argmin(x[:, 1]))
        return i, float(x[i, 0])

    def _is_leaf_condition(self, sub_population, depth):
        """Leaf condition: min_pop, max_depth, or pure class."""
        pop_size = int(np.sum(sub_population))
        if pop_size < self.min_pop:
            return True
        if depth >= self.max_depth:
            return True
        y = self.target[sub_population]
        if y.size == 0:
            return True
        return np.unique(y).size == 1

    def get_leaf_child(self, node, sub_population):
        """Create a Leaf child for this sub_population."""
        y = self.target[sub_population].astype(int)
        if y.size == 0:
            value = 0
        else:
            counts = np.bincount(y)
            value = int(np.argmax(counts))

        leaf_child = Leaf(value=value)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Create a Node child for this sub_population."""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        """Recursively fit the node and its children."""
        node.feature, node.threshold = self.split_criterion(node)

        go_left = self.explanatory[:, node.feature] > node.threshold
        left_population = np.logical_and(node.sub_population, go_left)
        right_population = np.logical_and(node.sub_population, ~go_left)

        is_left_leaf = self._is_leaf_condition(left_population, node.depth + 1)
        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        is_right_leaf = self._is_leaf_condition(right_population, node.depth + 1)
        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, target, verbose=0):
        """Train decision tree."""
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion

        self.explanatory = explanatory
        self.target = target

        self.root.sub_population = np.ones_like(self.target, dtype="bool")
        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(
                "  Training finished.\n"
                f"- Depth                     : {self.depth()}\n"
                f"- Number of nodes           : {self.count_nodes()}\n"
                f"- Number of leaves          : "
                f"{self.count_nodes(only_leaves=True)}\n"
                f"- Accuracy on training data : "
                f"{self.accuracy(self.explanatory, self.target)}"
            )

    def update_bounds(self):
        """Compute bounds for the whole tree."""
        self.root.update_bounds_below()

    def get_leaves(self):
        """Return all leaves."""
        return self.root.get_leaves_below()

    def update_predict(self):
        """Build efficient batch prediction function."""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        def _predict(a):
            preds = np.zeros(a.shape[0], dtype=int)
            for leaf in leaves:
                m = leaf.indicator(a)
                preds[m] = leaf.value
            return preds

        self.predict = _predict

    def pred(self, x):
        """Recursive prediction for one individual."""
        return self.root.pred(x)

    def accuracy(self, test_explanatory, test_target):
        """Return accuracy score."""
        return (
            np.sum(np.equal(self.predict(test_explanatory), test_target))
            / test_target.size
        )

    def count_nodes(self, only_leaves=False):
        """Count nodes (optionally only leaves)."""

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

    def depth(self):
        """Return maximum depth."""
        def rec(n):
            if getattr(n, "is_leaf", False):
                return n.depth
            return max(rec(n.left_child), rec(n.right_child))
        return rec(self.root)
