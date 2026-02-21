#!/usr/bin/env python3
"""
8-build_decision_tree.py

Trainable decision tree with:
- Random split criterion
- Gini impurity split criterion
- Fast vectorized predict based on leaf indicators
"""

import numpy as np


class Node:
    """Decision tree internal node."""

    def __init__(
        self,
        feature=None,
        threshold=None,
        left_child=None,
        right_child=None,
        depth=0,
        is_root=False,
    ):
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
        """Return list of leaves in the subtree."""
        leaves = []
        for child in (self.left_child, self.right_child):
            leaves.extend(child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        """
        Recursively compute lower/upper bound dictionaries for children.
        Convention:
        - Going to left child means feature value > threshold
          => update child's lower bound.
        - Going to right child means feature value <= threshold
          => update child's upper bound.
        """
        if self.is_root and (self.lower is None or self.upper is None):
            self.lower = {}
            self.upper = {}

        for child, is_left in (
            (self.left_child, True),
            (self.right_child, False),
        ):
            child.lower = self.lower.copy()
            child.upper = self.upper.copy()

            f = self.feature
            t = self.threshold
            if f is not None:
                if is_left:
                    child.lower[f] = t
                else:
                    child.upper[f] = t

        for child in (self.left_child, self.right_child):
            child.update_bounds_below()

    def update_indicator(self):
        """
        Build indicator(x) that returns a boolean mask of rows satisfying
        the bounds stored in self.lower / self.upper.
        """

        def is_large_enough(x):
            if not self.lower:
                return np.ones(x.shape[0], dtype=bool)
            keys = list(self.lower.keys())
            checks = np.array(
                [np.greater(x[:, k], self.lower[k]) for k in keys],
                dtype=bool,
            )
            return np.all(checks, axis=0)

        def is_small_enough(x):
            if not self.upper:
                return np.ones(x.shape[0], dtype=bool)
            keys = list(self.upper.keys())
            checks = np.array(
                [np.less_equal(x[:, k], self.upper[k]) for k in keys],
                dtype=bool,
            )
            return np.all(checks, axis=0)

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]),
            axis=0,
        )

    def pred(self, x):
        """Recursive prediction for one sample x."""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        return self.right_child.pred(x)


class Leaf:
    """Decision tree leaf."""

    def __init__(self, value, depth=0):
        self.value = value
        self.depth = depth
        self.is_leaf = True

        self.lower = {}
        self.upper = {}
        self.indicator = None
        self.sub_population = None

    def get_leaves_below(self):
        """Return itself as a leaf list."""
        return [self]

    def update_bounds_below(self):
        """Leaf: nothing to propagate."""
        return

    def update_indicator(self):
        """Same indicator construction as Node, using leaf bounds."""

        def is_large_enough(x):
            if not self.lower:
                return np.ones(x.shape[0], dtype=bool)
            keys = list(self.lower.keys())
            checks = np.array(
                [np.greater(x[:, k], self.lower[k]) for k in keys],
                dtype=bool,
            )
            return np.all(checks, axis=0)

        def is_small_enough(x):
            if not self.upper:
                return np.ones(x.shape[0], dtype=bool)
            keys = list(self.upper.keys())
            checks = np.array(
                [np.less_equal(x[:, k], self.upper[k]) for k in keys],
                dtype=bool,
            )
            return np.all(checks, axis=0)

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]),
            axis=0,
        )

    def pred(self, x):
        """Return leaf value for one sample x."""
        return self.value


class Decision_Tree:
    """Decision tree."""

    def __init__(
        self,
        split_criterion="random",
        max_depth=10,
        min_pop=1,
        seed=0,
        root=None,
    ):
        self.rng = np.random.default_rng(seed)
        self.split_criterion = split_criterion
        self.max_depth = max_depth
        self.min_pop = min_pop

        self.root = root if root is not None else Node(is_root=True, depth=0)
        self.explanatory = None
        self.target = None
        self.predict = None

    def get_leaves(self):
        """Return tree leaves."""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Compute bounds for the whole tree."""
        if self.root.lower is None:
            self.root.lower = {}
        if self.root.upper is None:
            self.root.upper = {}
        self.root.update_bounds_below()

    def update_predict(self):
        """
        Vectorized prediction:
        - update bounds
        - build leaf indicators
        - prediction is sum(value * indicator)
          (exactly one leaf should match each sample).
        """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        def _predict(a):
            parts = [leaf.value * leaf.indicator(a) for leaf in leaves]
            return np.sum(np.array(parts), axis=0)

        self.predict = _predict

    def pred(self, x):
        """Recursive prediction through root."""
        return self.root.pred(x)

    def depth(self):
        """Tree depth (max leaf depth)."""
        leaves = self.get_leaves()
        return max(leaf.depth for leaf in leaves) if leaves else 0

    def count_nodes(self, only_leaves=False):
        """Count nodes (optionally only leaves)."""
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

    def np_extrema(self, arr):
        """Return min/max of array."""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Pick a random (feature, threshold) with non-zero range."""
        diff = 0.0
        while diff == 0.0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            vals = self.explanatory[:, feature][node.sub_population]
            fmin, fmax = self.np_extrema(vals)
            diff = fmax - fmin
        x = self.rng.uniform()
        threshold = (1.0 - x) * fmin + x * fmax
        return feature, threshold

    def possible_thresholds(self, node, feature):
        """Midpoints between sorted unique values."""
        values = np.unique(self.explanatory[:, feature][node.sub_population])
        return (values[1:] + values[:-1]) / 2.0

    def Gini_split_criterion_one_feature(self, node, feature):
        """
        Return (best_threshold, best_score) for one feature.
        No for/while loops here (pure NumPy broadcasting).
        """
        mask = node.sub_population
        x = self.explanatory[:, feature][mask]
        y = self.target[mask]
        n_total = y.size

        thresholds = self.possible_thresholds(node, feature)
        if thresholds.size == 0:
            return 0.0, np.inf

        classes = np.unique(y)
        y_onehot = (y[:, None] == classes[None, :])  # (n, c)

        left_mask = x[:, None] > thresholds[None, :]  # (n, t)

        left_f = y_onehot[:, :, None] & left_mask[:, None, :]  # (n, c, t)
        left_counts = np.sum(left_f, axis=0).astype(float)  # (c, t)

        total_counts = np.sum(y_onehot, axis=0).astype(float)  # (c,)
        right_counts = total_counts[:, None] - left_counts  # (c, t)

        left_tot = np.sum(left_counts, axis=0)  # (t,)
        right_tot = n_total - left_tot  # (t,)

        left_tot_safe = np.where(left_tot == 0.0, 1.0, left_tot)
        right_tot_safe = np.where(right_tot == 0.0, 1.0, right_tot)

        left_p = left_counts / left_tot_safe[None, :]
        right_p = right_counts / right_tot_safe[None, :]

        gini_left = 1.0 - np.sum(left_p ** 2, axis=0)
        gini_right = 1.0 - np.sum(right_p ** 2, axis=0)

        w_left = left_tot / float(n_total)
        w_right = right_tot / float(n_total)

        score = w_left * gini_left + w_right * gini_right

        i = int(np.argmin(score))
        return float(thresholds[i]), float(score[i])

    def Gini_split_criterion(self, node):
        """Choose best feature by minimal Gini score."""
        x = np.array(
            [
                self.Gini_split_criterion_one_feature(node, i)
                for i in range(self.explanatory.shape[1])
            ],
            dtype=float,
        )
        i = int(np.argmin(x[:, 1]))
        return i, x[i, 0]

    def get_leaf_child(self, node, sub_population):
        """Create leaf child with majority class in sub_population."""
        y = self.target[sub_population]
        classes, counts = np.unique(y, return_counts=True)
        value = classes[int(np.argmax(counts))]
        leaf = Leaf(value=value, depth=node.depth + 1)
        leaf.sub_population = sub_population
        return leaf

    def get_node_child(self, node, sub_population):
        """Create internal node child."""
        n = Node(depth=node.depth + 1)
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        """Recursive training (no loop on individuals)."""
        node.feature, node.threshold = self.split_criterion(node)

        feat_vals = self.explanatory[:, node.feature]
        go_left = feat_vals > node.threshold

        left_pop = node.sub_population & go_left
        right_pop = node.sub_population & (~go_left)

        left_size = int(np.sum(left_pop))
        right_size = int(np.sum(right_pop))

        next_depth = node.depth + 1

        left_targets = self.target[left_pop]
        right_targets = self.target[right_pop]

        left_pure = np.unique(left_targets).size <= 1
        right_pure = np.unique(right_targets).size <= 1

        is_left_leaf = (
            left_size < self.min_pop
            or next_depth == self.max_depth
            or left_pure
        )
        is_right_leaf = (
            right_size < self.min_pop
            or next_depth == self.max_depth
            or right_pure
        )

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_pop)
        else:
            node.left_child = self.get_node_child(node, left_pop)
            self.fit_node(node.left_child)

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_pop)
        else:
            node.right_child = self.get_node_child(node, right_pop)
            self.fit_node(node.right_child)

    def accuracy(self, test_explanatory, test_target):
        """Return accuracy."""
        return (
            np.sum(np.equal(self.predict(test_explanatory), test_target))
            / test_target.size
        )

    def fit(self, explanatory, target, verbose=0):
        """Train the tree."""
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion

        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype=bool)

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(
                f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}
    - Accuracy on training data : {self.accuracy(self.explanatory, self.target)}"""
            )
