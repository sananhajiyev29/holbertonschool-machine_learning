#!/usr/bin/env python3
"""
5-build_decision_tree.py

Decision tree structure utilities:
- Node / Leaf representation
- Retrieving leaves
- Computing feature bounds
- Building indicator functions
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

    def get_leaves_below(self):
        """Return list of all leaves below this node."""
        leaves = []
        for child in [self.left_child, self.right_child]:
            leaves.extend(child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        """
        Recursively compute lower and upper bounds.

        left_child  -> >= threshold  (update lower)
        right_child -> <= threshold  (update upper)
        """
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1 * np.inf}

        feature = self.feature
        threshold = self.threshold

        for child in [self.left_child, self.right_child]:
            child.lower = dict(self.lower)
            child.upper = dict(self.upper)

            if child is self.left_child:
                previous = child.lower.get(feature, -1 * np.inf)
                child.lower[feature] = max(previous, threshold)
            else:
                previous = child.upper.get(feature, np.inf)
                child.upper[feature] = min(previous, threshold)

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def update_indicator(self):
        """
        Build indicator function from lower and upper bounds.
        """

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
            np.array([
                is_large_enough(x),
                is_small_enough(x)
            ]),
            axis=0
        )


class Leaf:
    """Decision tree leaf node."""

    def __init__(self, value, depth=0):
        """Initialize a Leaf."""
        self.value = value
        self.depth = depth
        self.is_leaf = True

        self.lower = {}
        self.upper = {}
        self.indicator = None

    def get_leaves_below(self):
        """Return itself as the only leaf."""
        return [self]

    def update_bounds_below(self):
        """Leaf: nothing to propagate."""
        pass

    def update_indicator(self):
        """Leaf uses same logic as Node."""
        Node.update_indicator(self)


class Decision_Tree:
    """Decision tree container."""

    def __init__(self, root=None):
        """Initialize Decision_Tree."""
        self.root = root

    def get_leaves(self):
        """Return all leaves."""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Compute bounds for whole tree."""
        self.root.update_bounds_below()
