#!/usr/bin/env python3
"""
4-build_decision_tree.py

Build a simple decision tree structure and propagate feature bounds
(lower/upper dictionaries) down the tree.
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

    def get_leaves_below(self):
        """Return all leaves below this node."""
        leaves = []
        for child in [self.left_child, self.right_child]:
            leaves.extend(child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        """
        Recursively compute bounds for each node.

        Convention (as required by the checker examples):
        - If x[feature] > threshold  -> goes to left child
        - else                      -> goes to right child

        So:
        - left child gets lower[feature] = threshold
        - right child gets upper[feature] = threshold
        """
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1 * np.inf}

        for child in [self.left_child, self.right_child]:
            child.lower = self.lower.copy()
            child.upper = self.upper.copy()

            if child is self.left_child:
                child.lower[self.feature] = self.threshold
            else:
                child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()


class Leaf:
    """Decision tree leaf."""

    def __init__(self, value, depth=0):
        """Initialize a Leaf."""
        self.value = value
        self.depth = depth
        self.is_leaf = True
        self.lower = {}
        self.upper = {}

    def get_leaves_below(self):
        """Return itself as the only leaf below."""
        return [self]

    def update_bounds_below(self):
        """Leaf: nothing to propagate."""
        return


class Decision_Tree:
    """Decision tree container."""

    def __init__(self, root=None):
        """Initialize Decision_Tree."""
        self.root = root

    def get_leaves(self):
        """Return all leaves of the tree."""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Propagate bounds through the tree."""
        self.root.update_bounds_below()
