#!/usr/bin/env python3
"""
4-build_decision_tree.py

Decision tree structure utilities:
- Node / Leaf representation
- Printing a tree
- Retrieving leaves
- Computing feature bounds (lower/upper) for each node/leaf
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

        # Bounds dictionaries to be computed by update_bounds_below()
        self.lower = {}
        self.upper = {}

    def __str__(self):
        """Return a readable multi-line string representation of the subtree."""
        return self._str_helper()

    def _str_helper(self):
        """Recursive helper for __str__."""
        indent = "  " * self.depth
        s = f"{indent}Node(feature={self.feature}, threshold={self.threshold})\n"
        s += f"{indent}  left:\n"
        s += self.left_child._str_helper() if hasattr(self.left_child, "_str_helper") else \
            f"{indent}    {self.left_child}\n"
        s += f"{indent}  right:\n"
        s += self.right_child._str_helper() if hasattr(self.right_child, "_str_helper") else \
            f"{indent}    {self.right_child}\n"
        return s

    def get_leaves_below(self):
        """Return the list of all leaves below this node."""
        leaves = []
        for child in [self.left_child, self.right_child]:
            leaves.extend(child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        """
        Recursively compute and attach bounds dicts (lower/upper) to descendants.

        Convention required by the project examples:
        - left_child corresponds to values >= threshold (updates LOWER bound)
        - right_child corresponds to values <= threshold (updates UPPER bound)
        """
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1 * np.inf}

        for child in [self.left_child, self.right_child]:
            # Start from parent's bounds (copy to avoid aliasing)
            child.lower = dict(self.lower)
            child.upper = dict(self.upper)

            f = self.feature
            t = self.threshold

            # Default bounds if feature not yet present
            parent_low = child.lower.get(f, -1 * np.inf)
            parent_up = child.upper.get(f, np.inf)

            if child is self.left_child:
                # left: >= threshold  -> raise lower bound
                child.lower[f] = max(parent_low, t)
                child.upper[f] = parent_up
            else:
                # right: <= threshold -> lower upper bound
                child.upper[f] = min(parent_up, t)
                child.lower[f] = parent_low

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()


class Leaf:
    """Decision tree leaf node."""

    def __init__(self, value, depth=0):
        """Initialize a Leaf."""
        self.value = value
        self.depth = depth
        self.is_leaf = True

        # Will be set by parent's update_bounds_below()
        self.lower = {}
        self.upper = {}

    def __str__(self):
        """Return a readable representation of the leaf."""
        indent = "  " * self.depth
        return f"{indent}Leaf(value={self.value})"

    def _str_helper(self):
        """Helper for Node printing recursion."""
        return f"{self}\n"

    def get_leaves_below(self):
        """Return itself as the only leaf below."""
        return [self]

    def update_bounds_below(self):
        """Leaf: nothing to propagate further."""
        pass


class Decision_Tree:
    """Decision tree container with a root node."""

    def __init__(self, root=None):
        """Initialize Decision_Tree."""
        self.root = root

    def __str__(self):
        """Return string representation of the full tree."""
        return str(self.root)

    def get_leaves(self):
        """Return all leaves of the tree."""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Compute bounds dictionaries for all nodes/leaves."""
        self.root.update_bounds_below()
