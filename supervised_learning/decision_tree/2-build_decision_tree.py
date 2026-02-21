#!/usr/bin/env python3
"""
2-build_decision_tree.py
Decision tree printing: implement __str__ for Node, Leaf, Decision_Tree.
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

    def left_child_add_prefix(self, text):
        """Prefix helper for left child printing."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Prefix helper for right child printing."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
        return new_text

    def __str__(self):
        """String representation of the subtree rooted at this node."""
        name = "root" if self.is_root else "node"
        base = f"{name} [feature={self.feature}, threshold={self.threshold}]"
        if self.left_child is None or self.right_child is None:
            return base

        left_txt = self.left_child_add_prefix(str(self.left_child))
        right_txt = self.right_child_add_prefix(str(self.right_child))
        return base + "\n" + left_txt + right_txt


class Leaf:
    """Decision tree leaf."""

    def __init__(self, value, depth=0):
        """Initialize a Leaf."""
        self.value = value
        self.depth = depth
        self.is_leaf = True

    def __str__(self):
        """Leaf printing (given by the task)."""
        return f"-> leaf [value={self.value}]"


class Decision_Tree:
    """Decision tree container."""

    def __init__(self, root=None):
        """Initialize Decision_Tree."""
        self.root = root

    def __str__(self):
        """Tree printing (given by the task)."""
        return self.root.__str__()
