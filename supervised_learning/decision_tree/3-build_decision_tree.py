#!/usr/bin/env python3
"""Decision tree structure with leaves retrieval."""

import numpy as np


class Node:
    """Represents an internal node."""

    def __init__(
        self,
        feature=None,
        threshold=None,
        left_child=None,
        right_child=None,
        is_root=False,
        depth=0,
    ):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Return maximum depth in subtree."""
        return max(
            self.left_child.max_depth_below(),
            self.right_child.max_depth_below(),
        )

    def count_nodes_below(self, only_leaves=False):
        """Return number of nodes in subtree."""
        left_count = self.left_child.count_nodes_below(
            only_leaves=only_leaves
        )
        right_count = self.right_child.count_nodes_below(
            only_leaves=only_leaves
        )
        if only_leaves:
            return left_count + right_count
        return 1 + left_count + right_count

    def get_leaves_below(self):
        """Return list of all leaves in subtree."""
        return (
            self.left_child.get_leaves_below()
            + self.right_child.get_leaves_below()
        )

    def left_child_add_prefix(self, text):
        """Prefix subtree text for a left child."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "    |  " + x + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Prefix subtree text for a right child."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "       " + x + "\n"
        return new_text

    def __str__(self):
        """Return formatted subtree representation."""
        if self.is_root:
            header = (
                f"root [feature={self.feature}, threshold={self.threshold}]"
            )
        else:
            header = (
                f"-> node [feature={self.feature}, threshold={self.threshold}]"
            )

        left_str = str(self.left_child)
        right_str = str(self.right_child)

        out = (
            header
            + "\n"
            + self.left_child_add_prefix(left_str)
            + self.right_child_add_prefix(right_str)
        )
        return out.rstrip()


class Leaf(Node):
    """Represents a leaf node."""

    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Return leaf depth."""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Return 1 for leaf."""
        return 1

    def get_leaves_below(self):
        """Return itself in a list."""
        return [self]

    def __str__(self):
        """Return leaf string representation."""
        return f"-> leaf [value={self.value}]"


class Decision_Tree:
    """Decision tree container."""

    def __init__(
        self,
        max_depth=10,
        min_pop=1,
        seed=0,
        split_criterion="random",
        root=None,
    ):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """Return maximum tree depth."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Return number of nodes in tree."""
        return self.root.count_nodes_below(
            only_leaves=only_leaves
        )

    def get_leaves(self):
        """Return list of all leaves in tree."""
        return self.root.get_leaves_below()

    def __str__(self):
        """Return printable tree representation."""
        return self.root.__str__()
