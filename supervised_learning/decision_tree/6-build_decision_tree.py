#!/usr/bin/env python3
"""
6-build_decision_tree.py

Decision tree structure utilities:
- Node / Leaf representation
- Leaves retrieval
- Bounds computation
- Indicator functions
- Efficient batch prediction (predict) vs recursive single prediction (pred)
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

        Convention (matches expected outputs):
        - left_child  -> >= threshold  (update lower)
        - right_child -> <= threshold  (update upper)

        Only constrained features should appear as keys.
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
                prev_low = child.lower.get(feature, -1 * np.inf)
                child.lower[feature] = max(prev_low, threshold)
            else:
                prev_up = child.upper.get(feature, np.inf)
                child.upper[feature] = min(prev_up, threshold)

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def update_indicator(self):
        """
        Build indicator function from lower and upper bounds.

        For array A (n_individuals, n_features), indicator(A) returns a boolean
        vector (n_individuals,) that is True iff:
          - for all keys in lower: A[:, key] > lower[key]
          - for all keys in upper: A[:, key] <= upper[key]
        """

        def is_large_enough(x):
            """True if all features are strictly greater than lower bounds."""
            if not self.lower:
                return np.ones(x.shape[0], dtype=bool)

            checks = np.array([
                np.greater(x[:, key], self.lower[key])
                for key in self.lower
            ])
            return np.all(checks, axis=0)

        def is_small_enough(x):
            """True if all features are <= upper bounds."""
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
        """
        Recursive prediction for a single individual (1D array x).
        """
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        return self.right_child.pred(x)


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
        """Leaf uses same indicator logic as Node."""
        Node.update_indicator(self)

    def pred(self, x):
        """
        Return the leaf value for a single individual.
        """
        return self.value


class Decision_Tree:
    """Decision tree container."""

    def __init__(self, root=None):
        """Initialize Decision_Tree."""
        self.root = root
        self.predict = None

    def get_leaves(self):
        """Return all leaves."""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Compute bounds for whole tree."""
        self.root.update_bounds_below()

    def pred(self, x):
        """
        Recursive prediction for a single individual.
        """
        return self.root.pred(x)

    def update_predict(self):
        """
        Compute efficient batch prediction function (vectorized over A).

        It uses leaf indicators: exactly one leaf indicator is True per row.
        """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        def _predict(A):
            preds = np.zeros(A.shape[0], dtype=int)
            for leaf in leaves:
                mask = leaf.indicator(A)
                preds[mask] = leaf.value
            return preds

        self.predict = _predict
