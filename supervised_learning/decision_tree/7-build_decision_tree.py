#!/usr/bin/env python3
"""
7-build_decision_tree.py

Trainable decision tree with random split criterion.
Includes: printing, bounds propagation, indicator functions, predict and fit.
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
        """Initialize a node."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.depth = depth
        self.is_root = is_root

        self.is_leaf = False
        self.lower = {}
        self.upper = {}
        self.sub_population = None
        self.indicator = None

    def left_child_add_prefix(self, text):
        """Prefix helper for left child printing."""
        lines = text.split("\n")
        out = "    +--" + lines[0]
        for line in lines[1:]:
            out += "\n    |  " + line
        return out

    def right_child_add_prefix(self, text):
        """Prefix helper for right child printing."""
        lines = text.split("\n")
        out = "    +--" + lines[0]
        for line in lines[1:]:
            out += "\n       " + line
        return out

    def __str__(self):
        """Print subtree rooted at this node."""
        name = "root" if self.is_root else "-> node"
        head = f"{name} [feature={self.feature}, threshold={self.threshold}]"
        if self.left_child is None or self.right_child is None:
            return head
        left_txt = self.left_child_add_prefix(str(self.left_child))
        right_txt = self.right_child_add_prefix(str(self.right_child))
        return head + "\n" + left_txt + "\n" + right_txt

    def get_leaves_below(self):
        """Return list of leaves below this node."""
        leaves = []
        for child in [self.left_child, self.right_child]:
            leaves.extend(child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        """
        Propagate lower/upper bounds down the subtree.

        Routing convention:
        - if x[feature] > threshold  -> left child
        - else                      -> right child

        Bounds update:
        - left child: lower[feature] = threshold
        - right child: upper[feature] = threshold
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

    def update_indicator(self):
        """Compute and store the indicator function for this node."""
        lower_keys = list(self.lower.keys())
        upper_keys = list(self.upper.keys())

        def is_large_enough(x):
            if not lower_keys:
                return np.ones(x.shape[0], dtype=bool)
            arr = np.array(
                [np.greater(x[:, k], self.lower[k]) for k in lower_keys]
            )
            return np.all(arr, axis=0)

        def is_small_enough(x):
            if not upper_keys:
                return np.ones(x.shape[0], dtype=bool)
            arr = np.array(
                [np.less_equal(x[:, k], self.upper[k]) for k in upper_keys]
            )
            return np.all(arr, axis=0)

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]),
            axis=0
        )

    def pred(self, x):
        """Slow recursive prediction for a single individual x."""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        return self.right_child.pred(x)


class Leaf:
    """Decision tree leaf."""

    def __init__(self, value, depth=0):
        """Initialize a leaf."""
        self.value = value
        self.depth = depth
        self.is_leaf = True

        self.lower = {}
        self.upper = {}
        self.sub_population = None
        self.indicator = None

    def __str__(self):
        """Leaf printing."""
        return f"-> leaf [value={self.value}]"

    def get_leaves_below(self):
        """Return itself."""
        return [self]

    def update_bounds_below(self):
        """Leaf: nothing to propagate."""
        return

    def update_indicator(self):
        """Compute indicator using bounds (same logic as Node)."""
        lower_keys = list(self.lower.keys())
        upper_keys = list(self.upper.keys())

        def is_large_enough(x):
            if not lower_keys:
                return np.ones(x.shape[0], dtype=bool)
            arr = np.array(
                [np.greater(x[:, k], self.lower[k]) for k in lower_keys]
            )
            return np.all(arr, axis=0)

        def is_small_enough(x):
            if not upper_keys:
                return np.ones(x.shape[0], dtype=bool)
            arr = np.array(
                [np.less_equal(x[:, k], self.upper[k]) for k in upper_keys]
            )
            return np.all(arr, axis=0)

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]),
            axis=0
        )

    def pred(self, x):
        """Return the stored class value."""
        return self.value


class Decision_Tree:
    """Decision tree class."""

    def __init__(
        self,
        split_criterion="random",
        max_depth=10,
        min_pop=1,
        seed=0,
        root=None
    ):
        """Initialize tree."""
        self.rng = np.random.default_rng(seed)
        self.split_criterion = split_criterion
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.root = root if root is not None else Node(is_root=True, depth=0)

        self.explanatory = None
        self.target = None
        self.predict = None

    def __str__(self):
        """Print the tree."""
        return self.root.__str__()

    def depth(self):
        """Return tree depth (max leaf depth)."""
        leaves = self.get_leaves()
        if not leaves:
            return 0
        return max(leaf.depth for leaf in leaves)

    def count_nodes(self, only_leaves=False):
        """Count nodes in the tree."""
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
                if n.left_child is not None:
                    stack.append(n.left_child)
                if n.right_child is not None:
                    stack.append(n.right_child)
        return count

    def get_leaves(self):
        """Return all leaves of the tree."""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Update bounds on all nodes/leaves."""
        self.root.update_bounds_below()

    def update_predict(self):
        """Build an efficient vectorized predict(A) using leaf indicators."""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        values = np.array([leaf.value for leaf in leaves])

        def _predict(a):
            ind = np.array([leaf.indicator(a) for leaf in leaves])
            idx = np.argmax(ind, axis=0)
            return values[idx]

        self.predict = _predict

    def np_extrema(self, arr):
        """Return min and max of a numpy array."""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Pick a random feature and random threshold in its range."""
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            vals = self.explanatory[:, feature][node.sub_population]
            feature_min, feature_max = self.np_extrema(vals)
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        """Create a leaf child with the majority class on sub_population."""
        y = self.target[sub_population]
        classes, counts = np.unique(y, return_counts=True)
        value = classes[np.argmax(counts)]

        leaf_child = Leaf(value=value)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Create an internal node child."""
        n = Node(depth=node.depth + 1)
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        """Recursively split the node and grow the tree."""
        node.feature, node.threshold = self.split_criterion(node)

        go_left = self.explanatory[:, node.feature] > node.threshold
        left_population = node.sub_population & go_left
        right_population = node.sub_population & (~go_left)

        left_size = int(np.sum(left_population))
        right_size = int(np.sum(right_population))

        left_pure = np.unique(self.target[left_population]).size <= 1
        right_pure = np.unique(self.target[right_population]).size <= 1

        next_depth = node.depth + 1
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
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, target, verbose=0):
        """Train the tree on (explanatory, target)."""
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
    acc = self.accuracy(
        self.explanatory,
        self.target
    )
    print(
        f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}
    - Accuracy on training data : {acc}"""
    )

    def accuracy(self, test_explanatory, test_target):
        """Return classification accuracy on a test set."""
        pred = self.predict(test_explanatory)
        return np.sum(np.equal(pred, test_target)) / test_target.size

    def pred(self, x):
        """Slow recursive prediction for one individual."""
        return self.root.pred(x)

    def Gini_split_criterion(self, node):
        """Placeholder for later tasks (not used in random-split task)."""
        raise NotImplementedError("Gini criterion is defined in a later task.")
