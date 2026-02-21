#!/usr/bin/env python3
"""
9-random_forest.py

Random Forest based on multiple random-split Decision_Trees.
Prediction is the per-sample mode across tree predictions.
"""

Decision_Tree = __import__("8-build_decision_tree").Decision_Tree
import numpy as np


class Random_Forest:
    """Random forest classifier using random-split decision trees."""

    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """Initialize Random_Forest."""
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed
        self.explanatory = None

    def predict(self, explanatory):
        """
        Predict class labels for explanatory (2D numpy array).

        Uses the mode (most frequent prediction) across all trees.
        """
        preds = np.array([p(explanatory) for p in self.numpy_preds])
        max_class = int(np.max(preds))
        counts = np.zeros((max_class + 1, preds.shape[1]), dtype=int)
        np.add.at(counts, (preds, np.arange(preds.shape[1])), 1)
        return np.argmax(counts, axis=0)

    def fit(self, explanatory, target, n_trees=100, verbose=0):
        """Fit the random forest using n_trees random-split decision trees."""
        self.target = target
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        accuracies = []
        for i in range(n_trees):
            t = Decision_Tree(
                max_depth=self.max_depth,
                min_pop=self.min_pop,
                seed=self.seed + i
            )
            t.fit(explanatory, target)
            self.numpy_preds.append(t.predict)
            depths.append(t.depth())
            nodes.append(t.count_nodes())
            leaves.append(t.count_nodes(only_leaves=True))
            accuracies.append(t.accuracy(t.explanatory, t.target))

        if verbose == 1:
            print(
                "  Training finished.\n"
                f"    - Mean depth                     : "
                f"{np.array(depths).mean()}\n"
                f"    - Mean number of nodes           : "
                f"{np.array(nodes).mean()}\n"
                f"    - Mean number of leaves          : "
                f"{np.array(leaves).mean()}\n"
                f"    - Mean accuracy on training data : "
                f"{np.array(accuracies).mean()}\n"
                f"    - Accuracy of the forest on td   : "
                f"{self.accuracy(self.explanatory, self.target)}"
            )

    def accuracy(self, test_explanatory, test_target):
        """Return accuracy score."""
        return (
            np.sum(np.equal(self.predict(test_explanatory), test_target))
            / test_target.size
        )
