#!/usr/bin/env python3
"""
11-isolation_forest.py

Isolation Random Forest:
- Build many Isolation_Random_Trees
- Predict returns mean leaf depth across trees
- suspects returns rows with smallest mean depth
"""

import numpy as np

Isolation_Random_Tree = __import__("10-isolation_tree").Isolation_Random_Tree


class Isolation_Random_Forest:
    """Isolation Random Forest."""

    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """Initialize Isolation_Random_Forest."""
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed
        self.explanatory = None

    def predict(self, explanatory):
        """Return mean depth across trees for each row."""
        predictions = np.array([f(explanatory) for f in self.numpy_preds])
        return predictions.mean(axis=0)

    def fit(self, explanatory, n_trees=100, verbose=0):
        """Fit the isolation forest on explanatory."""
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []

        for i in range(n_trees):
            t = Isolation_Random_Tree(
                max_depth=self.max_depth,
                seed=self.seed + i
            )
            t.fit(explanatory)
            self.numpy_preds.append(t.predict)
            depths.append(t.depth())
            nodes.append(t.count_nodes())
            leaves.append(t.count_nodes(only_leaves=True))

        if verbose == 1:
            print(
                "  Training finished.\n"
                f"    - Mean depth                     : "
                f"{np.array(depths).mean()}\n"
                f"    - Mean number of nodes           : "
                f"{np.array(nodes).mean()}\n"
                f"    - Mean number of leaves          : "
                f"{np.array(leaves).mean()}"
            )

    def suspects(self, explanatory, n_suspects):
        """
        Return (suspects, depths):
        - suspects: the n_suspects rows with smallest mean depth
        - depths: their corresponding mean depths
        """
        depths = self.predict(explanatory)
        idx = np.argsort(depths)[:n_suspects]
        return explanatory[idx, :], depths[idx]
