#!/usr/bin/env python3
"""Module that performs agglomerative clustering on a dataset."""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """Performs agglomerative clustering on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset.
        dist: the maximum cophenetic distance for all clusters.

    Returns:
        clss: numpy.ndarray of shape (n,) containing the cluster indices
            for each data point.
    """
    linkage = scipy.cluster.hierarchy.linkage(X, method='ward')
    clss = scipy.cluster.hierarchy.fcluster(
        linkage, t=dist, criterion='distance'
    )

    plt.figure()
    scipy.cluster.hierarchy.dendrogram(linkage, color_threshold=dist)
    plt.show()

    return clss
