#!/usr/bin/env python3
"""K-means clustering using scikit-learn."""

import sklearn.cluster


def kmeans(X, k):
    """
    Perform K-means clustering on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: int, number of clusters

    Returns:
        C: numpy.ndarray of shape (k, d) containing the centroid means
        clss: numpy.ndarray of shape (n,) containing the cluster index
              for each data point
    """
    model = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    C = model.cluster_centers_
    clss = model.labels_
    return C, clss

