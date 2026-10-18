import numpy as np

def k_means(data, k, threshhold=2):
    """
    Does k-means clustering of the data.
    Args:
        data - a n x d numpy array, where rows are taken to be samples.
        k - integer > 1 - number of clusters
        threshhold - integer - higher number -> earlier stopping, but less accuracy
    Returns:
        A 1-D numpy array of length n, where the ith element is the
        index of the cluster.
    """
    n = np.size(data, 0)
    # random initial assignment of clusters
    cluster_centers = np.random.choice(range(0, n), k)
    clustering = np.random.randint(0, k, n)
    cluster_means = data[cluster_centers]
    old_clustering = np.zeros(n)
    # while the clustering has not converged ... 
    while np.sum(clustering != old_clustering) > threshhold:
        # print np.sum(clustering != old_clustering)
        old_clustering = clustering

        # step 1: Assign points to clusters
        cluster_distances = np.zeros((n, k))
        for cluster in range(k):
            cluster_distances[:, cluster] = np.sum(np.sqrt((data - cluster_means[cluster])**2), 1)
        clustering = np.argmin(cluster_distances, 1)

        # step 2: re-calculating means
        # cluster_means should be a k x d array
        cluster_means = np.array([np.mean(data[clustering==c],0) for c in range(k)])
    return clustering

def k_means_update(point, k, cluster_means, cluster_counts):
    """
    Does an online k-means update on a single data point.
    Args:
        point - a 1 x d array
        k - integer > 1 - number of clusters
        cluster_means - a k x d array of the means of each cluster
        cluster_counts - a 1 x k array of the number of points in each cluster
    Returns:
        An integer in [0, k-1] indicating the assigned cluster.
    Updates cluster_means and cluster_counts in place.
    For initialization, random cluster means are needed.
    """
    cluster_distances = np.zeros(k)
    for cluster in range(k):
        cluster_distances[cluster] = sum(np.sqrt((point - cluster_means[cluster])**2))
    c = np.argmin(cluster_distances)
    cluster_counts[c] += 1
    cluster_means[c] += 1.0/cluster_counts[c]*(point - cluster_means[c])
    return c