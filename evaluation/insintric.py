from sklearn import metrics

def evaluateClusters(X, labels):
    #Silhouette Coefficient: -1 incorrect clustering and +1 highly dense clustering
    siScore = metrics.silhouette_score(X, labels, metric='precomputed')

    #Calinski-Harabasz Index: High score --> dense and well separated clusters
    chScore = metrics.calinski_harabasz_score(X, labels)

    #Davies-Bouldin Index: 0 best value --> indicates partition
    dbScore = metrics.davies_bouldin_score(X, labels)

    return siScore, chScore, dbScore