from gensim.models import Word2Vec
import argh
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
import re
from prettytable import PrettyTable

import text_process as tp
import util


def train(data, data_type, lang, save_path,
         size=100, window=5, min_count=100, workers=4,
         epochs=5, max_vocab_size=None):
    """
    Train w2v.
    :param data_path: str, json file path
    :param save_path: Model file path
    :return: trained model
    """
    print("Get sents...")
    texts = tp.get_sents(data, data_type=data_type, lang=lang)

    model = Word2Vec(texts, size=size, window=window, min_count=min_count, workers=workers,
                     max_vocab_size=max_vocab_size)
    model.save(save_path)

    model = Word2Vec.load(save_path)
    print("train...")
    model.train(texts, total_examples=len(list(texts)), epochs=epochs)

    return model


def dbscan_clustering(model, eps=0.5, min_samples=90):
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(model.wv.vectors)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    return labels


def kmeans(model, n_clusters=3, random_state=1):
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=random_state).fit(model.wv.vectors)
    labels = kmeans_model.labels_
    return labels


def cluster_eval(vectors, labels):
    """Unsupervised clustering metrics."""
    t = PrettyTable(['Metric', 'Score'])
    def safe_metric(metric):
        name = re.sub('_', ' ', metric.__name__).title()
        try:
            t.add_row([name, round(metric(vectors, labels), 4)])
        except ValueError as e:
            print("[{0}] {1}".format(name, e))

    safe_metric(metrics.silhouette_score)
    safe_metric(metrics.calinski_harabaz_score)
    safe_metric(metrics.davies_bouldin_score)

    print(t)


cluster_methods = {'dbscan': dbscan_clustering, 'kmeans': kmeans}


def run_clustering(model, cluster_method, **kwargs):
    labels = cluster_methods[cluster_method](model, **kwargs)
    cluster_eval(model.wv.vectors, labels)


#######################################################
############ Self-organised system metrics ############
#######################################################


############## Metrics at a given time t ##############


def velocity(Vt):
    """
    Return a series of velocity vectors that each point made in a time series.
    :param Vt: vector space series, NxDxT
    :return: NxDxT-1
    """
    return Vt[:, :, 1:] - Vt[:, :, :-1]


def avg_speed_throug_time(Vt):
    """L2 norm of the velocity matrices at every time step t."""
    V = velocity(Vt)
    return map(np.linalg.norm, [V[:, :, t] for t in range(V.shape[2])])


###### Order parameters ######


def order_local(Vt, n_neighbors, metric='l2'):
    """Average velocity distance from nearest neighbors."""
    V = Vt[:, :, -1]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='ball_tree', metric=metric).fit(V)
    distances, indices = nbrs.kneighbors(V)
    Vv = velocity(Vt)

    avg_velocity_series = []

    for t in range(Vv.shape[2]):
        Vvt = Vv[:, :, t]
        avg_nb_velocity_dists = []
        for ids in indices:
            avg_nb_velocity_dists.append(np.average(np.dot(Vvt[ids[1:]], Vvt[ids[0]])))
        avg_velocity_series.append(np.average(avg_nb_velocity_dists))
    return avg_velocity_series


def avg_pairwise_distances(V):
    """Average pairwise distances."""
    return np.average(metrics.pairwise_distances(V))


def main(data_path, save_path, data_type='article', lang='hungarian',
         size=100, window=5, min_count=100, workers=4, epochs=5, max_vocab_size=None):
    data = util.read_jl(data_path)
    return train(data, data_type, lang, save_path, size, window, min_count, workers, epochs,
                 max_vocab_size)


if __name__ == '__main__':
    argh.dispatch_command(main)