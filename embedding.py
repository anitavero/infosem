from gensim.models import Word2Vec
import argh
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
import re
from prettytable import PrettyTable
from tqdm import tqdm
from itertools import chain
import os

import text_process as tp
import util
from util import roundl


def train(corpus, data_type, lang, save_path,
          size=100, window=5, min_count=100, workers=4,
          epochs=5, max_vocab_size=None, pretraining_corpus=None):
    """
    Train w2v.
    :param data_path: str, json file path
    :param save_path: Model file path
    :return: trained model
    """
    # print("Get sents...")
    # texts = tp.get_sents(data, data_type=data_type, lang=lang)
    texts = tp.text2gensim(corpus, lang)

    if not os.path.exists(save_path):
        model = Word2Vec(texts, size=size, window=window, min_count=min_count, workers=workers,
                         max_vocab_size=max_vocab_size)
    else:
        model = Word2Vec.load(save_path)

    if pretraining_corpus:
        model.build_vocab(pretraining_corpus, update=True)
    # print("train...")
    model.train(texts, total_examples=len(list(texts)), epochs=epochs)

    model.save(save_path)

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


def avg_speed_through_time(Vt):
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


def avg_pairwise_distances_through_time(Vt):
    """Average pairwise distances at every time step t."""
    return map(avg_pairwise_distances, [Vt[:, :, t] for t in range(Vt.shape[2])])


#######################################################


def order_through_time(corpus_list, save_path, data_type='article', lang='hungarian',
         size=100, window=5, min_count=100, workers=4, epochs=5, max_vocab_size=None,
         n_neighbors=5):
    """
    Train Word2Vec on a series of corpora and evaluate order metrics after each training.
    :param corpus_list: str list
    :return: metrics
    """
    vocabs =list()
    for t, corpus in enumerate(tqdm(corpus_list)):
        model = train(corpus, data_type, lang, save_path, size, window, min_count,
                      workers, epochs, max_vocab_size,
                      pretraining_corpus=list(set(chain(list(corpus_list)[:t]))))
        vocabs.append(model.wv.vocab)
        if t == 0:
            Vt = np.empty((model.wv.vectors.shape[0], size, len(corpus_list)))
        Vt_prev = Vt.copy()
        Vt = np.empty((model.wv.vectors.shape[0], size, len(corpus_list)))
        # Add new words to all the models in previous time steps with full zero embeddings
        for tp in range(0, t):
            Vtp = Vt_prev[:, :, tp]
            # Make sure the embeddings belong to the same word indices in each matrix.
            # Invariant: vocab[t-1] is element of vocab[t] for each t=[1...n] because
            # we always train the model further from the previous one.
            for w in vocabs[t]:
                if w in vocabs[tp]:     # Keep vector from the tp time step
                    Vt[vocabs[t][w].index, :, tp] = Vtp[vocabs[tp][w].index]
                else:                   # Add new words with full zero embeddings
                    Vt[vocabs[t][w].index, :, tp] = np.zeros(size)
        Vt[:, :, t] = model.wv.vectors

    order_locals = order_local(Vt, n_neighbors, metric='l2')
    avg_speeds = avg_speed_through_time(Vt)
    avg_pw_dists = avg_pairwise_distances_through_time(Vt)

    return order_locals, avg_speeds, avg_pw_dists, vocabs


def main(data_path, save_path, data_type='article', lang='hungarian',
         size=100, window=5, min_count=1, workers=4, epochs=5, max_vocab_size=None,
         n_neighbors=5):
    data = util.read_jl(data_path)
    data.sort(key=lambda x: x['date'])
    news_per_month = tp.data_per_month(data, data_type=data_type, concat=True)
    order_locals, avg_speeds, avg_pw_dists, vocabs = \
        order_through_time(news_per_month.values(), save_path, data_type=data_type, lang=lang,
         size=size, window=window, min_count=min_count, workers=workers, epochs=epochs,
         max_vocab_size=max_vocab_size, n_neighbors=n_neighbors)
    print("Local order parameters:", roundl(order_locals))
    print("Average speeds:", roundl(avg_speeds))
    print("Average pairwise distances:", roundl(avg_pw_dists))
    print("Vocab sizes:", [len(v) for v in vocabs])


if __name__ == '__main__':
    argh.dispatch_command(main)