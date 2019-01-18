from gensim.models import Word2Vec
import argh
from argh import arg
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
import re
from prettytable import PrettyTable
from tqdm import tqdm
from itertools import chain, tee
import os
from glob import glob
from matplotlib import pyplot as plt
import json

import text_process as tp
import util
from util import roundl, subfix_filename


def train(corpus, lang, save_path,
          size=100, window=5, min_count=100, workers=4,
          epochs=5, max_vocab_size=None, pretraining_corpus=None):
    """
    Train w2v.
    :param data_path: str, json file path
    :param save_path: Model file path
    :return: trained model
    """
    # print("Convert to gensim format...")
    texts = tp.text2gensim(corpus, lang)
    texts, texts_l = tee(texts)
    total_examples = len(list(texts_l))

    if not os.path.exists(save_path):
        texts, texts0 = tee(texts)
        model = Word2Vec(texts0, size=size, window=window, min_count=min_count, workers=workers,
                         max_vocab_size=max_vocab_size, compute_loss=True)
    else:
        model = Word2Vec.load(save_path)

    if pretraining_corpus:
        model.build_vocab(pretraining_corpus, update=True)
    # print("train...")
    model.train(texts, total_examples=total_examples, epochs=epochs)

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
    return list(map(np.linalg.norm, [V[:, :, t] for t in range(V.shape[2])]))


###### Order parameters ######


def order_local(Vt, n_neighbors, metric='l2'):
    """Average velocity distance from nearest neighbors."""
    V = Vt[:, :, -1]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='ball_tree', metric=metric).fit(V)
    distances, indices = nbrs.kneighbors(V)
    Vv = velocity(Vt)

    avg_velocity_series = []

    for t in tqdm(range(Vv.shape[2]), desc='Local order'):
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
    return list(map(avg_pairwise_distances, [Vt[:, :, t] for t in range(Vt.shape[2])]))


#######################################################


def order_through_time(corpus_list, save_path, lang='hungarian',
         size=100, window=5, min_count=100, workers=4, epochs=20, max_vocab_size=None,
         n_neighbors=5):
    """
    Train Word2Vec on a series of corpora and evaluate order metrics after each training.
    :param corpus_list: str list
    :return: metrics
    """
    vocabs = list()
    Vt = np.empty((0, size, 0))
    for t, corpus in enumerate(tqdm(list(corpus_list), desc='Training')):
        model = train(corpus, lang, save_path, size, window, min_count,
                      workers, epochs, max_vocab_size,
                      pretraining_corpus=list(set(chain(list(corpus_list)[:t]))))
        # save model snapshots
        model.save(subfix_filename(save_path, t))

        vocabs.append(model.wv.vocab)
        Vt = add_embedding(Vt, vocabs, model)

    return sos_eval(Vt, n_neighbors) + (vocabs,)


def add_embedding(embeddings, vocabs, new_model):
    """
    Add new embedding to an embedding series by adding words to all the models
    in previous time steps with full zero embeddings so they have the same size.
    :param embeddings: tensor of NxDxT
    :param new_model: Word2Vec model
    :return: embeddings: tensor of MxDxT+1, where M = N + |new_model.wv.vocab|
    """
    t = embeddings.shape[2]
    size = embeddings.shape[1]
    vocab_size = new_model.wv.vectors.shape[0]
    Vt_prev = embeddings.copy()
    Vt = np.empty((vocab_size, size, t + 1))
    for tp in range(0, t):
        Vtp = Vt_prev[:, :, tp]
        # Make sure the embeddings belong to the same word indices in each matrix.
        # Invariant: vocab[t-1] is element of vocab[t] for each t=[1...n] because
        # we always train the model further from the previous one.
        for w in vocabs[t]:
            if w in vocabs[tp]:  # Keep vector from the tp time step
                Vt[vocabs[t][w].index, :, tp] = Vtp[vocabs[tp][w].index]
            else:  # Add new words with full zero embeddings
                Vt[vocabs[t][w].index, :, tp] = np.zeros(size)
    Vt[:, :, t] = new_model.wv.vectors
    return Vt


def prep_nltk_corpora():
    try:
        from nltk.corpus import brown, reuters, gutenberg, genesis, inaugural, webtext, nps_chat
    except:
        import nltk
        nltk.download('brown')
        nltk.download('reuters')
        nltk.download('gutenberg')
        nltk.download('genesis')
        nltk.download('inaugural')
        nltk.download('webtext')
        nltk.download('nps_chat')
    return [c.raw() for c in [brown, reuters, gutenberg, genesis, inaugural, webtext, nps_chat]]


def sos_eval(Vt, n_neighbors):
    order_locals = order_local(Vt, n_neighbors, metric='l2')
    avg_speeds = avg_speed_through_time(Vt)
    avg_pw_dists = avg_pairwise_distances_through_time(Vt)

    return order_locals, avg_speeds, avg_pw_dists


def eval_model_series(model_name, n_neighbors):
    vocabs = list()
    model_files = glob(model_name + '_*.model')
    for i in tqdm(range(len(model_files)), desc='Loading models'):
        model = Word2Vec.load('{}_{}.model'.format(model_name,  i))
        if i == 0:
            Vt = np.empty((0, model.wv.vector_size, 0))
        vocabs.append(model.wv.vocab)
        Vt = add_embedding(Vt, vocabs, model)
    return map(list, sos_eval(Vt, n_neighbors) + (vocabs,))


def plot_sos_metrics(order_locals, avg_speeds, avg_pw_dists, vocablens):
    fig, axes = plt.subplots(2, 2)

    axes[0, 0].set_ylabel('Local order')
    axes[0, 0].grid(True)
    axes[0, 0].plot(order_locals)

    axes[1, 0].set_xlabel('Corpora number')
    axes[1, 0].set_ylabel('Avg speed')
    axes[1, 0].grid(True)
    axes[1, 0].plot(list(avg_speeds))

    axes[0, 1].set_ylabel('Avg pairwise dist')
    axes[0, 1].grid(True)
    axes[0, 1].plot(list(avg_pw_dists))

    axes[1, 1].set_xlabel('Corpora number')
    axes[1, 1].set_ylabel('Vocabulary size')
    axes[1, 1].grid(True)
    axes[1, 1].plot(vocablens)

    plt.show()


@arg('--max-vocab-size', type=int)
@arg('--models', choices=['train', 'load'])
@arg('--plot', action='store_true')
@arg('--std', action='store_true')
@arg('--no-metrics-save', action='store_true')
def main(data_source, save_path=None, data_type='article', lang='hungarian',
         size=100, window=5, min_count=1, workers=4, epochs=20, max_vocab_size=None,
         n_neighbors=10, models='train', plot=False, std=False, no_metrics_save=False):
    vocablens = []
    if models == 'train':
        if data_source == 'nltk':
            print("Prepare NLTK corpora...")
            corpora = prep_nltk_corpora()
        elif data_source == 'inaugural':
            from nltk.corpus import inaugural as ing
            corpora = [ing.raw(ing.fileids()[i]) for i in range(len(ing.fileids()))]
        elif type(data_source) == tuple:
            corpora = data_source
        else:
            data = util.read_jl(data_source)
            data.sort(key=lambda x: x['date'])
            corpora = tp.data_per_month(data, data_type=data_type, concat=True).values()

        order_locals, avg_speeds, avg_pw_dists, vocabs = \
            order_through_time(corpora, save_path, lang=lang,
             size=size, window=window, min_count=min_count, workers=workers, epochs=epochs,
             max_vocab_size=max_vocab_size, n_neighbors=n_neighbors)
    elif models == 'load':
        save_path = data_source
        if os.path.exists(os.path.join(os.path.split(save_path)[0], 'metrics.json')):
            with open(os.path.join(os.path.split(save_path)[0], 'metrics.json')) as f:
                metrics = json.load(f)
                order_locals = metrics['order_locals']
                avg_speeds = metrics['avg_speeds']
                avg_pw_dists = metrics['avg_pw_dists']
                vocablens = metrics['vocab_lens']
        else:
            order_locals, avg_speeds, avg_pw_dists, vocabs = \
                        eval_model_series(data_source, n_neighbors)

    if not vocablens:
        vocablens = [len(v) for v in vocabs]

    if plot:
        plot_sos_metrics(order_locals, avg_speeds, avg_pw_dists, vocablens)

    if std:
        print("Local order parameters:", roundl(order_locals, 5))
        print("Average speeds:", roundl(avg_speeds))
        print("Average pairwise distances:", roundl(avg_pw_dists))
        print("Vocab sizes:", vocablens)

    # Save results
    if not no_metrics_save:
        with open(os.path.join(os.path.split(save_path)[0], 'metrics.json'), 'w') as f:
            json.dump({'order_locals': order_locals,
                      'avg_speeds': avg_speeds,
                      'avg_pw_dists': avg_pw_dists,
                      'vocab_lens': vocablens}, f)

    return order_locals, avg_speeds, avg_pw_dists, vocablens


if __name__ == '__main__':
    import logging
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.ERROR)
    argh.dispatch_command(main)
