from gensim.models import Word2Vec
import argh
from sklearn.cluster import DBSCAN, KMeans
from sklearn import metrics

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
    """Unsupervised metrics for a clustering."""
    try:
        print("Silhouette Coefficient: %0.3f"
              % metrics.silhouette_score(vectors, labels))
    except ValueError as e:
        print("[Silhouette Coefficient] " + str(e))


cluster_methods = {'dbscan': dbscan_clustering, 'kmeans': kmeans}


def run_clustering(model, cluster_method, **kwargs):
    labels = cluster_methods[cluster_method](model, **kwargs)
    cluster_eval(model.wv.vectors, labels)


def main(data_path, save_path, data_type='article', lang='hungarian',
         size=100, window=5, min_count=100, workers=4, epochs=5, max_vocab_size=None):
    data = util.read_jl(data_path)
    return train(data, data_type, lang, save_path, size, window, min_count, workers, epochs,
                 max_vocab_size)


if __name__ == '__main__':
    argh.dispatch_command(main)