import argh
from argh import arg
from itertools import permutations
from tqdm import tqdm
import os

import embedding as emb
from util import createFolder


def nltk_permutations(plot, models):
    from nltk.corpus import brown, reuters, gutenberg, genesis, webtext
    cs_n = [('brown', brown), ('reuters', reuters), ('gutenberg', gutenberg),
            ('genesis', genesis), ('webtext', webtext)]
    cs_nc = [(n, c.raw()) for n, c in cs_n]
    cs_perm = list(permutations(cs_nc))

    ols, avs, avpd, vs = [], [], [], []
    for csp in tqdm(cs_perm, desc='NLTK permutations'):
        names, corpora = list(zip(*csp))

        folder = os.path.join('models', 'nltk_permutations', '_'.join([n[:2] for n in names]))
        createFolder(folder)
        with open(os.path.join(folder, 'corpus_list.txt'), 'w') as f:
            f.write('\n'.join([n for n in names]))
        save_path = os.path.join(folder, 'nltk_w2v.model')

        if plot:
            models = 'load'
            no_metrics_save = True
        else:
            no_metrics_save = False

        if models == 'load':
            corpora = save_path

        order_locals, avg_speeds, avg_pw_dists, vocablens = \
            emb.main(corpora, save_path=save_path,
                 lang='english', size=300, window=5, min_count=1, workers=8,
                 epochs=20, max_vocab_size=None, n_neighbors=10,
                 models=models, plot=False, std=False, no_metrics_save=no_metrics_save,
                 samplesize=10000)

        ols.append(order_locals)
        avs.append(avg_speeds)
        avpd.append(avg_pw_dists)
        vs.append(vocablens)

    if plot:
        emb.plot_sos_metrics(order_locals=ols, avg_speeds=avs,
                             avg_pw_dists=avpd, vocablens=vs)


def news(plot, models):

    print('\n########## 444 ##########\n')
    data = '444.jl'
    save_path = 'models/444'
    createFolder(save_path)
    emb.main(data, save_path=save_path,
             lang='hungarian', size=300, window=5, min_count=1, workers=8,
             epochs=20, max_vocab_size=None, n_neighbors=10,
             models=models, plot=plot, std=False, no_metrics_save=False)


    print('\n########## Origo ##########\n')
    data = 'origo.jl'
    save_path = 'models/origo'
    createFolder(save_path)
    emb.main(data, save_path=save_path,
             lang='hungarian', size=300, window=5, min_count=1, workers=8,
             epochs=20, max_vocab_size=None, n_neighbors=10,
             models=models, plot=plot, std=False, no_metrics_save=False)



@arg('--plot', action='store_true')
@arg('--models', choices=['train', 'load'])
@arg('exp', choices=['news', 'nltk'])
def main(exp, plot=False, models='train'):
    if exp == 'nltk':
        nltk_permutations(plot=plot, models=models)
    elif exp == 'news':
        news(plot=plot, models=models)


if __name__ == '__main__':
    argh.dispatch_command(main)
