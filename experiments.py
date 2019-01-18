import argh
from itertools import permutations
from tqdm import tqdm
import os
import json

import embedding as emb
from util import createFolder


def nltk_permutations():
    from nltk.corpus import brown, reuters, gutenberg, genesis, webtext
    cs_n = [('brown', brown), ('reuters', reuters), ('gutenberg', gutenberg),
            ('genesis', genesis), ('webtext', webtext)]
    cs_nc = [(n, c.raw()) for n, c in cs_n]
    cs_perm = list(permutations(cs_nc))

    for csp in tqdm(cs_perm, desc='NLTK permutations'):
        names, corpora = list(zip(*csp))

        folder = os.path.join('nltk_permutations', '_'.join([n[:2] for n in names]))
        createFolder(folder)
        with open(os.path.join(folder, 'corpus_list.txt'), 'w') as f:
            f.write('\n'.join([n for n in names]))
        save_path = os.path.join(folder, 'nltk_w2v.model')

        order_locals, avg_speeds, avg_pw_dists, vocablens = \
            emb.main(corpora, save_path=save_path,
                 lang='english', size=300, window=5, min_count=1, workers=8,
                 epochs=20, max_vocab_size=None, n_neighbors=10,
                 models='train', plot=False)

        # Save results
        with open(os.path.join(folder, 'metrics.json'), 'w') as f:
            json.dump({'order_locals': order_locals,
                      'avg_speeds': avg_speeds,
                      'avg_pw_dists': avg_pw_dists,
                      'vocab_lens': vocablens}, f)


def main():
    nltk_permutations()


if __name__ == '__main__':
    argh.dispatch_command(main)
