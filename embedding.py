from gensim.test.utils import get_tmpfile
from gensim.models import Word2Vec
import argh

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


def main(data_path, data_type, lang, save_path,
         size=100, window=5, min_count=100, workers=4, epochs=5, max_vocab_size=None):
    data = util.read_jl(data_path)
    return train(data, data_type, lang, save_path, size, window, min_count, workers, epochs,
                 max_vocab_size)


if __name__ == '__main__':
    argh.dispatch_command(main)