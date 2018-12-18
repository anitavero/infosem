from wordcloud import WordCloud
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from gensim.models import Word2Vec
from tqdm import tqdm
import util
import argh
from argh import arg
import os
import numpy as np

import text_process as tp


def plt_wordcloud(text, lang='hungarian', animated=True):
    sw = tp.stopwords_lang[lang]
    wordcloud = WordCloud(stopwords=sw).generate(text)
    return plt.imshow(wordcloud, interpolation="bilinear", animated=animated)


def animate_wordclouds(text_dict_items, lang='hungarian', interval=200, repeat_delay=1000, save_name=None,
                       url_filter_ptrn=None):
    """
    Animates a wordcloud sequence.
    :param text_dict_items: sorted pairs by key (e.g date)
    :param lang: language of the corpus (nltk)
    :param interval: animation interval
    :param repeat_delay: animation repeat delay
    :param save_name: file path where we save the video. If False it doesn't get saved. (Default False)
    :return:
    """
    fig = plt.figure()
    ims = []
    for key, text in tqdm(text_dict_items):
        title = plt.text(170, -4, key)
        if url_filter_ptrn:
            text = tp.replace_links(text, url_filter_ptrn)
        im = plt_wordcloud(text, lang=lang, animated=True)
        ims.append([im, title])

    ani = animation.ArtistAnimation(fig, ims, interval=interval, blit=False,
                                    repeat_delay=repeat_delay)
    plt.axis('off')

    if save_name:
        ani.save("{}.mp4".format(save_name), bitrate=1000)

    plt.show()


def plot_num_per_months(date_data_dict, labelfreq):
    """Plot article number per months.
    :param date_data_dict: {date: data list} dict
    """
    plot_bar([(d, len(t)) for d, t in date_data_dict.items()], labelfreq)


def sent_len_hist(data):
    """Plot sentence length histogram.
    :param data: dict
    """
    plt.hist([len(t.split('.')) for t in tp.get_articles(data)], bins=200)
    plt.xlabel('Sentence num')
    plt.ylabel('Number of articles')
    plt.title('Sentence number distribution')
    plt.show()


def common_word_hist(data, data_type, lang, word_num=100):
    word_hist = tp.corpus_hist(data, data_type, lang)
    plt.xlabel('Words')
    plt.ylabel('Number of words')
    plt.title('Most common words')
    plot_bar(word_hist.most_common(word_num), 1)


def plot_facebook_msg_hist(msg_data, labelfreq=2):
    """Plot a message histogram using a bar."""
    msgcnt_hist = tp.facebook_msg_hist(msg_data)
    plot_bar(msgcnt_hist, labelfreq)


def plot_bar(key_value_list, labelfreq):
    """
    Plots a bar of a (key, value) list.
    :param key_value_list: list of (str, int)
    """
    x, y = list(zip(*key_value_list))
    plt.bar(x, y)
    plt.xticks(rotation=70)
    plt.xticks(x, [x[i] if i % labelfreq == 0 else '' for i in range(len(x))])
    plt.show()


################################
########## Embeddings ##########
################################

def tensorboard_emb(model, model_name, output_path):
    """
    Visualise gensim model using TensorBoard.
    Code from: https://gist.github.com/BrikerMan/7bd4e4bd0a00ac9076986148afc06507
    :param model: trained gensim model
    :param output_path: str, directory
    """
    file_name = "{}_metadata".format(model_name)
    meta_file = "{}.tsv".format(file_name)
    placeholder = np.zeros((len(model.wv.index2word), 100))

    with open(os.path.join(output_path, meta_file), 'wb') as file_metadata:
        for i, word in enumerate(model.wv.index2word):
            placeholder[i] = model[word]
            # temporary solution for https://github.com/tensorflow/tensorflow/issues/9094
            if word == '':
                print("Emply Line, should replecaed by any thing else, or will cause a bug of tensorboard")
                file_metadata.write("{0}".format('<Empty Line>').encode('utf-8') + b'\n')
            else:
                file_metadata.write("{0}".format(word).encode('utf-8') + b'\n')

    # define the model without training
    sess = tf.InteractiveSession()

    embedding = tf.Variable(placeholder, trainable=False, name=file_name)
    tf.global_variables_initializer().run()

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(output_path, sess.graph)

    # adding into projector
    config = projector.ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = file_name
    embed.metadata_path = meta_file

    # Specify the width and height of a single thumbnail.
    projector.visualize_embeddings(writer, config)
    saver.save(sess, os.path.join(output_path,'{}.ckpt'.format(file_name)))
    print('Run `tensorboard --logdir={0}` to run visualize result on tensorboard'.format(output_path))



@arg('--action', choices=['wc_animation',
                          'month_freq_bar',
                          'word_hist',
                          'fb_msg_hist',
                          'embedding'])
def main(source, data_path=None, save_name=None, interval=3000, url_filter_ptrn='',
         data_type='article', action='wc_animation', lang='hungarian',
         tn_dir='tnboard_data'):

    if source == 'news':
        if action != 'embedding':
            if not data_path:
                data_path = '444.jl'
            data = util.read_jl(data_path)
            data.sort(key=lambda x: x['date'])

        if action == 'wc_animation':
            news_per_month = tp.data_per_month(data, data_type=data_type, concat=True)
            animate_wordclouds(sorted(news_per_month.items(), key=lambda x: x[0]), interval=interval,
                               save_name=save_name)
        elif action == 'month_freq_bar':
            news_per_month = tp.data_per_month(data, data_type=data_type, concat=False)
            plot_num_per_months(news_per_month, labelfreq=2)
        elif action == 'word_hist':
            common_word_hist(data, 'article', lang, 70)
        elif action == 'embedding':
            model = Word2Vec.load(data_path)
            tensorboard_emb(model, save_name, tn_dir)


    elif source in ['fb', 'slack']:
        # Facebook/Slack messages
        if source == 'fb':
            if not data_path:
                data_path = '/Users/anitavero/projects/data/facebook_jk'
            data = tp.read_facebook_jsons(data_path)
            if action == 'fb_msg_hist':
                plot_facebook_msg_hist(data, labelfreq=20)
                return
            else:
                daily_messages = tp.faceboook_msg_per_day(data)
        elif source == 'slack':
            if not data_path:
                data_path = '/Users/anitavero/projects/data/Artificial General Emotional Intelligence Slack export Feb 17 2018 - Dec 10 2018'
            daily_messages = tp.slack_msg_per_day(data_path)

        animate_wordclouds(sorted(daily_messages.items(), key=lambda x: x[0]), lang='hunglish', interval=interval,
                           url_filter_ptrn='|http|www|com|org|hu' + url_filter_ptrn, save_name=save_name)


if __name__ == '__main__':
    argh.dispatch_command(main)
