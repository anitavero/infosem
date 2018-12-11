from wordcloud import WordCloud
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from unidecode import unidecode
import matplotlib.animation as animation
from tqdm import tqdm
import json

import util
import text_process as tp


hun_stopwords = stopwords.words('hungarian') + \
                ['is', 'ha', 'szerintem', 'szoval', 'na', 'hat', 'kicsit', 'ugye', 'amugy']
stopwords_lang = {'hungarian': hun_stopwords, 'english': stopwords.words('english'),
                  'hunglish': hun_stopwords + stopwords.words('english') + [unidecode(w) for w in hun_stopwords]}


def plt_wordcloud(text, lang='hungarian', animated=True):
    sw = stopwords_lang[lang]
    wordcloud = WordCloud(stopwords=sw).generate(text)
    return plt.imshow(wordcloud, interpolation="bilinear", animated=animated)


def animate_wordclouds(text_dict_items, lang='hungarian', interval=200, repeat_delay=1000, save_name=False):
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
        im = plt_wordcloud(text, lang=lang, animated=True)
        ims.append([im, title])

    ani = animation.ArtistAnimation(fig, ims, interval=interval, blit=False,
                                    repeat_delay=repeat_delay)
    plt.axis('off')

    if save_name:
        ani.save("{}.mp4".format(save_name), bitrate=1000)

    plt.show()


if __name__ == '__main__':
    # data = util.read_jl('444.jl')
    # data.sort(key=lambda x: x['date'])
    # # data = data[-50:]
    # # articles = tp.get_articles(data)
    # news_per_month = tp.articles_per_month(data)
    # animate_wordclouds(sorted(news_per_month.items(), key=lambda x: x[0]), interval=2000)


    # Facebook/Slack messages
    source = 'slack' # 'slack' or 'fb'

    stopwords_lang['hunglish'] += ['www', 'youtube', 'https', 'http', 'com', 'watch', 'facebook']

    if source == 'fb':
        with open('/Users/anitavero/projects/data/messages/inbox/jozsefkonczer_mud106plvq/message.json') as f:
            data = json.load(f)
        daily_messages = tp.faceboook_msg_per_day(data)
    elif source == 'slack':
        daily_messages = tp.slack_msg_per_day('/Users/anitavero/projects/data/Artificial General Emotional Intelligence Slack export Feb 17 2018 - Dec 10 2018')


    animate_wordclouds(sorted(daily_messages.items(), key=lambda x: x[0]), lang='hunglish', interval=2000)
