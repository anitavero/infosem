from wordcloud import WordCloud
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from unidecode import unidecode
import matplotlib.animation as animation
from tqdm import tqdm
import util
import argh

import text_process as tp


hun_stopwords = stopwords.words('hungarian') + \
                ['is', 'ha', 'szerintem', 'szoval', 'na', 'hat', 'kicsit', 'ugye', 'amugy']
stopwords_lang = {'hungarian': hun_stopwords, 'english': stopwords.words('english'),
                  'hunglish': hun_stopwords + stopwords.words('english') + [unidecode(w) for w in hun_stopwords]}


def plt_wordcloud(text, lang='hungarian', animated=True):
    sw = stopwords_lang[lang]
    wordcloud = WordCloud(stopwords=sw).generate(text)
    return plt.imshow(wordcloud, interpolation="bilinear", animated=animated)


def animate_wordclouds(text_dict_items, lang='hungarian', interval=200, repeat_delay=1000, save_name=None,
                       url_patterns=None):
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
        if url_patterns:
            text = tp.replace_links(text, url_patterns)
        im = plt_wordcloud(text, lang=lang, animated=True)
        ims.append([im, title])

    ani = animation.ArtistAnimation(fig, ims, interval=interval, blit=False,
                                    repeat_delay=repeat_delay)
    plt.axis('off')

    if save_name:
        ani.save("{}.mp4".format(save_name), bitrate=1000)

    plt.show()


def main(source, data_path=None, save_name=None, interval=3000):
    if source == 'news':
        if not data_path:
            data_path = '444.jl'
        data = util.read_jl(data_path)
        data.sort(key=lambda x: x['date'])
        # data = data[-50:]
        # articles = tp.get_articles(data)
        news_per_month = tp.articles_per_month(data)
        animate_wordclouds(sorted(news_per_month.items(), key=lambda x: x[0]), interval=interval,
                           save_name=save_name)

    elif source in ['fb', 'slack']:
        # Facebook/Slack messages
        if source == 'fb':
            if not data_path:
                data_path = '/Users/anitavero/projects/data/facebook_jk'
            data = tp.read_facebook_jsons(data_path)
            daily_messages = tp.faceboook_msg_per_day(data)
        elif source == 'slack':
            if not data_path:
                data_path = '/Users/anitavero/projects/data/Artificial General Emotional Intelligence Slack export Feb 17 2018 - Dec 10 2018'
            daily_messages = tp.slack_msg_per_day(data_path)

        animate_wordclouds(sorted(daily_messages.items(), key=lambda x: x[0]), lang='hunglish', interval=interval,
                           url_patterns='|http|www|com|org|hu', save_name=save_name)


if __name__ == '__main__':
    argh.dispatch_command(main)
