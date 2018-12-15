try:
    from infosem import util
except:
    import util
from itertools import groupby
from datetime import datetime, date, timedelta
from collections import Counter, defaultdict
from matplotlib import pyplot as plt
from glob import glob
import json
import os
import re
from nltk.corpus import stopwords
from unidecode import unidecode
import string


hun_stopwords = stopwords.words('hungarian') + \
                ['is', 'ha', 'szerintem', 'szoval', 'na', 'hat', 'kicsit', 'ugye', 'amugy']
stopwords_lang = {'hungarian': hun_stopwords, 'english': stopwords.words('english'),
                  'hunglish': hun_stopwords + stopwords.words('english') + [unidecode(w) for w in hun_stopwords]}

DATE_FORMAT = '%Y/%m/%d'


def replace_links(text, url_patterns):
    """Filter non words from links"""
    slink = re.sub(r'/|-|_|\.' + url_patterns, ' ', text)
    return ' '.join(filter(lambda s: s.isalnum(), slink.split()))


def get_year(date_str):
    """Get year from different date formats."""
    return int(re.findall('20\d\d', date_str)[0])


def filter_by_year(data, years):
    """Filter data by dates"""
    return filter(lambda x: x['date'] and get_year(x['date'][0].split('-')[0]) not in years, data)


def get_titles(data, filter=None):
    return [x['title'][0] for x in data if x['title']
            if not filter or filter.lower() in x['title'][0].lower()]

def get_articles(data):
    return [' '.join(x['article']) for x in data if x['article'] and len(x['article'][0]) > 1]

data_getters = {'article': get_articles, 'title': get_titles}


def get_text(data, data_type):
    return ' '.join(data_getters[data_type](data))


def corpus_hist(data, data_type, lang):
    # TODO: stemming
    text = get_text(data, data_type).lower()
    trtab = text.maketrans(string.punctuation, ''.join([' ' for i in range(len(string.punctuation))]))
    words = text.translate(trtab).split()
    words = filter(lambda w: w not in stopwords_lang[lang], words)
    return Counter(words)


def data_per_month(data, data_type, concat):
    """
    Group articles for months
    :param data: dict list (output of crawlers.spiders.crawler.py)
    :param data_getter: {get_articles | get_titles}
    :param concat: boolean, if True texts for each months are concatenated
    :return: dict of {srt: srt|list(str)}: <date: concatenated text | text list>
    """
    data.sort(key=lambda x: x['date'])
    def aggr(str_list):
        if concat:
            return ' '.join(str_list)
        return str_list
    groups = {year_month: aggr(data_getters[data_type](list(d))) for year_month, d in
             groupby(data, key=lambda x: x['date'][0][:7] if x['date'] else '') if year_month is not ''}
    return groups


def timestampms_format(ts):
    dt = datetime.fromtimestamp(ts/1000)
    return dt.strftime(DATE_FORMAT)


def days_interval(day1, day2):
    d1 = datetime.strptime(day1, DATE_FORMAT)
    d2 = datetime.strptime(day2, DATE_FORMAT)
    delta = d2 - d1
    return [(d1 + timedelta(i)).strftime(DATE_FORMAT) for i in range(delta.days + 1)]


def facebook_msg_hist(msg_data):
    """
    Returns a histogram of Facebook messages per day.
    :param msg_data: Facebook data from json.
    :return: Counter of dates
    """
    times = [timestampms_format(m['timestamp_ms']) for m in msg_data['messages']]
    msgcnt = Counter(times)
    ordered_days = sorted(msgcnt.keys())
    dint = days_interval(ordered_days[0], ordered_days[-1])
    cnt = {d: 0 for d in dint}
    msgcnt.update(cnt)
    msgcnt_hist = list(msgcnt.items())
    msgcnt_hist.sort(key=lambda x: x[0])
    return msgcnt_hist


#Facebook encoding is fd up so we use a workaround from here:
# https://stackoverflow.com/questions/50008296/facebook-json-badly-encoded
def faceboook_msg_per_day(msg_data):
    """
    Concatenates messages per day.
    :param msg_data: Loaded Faceebook json.
    :return: (date, str) dict
    """
    day_text_dict = {}
    for m in msg_data['messages']:
        k = timestampms_format(m['timestamp_ms'])
        if k in day_text_dict:
            day_text_dict[k] += ' ' + m['content'].encode('latin1').decode('utf8')
        else:
            day_text_dict[k] = m['content'].encode('latin1').decode('utf8')
    return day_text_dict


def read_facebook_jsons(dir):
    """Read and combine the messages of json Facebook files in a directory."""
    data = None
    files = glob(dir + '/*')
    for file in files:
        with open(file) as f:
            new_data = json.load(f)
            if data is None:
                data = new_data
            else:
                data['messages'] += new_data['messages']
    return data


def slack_msg_per_day(datadir):
    """
    Read slack data from the slack dump directory structure and create a {day: text} dict.
    :param datadir: slack dump path
    :return: {day: text} dict, where the text is the concatenated messages on a day.
    """
    day_text_dict = {}
    channels = glob(datadir + '/*')
    for channel in channels:
        days = glob(channel + '/*')
        for day in days:
            date = os.path.split(os.path.splitext(day)[0])[1]
            with open(day) as f:
                day_data = json.load(f)
            for m in day_data:
                if date in day_text_dict:
                    day_text_dict[date] += ' ' + m['text']
                else:
                    day_text_dict[date] = m['text']
    return day_text_dict
