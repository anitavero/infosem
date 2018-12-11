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


DATE_FORMAT = '%Y/%m/%d'


def replace_links(text, url_patterns):
    """Filter non words from links"""
    slink = re.sub(r'/|-|_|\.' + url_patterns, ' ', text)
    return ' '.join(filter(lambda s: s.isalpha(), slink.split()))


def get_articles(data):
    return [' '.join(x['article']) for x in data if x['article'] and len(x['article'][0]) > 1]


def articles_per_month(data):
    """
    Group articles for months
    :param data:
    :return: dict of {srt: srt}: <date: concatenated text>
    """
    data.sort(key=lambda x: x['date'])
    groups = {year_month: ' '.join(get_articles(list(d))) for year_month, d in
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
    times = [timestampms_format(m['timestamp_ms']) for m in msg_data['messages']]
    msgcnt = Counter(times)
    dint = days_interval(times[-1], times[0])
    cnt = {d: 0 for d in dint}
    msgcnt.update(cnt)
    msgcnt_hist = list(msgcnt.items())
    msgcnt_hist.sort(key=lambda x: x[0])
    return msgcnt_hist


def plot_facebook_msg_hist(msg_data):
    msgcnt_hist = facebook_msg_hist(msg_data)
    x, y = list(zip(*msgcnt_hist))
    plt.bar(x, y)
    plt.xticks(rotation=70)
    plt.xticks(x, [x[i] if i % 20 == 0 else '' for i in range(len(x))])
    plt.show()


#Facebook encoding is fd up so we use a workaround from here:
# https://stackoverflow.com/questions/50008296/facebook-json-badly-encoded
def faceboook_msg_per_day(msg_data):
    day_text_dict = {}
    for m in msg_data['messages']:
        k = timestampms_format(m['timestamp_ms'])
        if k in day_text_dict:
            day_text_dict[k] += ' ' + m['content'].encode('latin1').decode('utf8')
        else:
            day_text_dict[k] = m['content'].encode('latin1').decode('utf8')
    return day_text_dict


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




