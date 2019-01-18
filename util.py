import json
import os


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def read_jl(path):
    articles = []
    for line in open(path, "r"):
        article = json.loads(line)
        articles.append(article)
    return articles


def roundl(l, n=2):
    return [round(x, n) for x in l]


def subfix_filename(filename, subfix, separator='_'):
    fn, ext = filename.split('.')
    return fn + separator + str(subfix) + '.' + ext
