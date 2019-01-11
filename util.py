import json


def read_jl(path):
    articles = []
    for line in open(path, "r"):
        article = json.loads(line)
        articles.append(article)
    return articles


def roundl(l, n=2):
    return [round(x, n) for x in l]