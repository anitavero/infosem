import json


def read_jl(path):
    articles = []
    for line in open(path, "r"):
        article = json.loads(line)
        articles.append(article)
    return articles