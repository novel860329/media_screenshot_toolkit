import difflib
import csv
import re
import pandas as pd

keyword_filename = 'news_keyword3.txt'
split_filename = 'third_split_U16.csv'

def load_keywords(keyword_filename):
    fi = open(keyword_filename, 'r', encoding="utf-8")
    lines = fi.readlines()
    #print(lines)
    medias = ""
    for i,line in enumerate(lines):
#        print(line)
        if i==0:
            medias += line[0:-1]
        else:
            medias += '|'
            medias += line[0:-1]
    return medias

def cut_compare(posts, medias, split_filename):
    split_pd = pd.read_csv(split_filename)
#    print(split_pd)
    texts = []
    text_len = 0
    for post in posts:
        post_detail = post.split('_')
#        print(post_detail)
        text = split_pd[(split_pd.qid==int(post_detail[0])) & (split_pd.images==post_detail[1]) & (split_pd.pid==int(post_detail[2]))].context
        texts.append(text)
    
    result = max(text, key=len)
    match = re.search(medias, result)
    if match:
        news = 1
        keyword = match.group(0)
    else:
        news = 0
        keyword = 0
    
    return result, news, keyword
"""
def concat_compare(posts, media, split_filename):
    split_pd = pd.read_csv(split_filename)
    texts = []
    for post in posts:
        post_detail = post.split('_')
        text = split_pd[(split_pd.qid==int(post_detail[0])) & (split_pd.images==post_detail[1]) & (split_pd.pid==int(post_detail[2]))].context
        texts.append(text)
"""

#medias = load_keywords(keyword_filename)
#cut_compare(['4_1615227696000-2021-03-09-02-21-36-57-Facebook.jpg_0','4_1615227697000-2021-03-09-02-21-37-57-Facebook.jpg_0'] ,medias , split_filename)