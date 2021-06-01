import warnings
warnings.filterwarnings("ignore")

import spacy.cli
spacy.cli.download("en_core_web_sm")

import pytextrank
import pandas as pd
import numpy as np
import spacy
from tqdm import tqdm
tqdm.pandas()
import nltk
nltk.download('punkt')
nltk.download('wordnet')
import re
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
import preprocessor as p
from icecream import ic
from operator import itemgetter


directory = "3.AprilExpertsTopicModelling.csv"
df = pd.read_csv(directory)

df_topic1 = df[df['TopicForIndividualMonth']== 'Topic 4']


sp = spacy.load('en_core_web_sm')
all_stopwords = sp.Defaults.stop_words

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank")
#nlp = spacy.load(‘en_core_web_sm’, disable=[‘parser’, ‘ner’])

def bytes2string(text):
  encoded_string = text.encode("ascii", "ignore")
  decode_string = encoded_string.decode()
  decode_string=re.sub(r'(\\x(.){2})', '',decode_string)
  if "b’" in decode_string:
    decode_string= decode_string.replace("b'", "'", 1)
  elif 'b”' in decode_string:
    decode_string=decode_string.replace('b"', '"', 1)
  decode_string = decode_string[1:-1]
  cleaned=p.clean(decode_string)
  decode_string1=cleaned.lower()
  opt = re.sub(r'[^\w\s]','', decode_string1)
  text_tokens = word_tokenize(opt)
  tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
  filtered_sentence = (" ").join(tokens_without_sw)
  return filtered_sentence

df_topic1['text_clean'] = df_topic1['Text'].progress_apply(lambda x: bytes2string(x))
tweet_str_0 = df_topic1['text_clean'].tolist()

newlist=[]
for i in tweet_str_0:
    if str(i)!='nan':
      txt=i.replace('rt','')
      newlist.append(txt)

newlist1=[]
for i in newlist:
    if str(i)!='nan':
      txt=i.replace('amp','')
      newlist1.append(txt)

newlist2=[]
for i in newlist1:
    if str(i)!='nan':
      txt=i+"."
      newlist2.append(txt)

#print('newlist', newlist)

listoftweets_topic1 = ' '.join([str(elem) for elem in newlist2])
doc0 = nlp(listoftweets_topic1)
print('doc0', doc0)
for p in doc0._.phrases:
    ic(p.rank, p.count, p.text)
    ic(p.chunks)

sent_bounds = [ [s.start, s.end, set([])] for s in doc0.sents ]

limit_phrases = 100

phrase_id = 0
unit_vector = []

for p in doc0._.phrases:
    ic(phrase_id, p.text, p.rank)

    unit_vector.append(p.rank)

    for chunk in p.chunks:
        ic(chunk.start, chunk.end)

        for sent_start, sent_end, sent_vector in sent_bounds:
            if chunk.start >= sent_start and chunk.end <= sent_end:
                ic(sent_start, chunk.start, chunk.end, sent_end)
                sent_vector.add(phrase_id)
                break

    phrase_id += 1

    if phrase_id == limit_phrases:
        break

sum_ranks = sum(unit_vector)

unit_vector = [ rank/sum_ranks for rank in unit_vector ]

from math import sqrt

sent_rank = {}
sent_id = 0

for sent_start, sent_end, sent_vector in sent_bounds:
    ic(sent_vector)
    sum_sq = 0.0
    ic
    for phrase_id in range(len(unit_vector)):
        ic(phrase_id, unit_vector[phrase_id])

        if phrase_id not in sent_vector:
            sum_sq += unit_vector[phrase_id]**2.0

    sent_rank[sent_id] = sqrt(sum_sq)
    sent_id += 1

sorted(sent_rank.items(), key=itemgetter(1))

limit_sentences = 150

sent_text = {}
sent_id = 0

for sent in doc0.sents:
    sent_text[sent_id] = sent.text
    sent_id += 1

num_sent = 0

data0 = pd.DataFrame()
for sent_id, rank in sorted(sent_rank.items(), key=itemgetter(1)):
    ic(sent_id, sent_text[sent_id])
    num_sent += 1

    data0 = data0.append({'sentence_id':sent_id, 'rank':rank, 'sentence_text':sent_text[sent_id]}, ignore_index=True)

    if num_sent == limit_sentences:
        break

data0.to_csv("28.AprilTopic4ExtractiveSummary.csv")
