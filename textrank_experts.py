import spacy.cli
spacy.cli.download("en_core_web_sm")
import pytextrank
import spacy
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()
import spacy
from icecream import ic
import networkx as nx
#%matplotlib inline
import matplotlib.pyplot as plt

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 1700000
nlp.add_pipe("textrank")

directory = "2.SeptemberExpertsCleaned.csv"
df = pd.read_csv(directory)

listofTweets=df['TweetTextProcessed'].tolist()

newlist=[]
for i in listofTweets:
    if str(i)!='nan':
      txt=i.replace('rt','')
      txt=txt.replace('amp','')
      txt=txt+"."
      newlist.append(txt)

str_listoftweets = ' '.join([str(elem) for elem in newlist])
print('length',len(str_listoftweets))
doc = nlp(str_listoftweets)

tr = doc._.textrank

def increment_edge (graph, node0, node1):
    ic(node0, node1)

    if graph.has_edge(node0, node1):
        graph[node0][node1]["weight"] += 1.0
    else:
        graph.add_edge(node0, node1, weight=1.0)

POS_KEPT = ["NUM", "NOUN", "PROPN", "VERB"]

def link_sentence (doc, sent, lemma_graph, seen_lemma):
    visited_tokens = []
    visited_nodes = []

    for i in range(sent.start, sent.end):
        token = doc[i]
        print('token', token)

        if token.pos_ in POS_KEPT:
            key = (token.lemma_, token.pos_)
            print('key', key)

            if key not in seen_lemma:
                seen_lemma[key] = set([token.i])
            else:
                seen_lemma[key].add(token.i)

            node_id = list(seen_lemma.keys()).index(key)

            if not node_id in lemma_graph:
                lemma_graph.add_node(node_id)

            ic(visited_tokens, visited_nodes)
            ic(list(range(len(visited_tokens) - 1, -1, -1)))

            for prev_token in range(len(visited_tokens) - 1, -1, -1):
                ic(prev_token, (token.i - visited_tokens[prev_token]))

                if (token.i - visited_tokens[prev_token]) <= 3:
                    increment_edge(lemma_graph, node_id, visited_nodes[prev_token])
                else:
                    break

            ic(token.i, token.text, token.lemma_, token.pos_, visited_tokens, visited_nodes)

            visited_tokens.append(token.i)
            visited_nodes.append(node_id)

lemma_graph = nx.Graph()
seen_lemma = {}

for sent in doc.sents:
    link_sentence(doc, sent, lemma_graph, seen_lemma)
    #break # only test one sentence

labels = {}
keys = list(seen_lemma.keys())

for i in range(len(seen_lemma)):
    labels[i] = keys[i][0].lower()

fig = plt.figure(figsize=(15, 15))
pos = nx.spring_layout(lemma_graph)

nx.draw(lemma_graph, pos=pos, with_labels=False, font_weight="bold")
nx.draw_networkx_labels(lemma_graph, pos, labels);
#plt.savefig("april_experts_textrank.png", format="PNG")

ranks = nx.pagerank(lemma_graph)

data = pd.DataFrame()
for node_id, rank in sorted(ranks.items(), key=lambda x: x[1], reverse=True):
    ic(node_id, rank, labels[node_id])
    data = data.append({'node_id':[node_id], 'rank':[rank], 'labels':[labels[node_id]]}, ignore_index=True)

data.to_csv("SeptemberTextrankExperts.csv")

#Define a function to collect the top-ranked phrases from the lemma graph.
import math

def collect_phrases (chunk, phrases, counts):
    chunk_len = chunk.end - chunk.start
    sq_sum_rank = 0.0
    non_lemma = 0
    compound_key = set([])

    for i in range(chunk.start, chunk.end):
        token = doc[i]
        key = (token.lemma_, token.pos_)

        if key in seen_lemma:
            node_id = list(seen_lemma.keys()).index(key)
            rank = ranks[node_id]
            sq_sum_rank += rank
            compound_key.add(key)

            ic(token.lemma_, token.pos_, node_id, rank)
        else:
            non_lemma += 1

    # although the noun chunking is greedy, we discount the ranks using a
    # point estimate based on the number of non-lemma tokens within a phrase
    non_lemma_discount = chunk_len / (chunk_len + (2.0 * non_lemma) + 1.0)

    # use root mean square (RMS) to normalize the contributions of all the tokens
    phrase_rank = math.sqrt(sq_sum_rank / (chunk_len + non_lemma))
    phrase_rank *= non_lemma_discount

    # remove spurious punctuation
    phrase = chunk.text.lower().replace("'", "")

    # create a unique key for the the phrase based on its lemma components
    compound_key = tuple(sorted(list(compound_key)))

    if not compound_key in phrases:
        phrases[compound_key] = set([ (phrase, phrase_rank) ])
        counts[compound_key] = 1
    else:
        phrases[compound_key].add( (phrase, phrase_rank) )
        counts[compound_key] += 1

    ic(phrase_rank, chunk.text, chunk.start, chunk.end, chunk_len, counts[compound_key])

phrases = {}
counts = {}

for chunk in doc.noun_chunks:
    collect_phrases(chunk, phrases, counts)

for ent in doc.ents:
    collect_phrases(ent, phrases, counts)

#Since noun chunks can be expressed in different ways (e.g., they may have articles or prepositions), we need to find a minimum span for each phrase based on combinations of lemmas
import operator

min_phrases = {}

for compound_key, rank_tuples in phrases.items():
    l = list(rank_tuples)
    l.sort(key=operator.itemgetter(1), reverse=True)

    phrase, rank = l[0]
    count = counts[compound_key]

    min_phrases[phrase] = (rank, count)

data1 = pd.DataFrame()
for phrase, (rank, count) in sorted(min_phrases.items(), key=lambda x: x[1][0], reverse=False):
    ic(phrase, count, rank)
    data1 = data1.append({'phrase':phrase, 'count':count,'rank':rank}, ignore_index=True)

data1.to_csv("SeptemberTextRankPhrases.csv")
