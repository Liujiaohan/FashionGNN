import gensim
import numpy as np
import os
import os.path as osp
import json



polyvore_path = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), 'data', 'Polyvore', 'raw')

filename = "filtered_training_outfits.json"

with open(osp.join(polyvore_path, filename)) as f:
    all_outfit = json.load(f)

sentences = {}
for outfit in all_outfit:
    outfit_name = outfit["set_id"]
    # print outfit["items"]
    for item in outfit["items"]:
        item_name = item["index"]
        sentences[outfit_name + '_' + str(item_name)] = item["name"].split(' ')

idx = 0
jdx = 0
# delete the word less than 3 char
for name in sentences.keys():
    while jdx < len(sentences[name]):
        if len(sentences[name][jdx]) < 3:
            del sentences[name][jdx]

        else:
            jdx += 1

sentences_list = list(sentences.values())

# word_list = []
# for idx in range(len(sentences_list)):
#    for jdx in range(len(sentences_list[idx])):
#        word_list.append(sentences_list[idx][jdx])
# print(sentences_list)
word_list = [sentences_list[idx][jdx] for idx in range(len(sentences_list)) for jdx in range(len(sentences_list[idx]))]

from collections import Counter

word_count = Counter(word_list)
need_to_del = []
for key, value in word_count.items():
    if value < 5:
        need_to_del.append(key)
for key in need_to_del:
    del word_count[key] 
word2id = dict(zip(*(word_count.keys(), range(len(word_count)))))
word_set = set(word2id.keys())
len_wordlist = len(word2id) 

print (len_wordlist)

def generate_vector(_file_name, word2id):
    with open(_file_name) as f:
        all_outfit = json.load(f)

    sentences = {}
    for outfit in all_outfit:
        outfit_name = outfit["set_id"]
        for item in outfit["items"]:
            item_name = item["index"]
            sentences[outfit_name + '_' + str(item_name)] = item["name"].split(' ')

    sentence_long = len(sentences)
    tt = 0
    for key, value in sentences.items():
        tt += 1
        if tt % 10000 == 0:
            print(tt, sentence_long)
        for word in value:
            vect = np.zeros((len_wordlist,))
            word_num = 0.
            if word in word_set:
                vect[word2id[word]] = 1.
            #     word_num += 1.
            # vect = vect / word_num

        with open(osp.join(polyvore_path,'polyvore_text_onehot_vectors', key + '.json'), 'w') as f:
            f.write(json.dumps(list(vect)))


generate_vector(osp.join(polyvore_path,filename) , word2id)
     



