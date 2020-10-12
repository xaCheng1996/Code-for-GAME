#"D:\data/tacred\data\json/train.json"
#使用spacy处理TACRED, 主要是为了省时间
"""
JSON sample of TACRED
[{"id": "e7798fb926b9403cfcd2",
"docid": "APW_ENG_20101103.0539",

"relation": "per:title",
"token": ["At", "the", "same", "time", ",", "Chief", "Financial", "Officer", "Douglas", "Flint", "will", "become", "chairman",
",", "succeeding", "Stephen", "Green", "who", "is", "leaving", "to", "take", "a", "government", "job", "."],

"subj_start": 8,
"subj_end": 9,
"obj_start": 12,
"obj_end": 12,
"subj_type": "PERSON",
"obj_type": "TITLE",

"stanford_pos": ["IN", "DT", "JJ", "NN", ",", "NNP", "NNP", "NNP", "NNP", "NNP", "MD", "VB", "NN", ",", "VBG",
"NNP", "NNP", "WP", "VBZ", "VBG", "TO", "VB", "DT", "NN", "NN", "."],

"stanford_ner": ["O", "O", "O", "O", "O", "O", "O", "O", "PERSON", "PERSON", "O", "O", "O", "O", "O", "PERSON", "PERSON",
"O", "O", "O", "O", "O", "O", "O", "O", "O"],
"stanford_head": [4, 4, 4, 12, 12, 10, 10, 10, 10, 12, 12, 0, 12, 12, 12, 17, 15, 20, 20, 17, 22, 20, 25, 25, 22, 12],

"stanford_deprel": ["case", "det", "amod", "nmod", "punct", "compound", "compound", "compound",
"compound", "nsubj", "aux", "ROOT", "xcomp", "punct", "xcomp", "compound", "dobj", "nsubj", "aux",
"acl:relcl", "mark", "xcomp", "det", "compound", "dobj", "punct"]},
"""

import spacy
import json
import re
import numpy as np

parser = spacy.load('en_core_web_md')

_pad_word = 'word'
default_vector = np.array(parser(_pad_word)[0].vector, dtype=np.float64)

word_substitutions = {'-LRB-': '(',
                      '-RRB-': ')',
                      '``': '"',
                      "''": '"',
                      "--": '-',
                      }

def vector_is_empty(vector):
    to_throw = 0
    for item in vector:
        if item == 0.0:
            to_throw += 1
    if to_throw == len(vector):
        return True
    return False

def clean_word(word):
    word = word
    if word in word_substitutions:
        word = word_substitutions[word]
    word = re.sub(r'\+([a-zA-Z])', r'\1', word)
    word = re.sub(r'\=([a-zA-Z])', r'\1', word)
    word = re.sub(r'([a-zA-Z]+)_([a-zA-Z]+)', r' \1-\2 ', word)
    return word

def get_clean_word_vector(full_Sentence_par):
    sentence_vector = []
    for i in full_Sentence_par:
        vector = i.vector
        if vector_is_empty(i.vector):
            vector = default_vector
        sentence_vector.append(vector)
    return np.array(sentence_vector).tolist()

with open("D:/data/tacred/data/json/test.json", "r", encoding="utf-8") as train_json:
    data_input = json.load(train_json)
    #使用Spacy的话会使句子分词出现问题, 从而影响实体位置, 因此要统计实体位置错误的句子以调整.
    cnt = 0
    cnt_wrong = 0
    relation_type = dict()
    taggg = 0
    with open("./test.json", "a", encoding="utf-8") as wt:
        wt.write('[')
    for i in data_input:
        if i['relation'] == 'no_relation': continue
        # if cnt >= 30: break
        ifwrong = 0

        sentence = ""
        subj_start_spacy = 0
        subj_end_spacy = 0
        obj_start_spacy = 0
        obj_end_spacy = 0

        sub_start = i['subj_start']
        sub_end = i['subj_end']
        obj_start = i['obj_start']
        obj_end = i['obj_end']
        token = i['token']

        #token组成句子, 然后用spacy解析一下
        for word in i['token']:
            sentence += word
            sentence += " "
        sentence = sentence.strip()
        sentence_par = parser(sentence)

        #原数据集里实体位置的修正, 具体做法是在spacy分割的句子里面找实体代表的单词, 然后修改位置
        #但是仍然会有一些错误, 斯坦福所认定的一些专有名词Spacy认不出来, 这也是没办法的事, 只能放弃了这些数据了. 统计上这些数据
        #到也不多. 大概有个1/25, 不影响结果.
        entity_cnt = 0
        for word in sentence_par:
            word = str(word)
            # print(word)
            if word == token[sub_start]:
                subj_start_spacy = entity_cnt
            if word == token[sub_end]:
                subj_end_spacy = entity_cnt
            if word == token[obj_start]:
                obj_start_spacy = entity_cnt
            if word == token[obj_end]:
                obj_end_spacy = entity_cnt
            entity_cnt += 1

        if str(sentence_par[subj_start_spacy]) != str(token[sub_start]) or str(sentence_par[subj_end_spacy]) != str(token[sub_end]) \
                or str(sentence_par[obj_start_spacy]) != str(token[obj_start]) or str(sentence_par[obj_end_spacy]) != str(token[obj_end]):
            # print(str(sentence_par[subj_start_spacy]) + " " + str(token[sub_start]))
            # print(str(sentence_par[subj_end_spacy]) + " " + str(token[sub_end]))
            # print(str(sentence_par[obj_start_spacy]) + " " + str(token[obj_start]))
            # print(str(sentence_par[obj_end_spacy]) + " " + str(token[obj_end]))
            cnt_wrong += 1
            continue
            # print(sentence_par)
            # print(token)
        # print(sentence_par[0])
        sentence_par_list = list(clean_word(str(word)) for word in sentence_par)
        # print(sentence_par_list)
        cnt += 1
        if cnt % 100 == 0:
            print(str(cnt) + "-" + str(cnt_wrong))
        # if cnt > 500:
        #     break
        data_process = {
            'sentence': sentence_par_list,
            'subj_start': subj_start_spacy,
            'subj_end': subj_end_spacy,
            'obj_start': obj_start_spacy,
            'obj_end': obj_end_spacy,
            'relation': i['relation']
        }
        if relation_type.get(i['relation']) is None:

            relation_type[i['relation']] = 1
        # data_json = json.dumps(data_process)
        with open("./test.json", "a", encoding="utf-8") as wt:
            if taggg == 1:
                wt.write(',')
            else: taggg = 1
            json.dump(data_process, wt)
            # wt.write(data_json)
    with open("./test.json", "a", encoding="utf-8") as wt:
        wt.write(']')
    print(relation_type)
    print(len(relation_type))
    print(cnt)
    print(cnt_wrong)