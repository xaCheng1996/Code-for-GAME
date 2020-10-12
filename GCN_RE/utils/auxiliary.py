import spacy
import re
import numpy as np
import json
import torch

parser = spacy.load('en_core_web_md')


_pad_word = 'word'

default_vector = np.array(parser(_pad_word)[0].vector, dtype=np.float64)

tags = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNS", "NNP", "NNPS", "PDT", "POS",
        "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP",
        "WP$", "WRB", '``', "''", '.', ',', ':', '-LRB-', '-RRB-']

classes = ["PERSON", "ORGANIZATION", "LOCATION"]

relation_TacRED = ['org:founded_by', 'no_relation', 'per:employee_of', 'per:cities_of_residence',
                    'per:children', 'per:title', 'per:siblings', 'per:religion', 'org:alternate_names',
                    'org:website', 'per:stateorprovinces_of_residence', 'org:member_of', 'org:top_members/employees',
                    'per:countries_of_residence', 'org:city_of_headquarters', 'org:members',
                    'org:country_of_headquarters', 'per:spouse', 'org:stateorprovince_of_headquarters',
                    'org:number_of_employees/members', 'org:parents', 'org:subsidiaries', 'per:origin',
                    'org:political/religious_affiliation', 'per:age', 'per:other_family',
                    'per:stateorprovince_of_birth', 'org:dissolved', 'per:date_of_death', 'org:shareholders',
                    'per:alternate_names', 'per:parents', 'per:schools_attended', 'per:cause_of_death',
                    'per:city_of_death', 'per:stateorprovince_of_death', 'org:founded', 'per:country_of_birth',
                    'per:date_of_birth', 'per:city_of_birth', 'per:charges', 'per:country_of_death'
                   ]

"""
{'org:founded_by': 124, 'no_relation': 55112, 'per:employee_of': 1524, 'org:alternate_names': 808, 
'per:cities_of_residence': 374, 'per:children': 211, 'per:title': 2443, 'per:siblings': 165, 'per:religion': 53, 
'per:age': 390, 'org:website': 111, 'per:stateorprovinces_of_residence': 331, 'org:member_of': 122, 
'org:top_members/employees': 1890, 'per:countries_of_residence': 445, 'org:city_of_headquarters': 382, 
'org:members': 170, 'org:country_of_headquarters': 468, 'per:spouse': 258, 
'org:stateorprovince_of_headquarters': 229, 'org:number_of_employees/members': 75, 
'org:parents': 286, 'org:subsidiaries': 296, 'per:origin': 325, 
'org:political/religious_affiliation': 105, 'per:other_family': 179, 
'per:stateorprovince_of_birth': 38, 'org:dissolved': 23, 'per:date_of_death': 134, 
'org:shareholders': 76, 'per:alternate_names': 104, 'per:parents': 152, 'per:schools_attended': 149, 
'per:cause_of_death': 117, 'per:city_of_death': 81, 'per:stateorprovince_of_death': 49, 'org:founded': 91, 
'per:country_of_birth': 28, 'per:date_of_birth': 63, 'per:city_of_birth': 65, 'per:charges': 72, 
'per:country_of_death': 6}
"""
# relation_NYT = ['/people/person/children', '/people/person/ethnicity',
# '/location/neighborhood/neighborhood_of', '/location/administrative_division/country',
# '/people/person/religion', '/location/country/administrative_divisions',
# '/business/company/place_founded', '/people/person/nationality', '/people/person/place_lived',
# '/people/deceased_person/place_of_death', '/location/location/contains',
# '/business/company/founders', '/business/person/company', '/people/person/place_of_birth',
# '/location/country/capital', '/people/ethnicity/geographic_distribution', '/sports/sports_team/location',
# '/business/company/major_shareholders', '/people/person/profession', '/business/company/advisors',
# 'NA', '/film/film/featured_film_locations', '/people/ethnicity/included_in_group',
# '/people/place_of_interment/interred_here', '/time/event/locations', '/location/de_state/capital',
# '/location/us_state/capital', '/business/company_advisor/companies_advised'
# '/people/deceased_person/place_of_burial', '/broadcast/content/location', '/location/us_county/county_seat',
# '/film/film_festival/location', '/location/it_region/capital',
# '/business/shopping_center_owner/shopping_centers_owned',
# '/location/in_state/legislative_capital', '/location/in_state/administrative_capital',
# '/business/business_location/parent_company', '/people/family/members', '/location/jp_prefecture/capital',
# '/film/film_location/featured_in_films', '/people/family/country', '/business/company/locations',
#  '/people/profession/people_with_this_profession', '/location/br_state/capital', '/location/cn_province/capital',
#  '/broadcast/producer/location', '/location/fr_region/capital', '/location/province/capital',
#  '/location/in_state/judicial_capital', '/business/shopping_center/owner', '/location/mx_state/capital']

relation_NYT = ['/location/location/contains', '/people/person/place_of_birth', '/business/person/company',
                '/people/person/place_lived', '/location/administrative_division/country',
                '/location/country/administrative_divisions', '/people/person/religion', '/people/person/nationality',
                '/people/person/children', '/location/country/capital', '/business/company/place_founded',
                '/people/deceased_person/place_of_death', '/business/company/founders',
                '/location/neighborhood/neighborhood_of', '/business/company/advisors','/people/ethnicity/geographic_distribution',
                '/sports/sports_team/location', '/sports/sports_team_location/teams', '/business/company/major_shareholders',
                '/business/company_shareholder/major_shareholder_of', '/people/person/ethnicity', '/people/ethnicity/people',
                '/people/person/profession', '/business/company/industry']

relation_Semeval = ['Component-Whole', 'Other', 'Instrument-Agency', 'Member-Collection', 'Cause-Effect',
                    'Entity-Destination', 'Content-Container', 'Message-Topic', 'Product-Producer', 'Entity-Origin']


word_substitutions = {'-LRB-': '(',
                      '-RRB-': ')',
                      '``': '"',
                      "''": '"',
                      "--": '-',
                      }



# just clean the sentence

def create_full_sentence(words):
    import re

    sentence = ' '.join(words)
    sentence = re.sub(r' (\'[a-zA-Z])', r'\1', sentence)
    sentence = re.sub(r' \'([0-9])', r' \1', sentence)
    sentence = re.sub(r' (,.)', r'\1', sentence)
    sentence = re.sub(r' " (.*) " ', r' "\1" ', sentence)
    sentence = sentence.replace('do n\'t', 'don\'t')
    sentence = sentence.replace('did n\'t', 'didn\'t')
    sentence = sentence.replace('was n\'t', 'wasn\'t')
    sentence = sentence.replace('were n\'t', 'weren\'t')
    sentence = sentence.replace('is n\'t', 'isn\'t')
    sentence = sentence.replace('are n\'t', 'aren\'t')
    sentence = sentence.replace('\' em', '\'em')
    sentence = sentence.replace('s \' ', 's \'s ')
    return sentence


def vector_is_empty(vector):
    to_throw = 0
    for item in vector:
        if item == 0.0:
            to_throw += 1
    if to_throw == len(vector):
        return True
    return False

# get vector from word with Spacy

def get_clean_word_vector(word):
    if word in word_substitutions:
        word = word_substitutions[word]
    word_vector = np.array(parser.vocab[word].vector, dtype=np.float64)
    # try:
    #     word_vector = np.array(model[word], dtype=np.float64)
    # except:
    #     word_vector = model['word']
    if vector_is_empty(word_vector):
        word_vector = default_vector
    return word_vector

def get_relation_vector(relation, relation_class):
    relation_vector = [0] * len(relation_class)
    # print(relation)
    for rel in relation:
        index = relation_class.index(rel)
        relation_vector[index] = 1
    return relation_vector


def create_graph_from_sentence_and_word_vectors(words, word_vectors, subj_start, subj_end, obj_start, obj_end, edges, entity):
    matrix_len = len(words)
    # print(words)
    # print(matrix_len)
    A_fw = np.zeros(shape=(matrix_len, matrix_len))
    A_bw = np.zeros(shape=(matrix_len, matrix_len))

    for i in range(matrix_len):
        A_fw[i][i] = 1
        A_bw[i][i] = 1

    for (word1,word2) in edges:
        if word1 >= matrix_len or word2 >= matrix_len:
            continue
        else:
            A_fw[word1][word2] = 1
            A_fw[word2][word1] = 1

    for i in entity:
        for j in entity:
            A_bw[i][j] = 1
            A_bw[j][i] = 1

    return A_fw, A_bw, word_vectors

def get_weight(relation_dic, relation):
    min_key = min(relation_dic, key=relation_dic.get)
    min_value = relation_dic[min_key]
    relation_vector = [1.] * len(relation)
    for key in relation_dic.keys():
        index = relation_TacRED.index(key)
        relation_vector[index] = 1/(relation_dic[key] / min_value)

    # max_key = max(relation_dic, key=relation_dic.get)
    # max_value = relation_dic[max_key]
    # relation_vector = [1.] * len(relation)
    # for key in relation_dic.keys():
    #     index = relation_TacRED.index(key)
    #     relation_vector[index] = max_value/relation_dic[key]
    return relation_vector

def get_all_sentence(filename, data_name):
    if data_name == 'TacRED':
        maxlength = 256
        # relation_dic = dict()
        with open(filename, 'r', encoding='utf-8') as data_input:
            sentences = []
            lines = json.load(data_input)
            for line in lines:
                sentence = line['sentence']
                subj_start = line['subj_start']
                subj_end = line['subj_end']
                obj_start = line['obj_start']
                obj_end = line['obj_end']
                relation = line['relation']
                edges = line['edges']
                entity = line['entity_mention']

                # if relation_dic.get(relation) is None:
                #     relation_dic[relation] = 1
                # else:
                #     relation_dic[relation] += 1
                relation_vector = get_relation_vector(relation, relation_TacRED)
                # print(relation_vector)
                sentence = list(sentence)

                sentences.append([sentence, subj_start, subj_end, obj_start, obj_end,
                                  relation_vector, edges, entity])
            # relation_pos_weight = get_weight(relation_dic, relation_TacRED)
            return sentences
    if data_name == 'NYT':
        maxlength = 256
        cnt = 0
        # relation_dic = dict()
        with open(filename, 'r', encoding='utf-8') as data_input:
            sentences = []
            lines = json.load(data_input)
            for line in lines:
                sentence = line['sentence']
                subj_start = line['subj_start']
                subj_end = line['subj_end']
                obj_start = line['obj_start']
                obj_end = line['obj_end']
                relation = line['relation']
                edges = line['edges']
                entity = line['entity_mention']


                # if relation_dic.get(relation) is None:
                #     relation_dic[relation] = 1
                # else:
                #     relation_dic[relation] += 1
                relation_vector = get_relation_vector(relation, relation_NYT)
                # print(relation_vector)
                sentence = list(sentence)
                if len(sentence) > maxlength:
                    cnt += 1
                    continue
                sentences.append([sentence, subj_start, subj_end, obj_start, obj_end,
                                  relation_vector, edges, entity])
                # relation_pos_weight = get_weight(relation_dic, relation_TacRED)
            print('{} sentences has been abandoned cause the length > 256'.format(cnt))
            return sentences

    if data_name == 'SemEval':
        maxlength = 256
        with open(filename, 'r', encoding='utf-8') as data_input:
            sentences = []
            lines = json.load(data_input)
            for line in lines:
                sentence = line['sentence']
                subj_start = line['subj_start']
                subj_end = line['subj_end']
                obj_start = line['obj_start']
                obj_end = line['obj_end']
                relation = line['relation']
                edges = line['edges']
                entity = line['entity_mention']


                relation_vector = get_relation_vector(relation, relation_Semeval)
                # print(relation_vector)
                sentence = list(sentence)

                sentences.append([sentence, subj_start, subj_end, obj_start, obj_end,
                                  relation_vector, edges, entity])
                return sentences
    else:
        print("no dataset! ")


def get_data_from_sentences(sentences, maxlength):
    all_data = []
    cnt = 0
    for sentence in sentences:
        word_vector = []
        word_list = sentence[0]
        subj_start = sentence[1]
        subj_end = sentence[2]
        obj_start = sentence[3]
        obj_end = sentence[4]
        relation_vector = sentence[5]
        relation_vector = [np.array(relation_vector)]
        edges = sentence[6]
        entity = sentence[7]

        # print(maxlength)
        for word in word_list:
            word_vector.append(get_clean_word_vector(word))

        if len(word_list) < maxlength:
            for i in range(maxlength - len(word_list)):
                word_list.append(_pad_word)

        if len(word_vector) < maxlength:
            for i in range(maxlength - len(word_vector)):
                word_vector.append(default_vector)

        # if cnt % 50000 == 0:
        #     print("Have processed data: " + str(cnt))
        # cnt += 1

        A_fw, A_bw, X = create_graph_from_sentence_and_word_vectors(word_list, word_vector, subj_start,
                                                                              subj_end, obj_start, obj_end, edges,entity)
        # print(A_fw.shape)
        all_data.append((A_fw, A_bw, X, relation_vector,subj_start,subj_end, obj_start, obj_end))

    return all_data
