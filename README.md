This is a release code for paper "Multi-Relation Extraction via A Global-Local Graph Convolutional Network".

## Environment

- Python 3.8
- Pytorch 1.4
- Spacy

## Data

train.json: [link_train](https://drive.google.com/file/d/1ukDBUeTjAAwO4ccnSONe9tTRemGcLk0L/view?usp=share_link)

test.json: [link_test](https://drive.google.com/file/d/1OEHnEuR0vvO-F_JSfjoaEJ2OZYezKaMC/view?usp=share_link)

Please place it in ./data/nyt/ and modify the parameters of train.py and eval.py.

## Quick Start

Please ensure that Spacy is installed before starting.

```
pip install -U pip setuptools wheel
pip install -U spacy
python -m spacy download en_core_web_md
```

You can also consider installing the GPU version of Spacy, please refer to official website of [Spacy](https://spacy.io/usage).

The dataset we use are NYT(public) and [TACRED](https://catalog.ldc.upenn.edu/LDC2018T24), and the data format is as follows:

```
{
"sentence": ["Not", "many", "people", "have", "cooler", "family", "closets", "to", "raid", "than", "Theodora", "Richards", ",", "the", "daughter", "of", "the", "Rolling", "Stones", "guitarist", "Keith", "Richards", "and", "the", "70", "'s", "supermodel", "Patti", "Hansen", "."], 
"subj_start": 27, 
"subj_end": 28, 
"obj_start": 10, 
"obj_end": 11, 
"relation": "/people/person/children", 
"edges": [[2, 0], [2, 1], [3, 2], [3, 6], [3, 7], [3, 9], [3, 29], [6, 4], [6, 5], [7, 8], [9, 11], [11, 10], [11, 12], [11, 14], [14, 13], [14, 15], [15, 21], [18, 17], [19, 16], [19, 18], [21, 19], [21, 20], [21, 22], [21, 26], [24, 23], [24, 25], [26, 24], [26, 28], [28, 27]]
}
~~~~~
{
"sentence": Original sentence after participle,  
"subj_start": The start position of the first entity in a relation, 
"subj_end": The end position of the first entity in a relation,
"obj_start": The start position of the second entity in a relation, 
"obj_end": The end position of the second entity in a relation, 
"relation": The relation type (including no_relation), 
"edges": The edges in adjancy matrix.
}
```

If a sentence has multiple relations, it will appear multiple times.  This preprocessing method is mainly for the convenience of training, you can also customize the preprocessing (by modifying the file in /GCN_RE/utils). 

Because TACRED is not a fully public dataset (access fee is required), we will not upload pre-processed TACRED dataset. While we will upload the pre-processing results of NYT.

```
Scripts for training and testing: 
#train
python train.py
#dev&test
python eval.py
```
