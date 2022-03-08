from collections import defaultdict
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import sklearn

import pandas as pd

from nltk.tokenize import sent_tokenize
import re

from flair.models import SequenceTagger
from flair.data import Sentence

# protest keyword selector
protest_regex = re.compile(r'protest|versamm|demonstr|kundgebung|kampagne|soziale bewegung|hausbesetz|streik|unterschriftensammlung|hasskriminalität|unruhen|aufruhr|aufstand|boykott|riot|aktivis|widerstand|mobilisierung|petition|bürgerinitiative|bürgerbegehren|aufmarsch', re.UNICODE | re.IGNORECASE)

tagger = None

def precision_macro(y_true, y_pred):
    return sklearn.metrics.precision_score(y_true, y_pred, average='macro')
def recall_macro(y_true, y_pred):
    return sklearn.metrics.recall_score(y_true, y_pred, average='macro')
def f1_macro(y_true, y_pred):
    return sklearn.metrics.f1_score(y_true, y_pred, average='macro')

def reformat_df(data_df, filter_size = -1, replace_entities = False):
    df = data_df[['text', 'labels']].copy()
    df["labels"] = df["labels"].astype(int)

    if filter_size >= 0:
        new_docs = []
        for d in tqdm(df['text']):
            token_text = sent_tokenize(d, language='german')
            keep_s = [0] * len(token_text)
            d_contains_any_keyterm = False
            for i, s in enumerate(token_text):
                if protest_regex.search(s):
                    # keep current sentence
                    d_contains_any_keyterm = True
                    keep_s[i] = 1
                    # print(s)

                    if filter_size > 0:
                        # keep prev and next sentence
                        if i > 0:
                             keep_s[i-1] = 1
                        if i < (len(token_text) - 1):
                            keep_s[i+1] = 1

                        if filter_size == 2:
                            # keep prev-1 and next+1 sentence
                            if i > 1:
                                keep_s[i-2] = 1
                            if i < (len(token_text) - 2):
                                keep_s[i+2] = 1
            new_d = []
            if d_contains_any_keyterm:
                # make sure that headline is included, too
                keep_s[0] = 1
            for i, k in enumerate(keep_s):
                if k:
                    new_d.append(token_text[i])
            if new_d:
                new_d = " ".join(new_d)
            else:
                # keep entire document, if no protest term matches
                new_d = d
        
            # print("*************************************+")
            # print(new_d)

            new_docs.append(new_d)     
    
        df['text'] = new_docs

    if replace_entities:
        new_docs = []
        for d in tqdm(df['text']):
            new_d = replace_ne_in_doc(d, replace_entities)
            new_docs.append(new_d)
        df['text'] = new_docs

    df = df.rename(columns={"text": "text_a", "labels" : "labels"})
    return df

def replace_ne_in_doc(doc, tag_replacement):

    # import pdb; pdb.set_trace()

    global tagger
    if tagger is None:
        tagger = SequenceTagger.load('de-ner-large')

    doc = sent_tokenize(doc, language='german')
    doc = [Sentence(s) for s in doc]

    tagger.predict(doc)

    new_doc = []

    for sentence in doc:
        new_sentence = ""
        prev_t = None
        for t in sentence:
            tag = t.get_labels('ner')[0].value
            if tag.startswith("B-") or tag.startswith("S-"):
                if tag[2:] in tag_replacement:
                    this_token_text = tag_replacement[tag[2:]]
                else:
                    this_token_text = t.text
            elif tag == "O":
                this_token_text = t.text
            else:
                if tag[2:] in tag_replacement:
                    # I-tag
                    prev_t = t
                    continue
                this_token_text = t.text
            if prev_t is not None and prev_t.whitespace_after:
                this_token_text = " " + this_token_text

            #print(this_token_text, tag)

            new_sentence += this_token_text

            prev_t = t
        new_doc.append(new_sentence)
    
    return " ".join(new_doc)
