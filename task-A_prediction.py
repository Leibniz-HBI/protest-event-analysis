from collections import defaultdict
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from flair.data import Sentence
from flair.models import SequenceTagger

from simpletransformers.classification import ClassificationModel
import argparse
import pandas as pd
import logging
import sklearn
from sklearn.model_selection import train_test_split

from nltk.tokenize import sent_tokenize
import re
import pickle
import numpy as np
import scipy

from utils import reformat_df

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='PE relevance prediction')
    parser.add_argument('--input_csv', type=str, help='input CSV file path', required=True)
    parser.add_argument('--output_csv', type=str, help='output CSV file path', required=True)
    parser.add_argument('--n', type=int, help='Number of rows to predict (default: 0). Use smaller n for testing')
    
    args = parser.parse_args()

    assert args.output_csv != args.input_csv, "Input and output CSV path must not be equal"

    # raw data for prediction
    raw_df = pd.read_csv(args.input_csv, encoding="utf8")
    if args.n:
        raw_df = raw_df.iloc[:args.n]

    predict_df = reformat_df(raw_df, filter_size=-1)["text_a"].tolist()
    # import pdb; pdb.set_trace()

    # start prediction
    model_type = "bert"
    model_dir = "models/task-A/"

    model = ClassificationModel(model_type, model_dir)
    predictions, raw_outputs = model.predict(predict_df)

    class_probs = scipy.special.softmax(raw_outputs, axis = -1)

    raw_df["pred_text"] = predict_df
    raw_df["pred_label"] = predictions
    raw_df["pred_score"] = class_probs[:,1]

    raw_df.to_csv(args.output_csv)

