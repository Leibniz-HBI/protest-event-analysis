import argparse
import logging
import nltk
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from syntok.tokenizer import Tokenizer

from flair.models import MultitaskModel
from flair.data import Sentence
from flair.datasets import ColumnCorpus

logging.basicConfig(level=logging.INFO)

# set doc_sep_token for flair dataset format
doc_sep_token = "-DOCSTART-"

# convert fulltexts to one token per row table format
def extract_tokens(document_df, language="german"):
    assert "text" in document_df.columns, "Expecting a column 'text' in document_df"
    assert "doc_id" in document_df.columns, "Expecting a column 'doc_id' in document_df"
    set_categories = []
    tok = Tokenizer(emit_hyphen_or_underscore_sep = True, replace_not_contraction = False)
    all_tokens = []
    for doc_i, (index, row) in enumerate(document_df.iterrows()):
        sentences = nltk.sent_tokenize(row.text, language=language)
        n_toks = 0
        for sent_i, sentence in enumerate(sentences):
            tokens = tok.tokenize(sentence)
            for tok_i, token in enumerate(tokens):
                token_row = (
                    str(row.doc_id), 
                    sent_i, 
                    token.value, 
                    "ws_before=no" if not token.spacing and tok_i > 0 else "-", 
                    row.label
                ) 
                all_tokens.append(token_row)
                n_toks += 1 
    dataset = pd.DataFrame(all_tokens)
    dataset.columns = ["doc_id", "sent_id", "token", "spacing", "label"]
    return dataset

# convert token table format to CoNLL output string
def dataframe_to_conll(dataframe):
    target_cols = set(["doc_id", "sent_id", "token", "spacing", "label"])
    assert target_cols.issubset(set(dataframe.columns)), 'Expecting dataframe with columns ["doc_id", "sent_id", "token", "spacing", "label"]'
    
    # column separator to use
    sep = "\t"

    # split dataframe with docs and sentences into lists of lists
    last_doc_id = -1
    last_sent_id = -1
    current_doc = []
    current_sent = []
    all_documents = []
    for index, row in dataframe.iterrows():
        if row["doc_id"] != last_doc_id and last_doc_id != -1:
            current_doc.append(current_sent)
            current_sent = []

            all_documents.append(current_doc)
            current_doc = []
        else:
            if last_sent_id != row["sent_id"] and current_sent:
                current_doc.append(current_sent)
                current_sent = []

        token = (row["token"], row["doc_id"], str(row["sent_id"]), row["spacing"].replace("\n", "\\n"), row["label"] if row["label"] is not None else "NA")
        current_sent.append(sep.join(token))
        last_doc_id = row["doc_id"]
        last_sent_id = row["sent_id"]

    if current_sent:
        current_doc.append(current_sent)
    if current_doc:
        all_documents.append(current_doc)

    # create CoNLL output format data
    documents = []
    for d in all_documents:
        sentences = []
        for s in d:
            sentences.append("\n".join(s))
        documents.append("\n\n".join(sentences))

    doc_sep = "\n\n" + doc_sep_token + sep + sep.join(["NONE"] * 4) + "\n\n"

    doc_series = pd.Series(documents)

    all_content = doc_sep[2:] + doc_sep.join(doc_series.tolist())
    return all_content


# run prediction
if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Tagger for PROTEST and VARIABLES (FORM, ZAHL, CLAIM, TRAEGER, DATUM)')
    parser.add_argument('--input_csv', type=str, help='input CSV file path (Format: 2 columns "doc_id", and "text")', required=True)
    parser.add_argument('--output_csv', type=str, help='output CSV file path (Format: CoNLL-like with token tags)', required=True)
    parser.add_argument('--n', type=int, help='Number of rows to predict (default: 0). Use smaller n for testing')
    
    args = parser.parse_args()

    assert args.output_csv != args.input_csv, "Input and output CSV path must not be equal"

    # raw data for prediction
    pe_data = pd.read_csv(args.input_csv, encoding="utf8")
    if args.n:
        pe_data = pe_data.iloc[:args.n]

    # check format
    assert set(["doc_id", "text"]).issubset(pe_data.columns), "Input CSV must have at least two columns: doc_id, and text"
    
    # set label column if it not exists
    if "label" not in pe_data.columns:
        pe_data["label"] = None
    
    # convert fulltexts
    token_df = extract_tokens(pe_data)
    conll_data = dataframe_to_conll(token_df)
    
    # write converted data to temporary to disk (flair requires it)
    path = "tmp_predict_corpus"
    # create the new folder
    Path(path).mkdir(exist_ok = True)
    with open(path + "/test.csv", "w") as f:
        f.write(conll_data)
    
    # load the temporary corpus
    corpus = ColumnCorpus(
      path, 
      {0: "text", 4: "label"},
      document_separator_token=doc_sep_token,
    )
    
    # load flair multitask model and preict
    model = MultitaskModel.load("models/task-B-C/best-model.pt")
    model.predict(corpus.test, verbose=True)
    
    # extract all non doc_sep_tokens
    predicted_tags = []
    for s in tqdm(corpus.test):
        if (s.text != doc_sep_token):
            
            # expand "PROTEST" label to entire sentences
            contains_protest = "NONE"
            for t in s:
                if t.get_labels("protest")[0].value == "PROTEST":
                    contains_protest = "PROTEST"
            
            # keep two output labels
            for t in s:
                predicted_tags.append([contains_protest, t.get_labels("variables")[0].value])
    
    predicted_tags = pd.DataFrame(predicted_tags)
    predicted_tags.columns = ["protest", "variable"]
    output_df = pd.concat([token_df, predicted_tags], axis=1)
    output_df.to_csv(args.output_csv)
