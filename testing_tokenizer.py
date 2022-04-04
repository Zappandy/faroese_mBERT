from transformers import BertTokenizer, BertModel
from tokenizers import trainers
import torch
from io import open
from conllu import parse_incr
import pandas as pd
import re

corpus_file = "fao_wikipedia_2021_30K-sentences.txt"
f = open(corpus_file, 'r', encoding="utf-8")
faroeseRegex = re.compile(r"^\d+\s+")
faroese_sents = [faroeseRegex.sub('', sent) for sent in f.readlines()]  # for faroese
faroese_words = [sent.split() for sent in faroese_sents]
punc_tokens = ['“', '”', '´', '`', '–', '‐', '’', '‘', '—', '…'] 
faroese_words = [word for sent in faroese_words for word in sent]
f.close()
str_corpus = "".join(faroese_sents)
leipzig_corpus = pd.read_csv(corpus_file, delimiter='\t', header=None)
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", do_lower_case=False)
tokenizer.add_tokens(punc_tokens)
#bert_model = BertModel.from_pretrained("bert-base-multilingual-cased")

unk_tokens = dict()
for w in faroese_words: 
    single_tokens = tokenizer.tokenize(w)
    if "[UNK]" in single_tokens:
        unk_tokens[w] = single_tokens
print(unk_tokens)

#bert_vocab = tokenizer.get_vocab().keys()  
#with open("bert_vocabulary.txt", 'w') as f:
#    for token in bert_vocab:
#        f.write(token + '\n')
#
text = tokenizer.tokenize(str_corpus)
final_unks = []
for i,  w in enumerate(text):
    if 'UNK' in w:
        final_unks.append((text[i-1], text[i+1]))
print(final_unks)

leipzig_corpus["tokenized"] = leipzig_corpus.iloc[:, 1].map(tokenizer.tokenize)
#sample_sent = tokenizer.convert_tokens_to_ids(leipzig_corpus.iloc[1, 1])

#leipzig_corpus.loc[1, "tokenized"].append("TITO_@323")  # adding token to test new token
#leipzig_corpus.iloc[1, 1] = leipzig_corpus.iloc[1, 1] + ["TITO_@323"]  # tokenized col!
#new_tokens = [word for sent in leipzig_corpus["tokenized"].tolist() for word in sent]
for sent in leipzig_corpus["tokenized"]:
    if "[UNK]" in sent:
        print(sent)
new_vocab = tokenizer.get_vocab().keys()
#with open("test_vocab.txt", "w") as f:
#    for token in new_vocab:
#        f.write(token + '\n')

#model.resize_token_embeddings(len(tokenizer))
#updated_tokenizer = pre_trained_tokenizer.train(
#          technical_text,
#            initial_vocabulary=vocab
#            )
#
#new_vocab = updated_tokenizer.get_vocab()  # 'new_vocab' contains all words in 'vocab' plus some new words

# AFTER SEP TOKENS, we see the augmented tokens
# Standardize numbers to unicodes and/or ASCII

raise SystemExit


# Do we need this encoding for pre-training?
# 99 tokens
encoded = tz.encode_plus(add_special_tokens=True,  # Add [CLS] and [SEP]
                         max_length = 64,  # maximum length of a sentence
                         pad_to_max_length=True,  # Add [PAD]s
                         return_attention_mask = True,  # Generate the attention mask
                         return_tensors = 'pt',  # ask the function to return PyTorch tensors
                         )

# language modeling from hugging face. 
class fineBERT(torch.nn.Module):

    def __init__(self):
        print("testing")

eval_file = open("UD_Faroese-OFT/fo_oft-ud-test.conllu", "r", encoding="utf-8")  # https://www.youtube.com/watch?v=lvJRFMvWtFI
faroese_oft = [sent for sent in parse_incr(eval_file)]

def read_conll(input_file):
        """Reads a conllu file."""
        ids = []
        texts = []
        tags = []
        #
        text = []
        tag = []
        idx = None
        for line in open(input_file, encoding="utf-8"):
            if line.startswith("# sent_id ="):
                idx = line.strip().split()[-1]
                ids.append(idx)
            elif line.startswith("#"):
                pass
            elif line.strip() == "":
                texts.append(text)
                tags.append(tag)
                text, tag = [], []
            else:
                try:
                    splits = line.strip().split("\t")
                    token = splits[1] # the token
                    label = splits[3] # the UD POS Tag label
                    text.append(token)
                    tag.append(label)
                except:
                    print(line)
                    print(idx)
                    raise
        return ids, texts, tags
ids, texts, tags = read_conll("UD_Faroese-OFT/fo_oft-ud-test.conllu")
print(ids[-4])
print(tags[-4])
print(faroese_oft[-4])


def subword_tokenize(tokens, labels):  # Faroese to closest Icelandic
    """
    tokens: List of word tokens.
    labels: List of PoS tags.
    Returns:
    List of subword tokens.
    List of propagated tags.
    List of indexes mapping subwords to the original word.
    """

    # english bert instead of mBERT? Zero-shot in fine tuning... 
    # fine tune on Danish or closest related language to Faroese
    # downstream task would be the focus

    # NEW NOTES: mBERT with NO PRE TRAINING IN FAROESE. Just fine tuning mBERT with speech
    # tagging to Faroese. 
    # New new: no no pre-training fine tune English
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", do_lower_case=False)
    split_tokens, split_labels = [], []
    idx_map = []
    for ix, sub_token in enumerate(tokens):
        split_tokens.append(sub_token)
        split_labels.append(labels[ix])
        idx_map.append(ix)
    return split_tokens, split_labels, idx_map


all_tokens, pos_tags, idx = subword_tokenize(texts, tags)
print(all_tokens[-4])
