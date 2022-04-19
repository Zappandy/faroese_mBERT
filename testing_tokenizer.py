from transformers import BertTokenizer, BertModel
from tokenizers import trainers
from unidecode import unidecode
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
#from torchsummary import summary
import pandas as pd
from tqdm import tqdm
from conllu import parse_incr
import re

# https://towardsdatascience.com/how-to-use-bert-from-the-hugging-face-transformer-library-d373a22b0209
# https://colab.research.google.com/github/huggingface/notebooks/blob/master/transformers_doc/multilingual.ipynb
corpus_file = "fao_wikipedia_2021_30K-sentences.txt"

f = open(corpus_file, 'r', encoding="utf-8")
faroese_Regex = re.compile(r"^\d+\s+")
faroese_sents = [faroese_Regex.sub('', sent) for sent in f.readlines()]  # for faroese
punctuation = {0x2018:0x27, 0x2019:0x27, 0x201C:0x22, 0x201D:0x22, 0x2013:0x2D, 0x2010:0x2D, 0x2014:0x2D, 0x2026:0x85}
faroese_sents = [sent.translate(punctuation) for sent in faroese_sents]

faroese_words = [sent.split() for sent in faroese_sents]
faroese_words = [word for sent in faroese_words for word in sent]
f.close()
# https://neptune.ai/blog/how-to-code-bert-using-pytorch-tutorial
train_corpus = "".join(faroese_sents)
#tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", do_lower_case=False)
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
bert_model = BertModel.from_pretrained("bert-base-multilingual-cased")

unk_tokens = dict()
long_tokens = list()
for w in faroese_words: 
    single_tokens = tokenizer.tokenize(w)
    long_tokens.append((w, single_tokens))

testing = [len(subwords[1]) for subwords in long_tokens]
print(set(testing))
print(max(testing))  # 25 highest segmentations.

longest = [subwords[0] for subwords in long_tokens if len(subwords[1]) >= 11]  # try segment from 10 or higher and clean web addresses as well as () and '
# 12 subtokens == 44 words
# 11 subtokens == 96 words

bert_vocab = tokenizer.get_vocab().keys()
words_in_bert = set([word in bert_vocab for word in longest])

longest = [word for word in longest if word[:4] != "http"]
foreign_tokensRegex = re.compile(r"^(\(|'|\")")
subword_tokens = [word for word in longest if not foreign_tokensRegex.search(word)] 

#tokenizer.add_tokens(["TITO1", "TITO2"], special_tokens=True)
#tokenizer.add_special_tokens(["TITO1", "TITO2"])
print(len(tokenizer))
special_tokens_dict = {'additional_special_tokens': subword_tokens}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
print(len(tokenizer))
# RESIZE...

bert_model.resize_token_embeddings(len(tokenizer)) 

#print(tokenizer.get_vocab()[bert_vocab[10]])

#text = tokenizer.tokenize(train_corpus)
#final_unks = []
#for i,  w in enumerate(text):
#    if 'UNK' in w:
#        final_unks.append((text[i-1], text[i+1]))
#print(final_unks)
#new_vocab = tokenizer.get_vocab().keys()

#bert_model.resize_token_embeddings(len(tokenizer))

# 512 for max length?
#dataset = tokenizer.encode_plus(train_corpus, truncation=True, max_length=512, return_attention_mask=True, return_tensors="pt")
#output = bert_model(**dataset)
#print(dataset.keys())  # input_ids --> tensors

class BERTDataset(Dataset):
    def __init__(self, path, txt_file, tokenizer, max_length):
        super(BERTDataset, self).__init__()
        self.path = path
        self.train_set = pd.read_csv(txt_file, delimiter='\t', header=None, index_col=None)
        self.train_set.drop(0, inplace=True, axis=1)
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.train_set)
    def __getitem__(self, index):
        sent_1 = self.train_set.iloc[index]
        print(sent_1)
        inputs = self.tokenizer.encode_plus(sent_1, truncation=True, max_length=self.max_length, return_attention_mask=True, return_tensors="pt")
        ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]
        return {"ids": torch.tensor(ids, dtype=torch.long), "mask": torch.tensor(mask, dtype=torch.long), "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long)}

# https://colab.research.google.com/github/YuvalPeleg/transformers-workshop/blob/master/Fine_Tuning_Sentence_Classification.ipynb
corpus_file = "fao_wikipedia_2021_30K-sentences.txt"
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
dataset = BERTDataset('.', corpus_file, tokenizer, max_length=100)
dataloader = DataLoader(dataset=dataset,batch_size=32)
#print(dataloader)
#print(dataset.train_set)

#dataloader=DataLoader(dataset=dataset,batch_size=32)

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
print(texts[-4])
print(tags[-4])
print(faroese_oft[-4])

raise SystemExit


# MODELS!!! 

class fineBERT(torch.nn.Module):  # https://luv-bansal.medium.com/fine-tuning-bert-for-text-classification-in-pytorch-503d97342db2
    def __init__(self):
        super(fineBERT, self).__init__()
        self.bert_model = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.out = torch.nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        _, output = self.bert_model(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        return self.out(output)

loss_fn = torch.nn.BCEWithLogitsLoss()

model = fineBERT()
optimizer= optim.Adam(model.parameters(),lr= 0.0001)

# no pre-training  - https://colab.research.google.com/github/hybridnlp/tutorial/blob/master/01a_nlm_and_contextual_embeddings.ipynb

for param in model.bert_model.parameters():
        param.requires_grad = False
